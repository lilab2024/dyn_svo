"""Provenance recorded with every result: commit, DIRTY flag, config md5s, versions."""
import datetime
import platform
import socket
import subprocess
from importlib.metadata import version

from dyn_svo.config import config_md5
from dyn_svo.data.manifest import REPO_ROOT

PACKAGES = ("numpy", "pandas", "scipy", "PyYAML")


class ProvenanceMismatch(RuntimeError):
    """Raised when stored results were produced by different code or settings."""


def _git(*args):
    p = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError("git " + " ".join(args) + " failed: " + p.stderr.strip())
    return p.stdout.strip()


def git_commit():
    return _git("rev-parse", "HEAD")


def git_dirty():
    return _git("status", "--porcelain") != ""


def provenance(config_names, data_check):
    """Snapshot of code, settings and environment for a result file."""
    return {"commit": git_commit(), "dirty": git_dirty(),
            "config_md5": {n: config_md5(n) for n in config_names},
            "data_check": data_check, "python": platform.python_version(),
            "packages": {p: version(p) for p in PACKAGES}, "host": socket.gethostname(),
            "platform": platform.platform(),
            "utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")}


def compatible(a, b):
    """Results may be reused only if commit, DIRTY flag and every config md5 agree."""
    return a["commit"] == b["commit"] and a["dirty"] == b["dirty"] and a["config_md5"] == b["config_md5"]


def require_clean(prov):
    if prov["dirty"]:
        raise RuntimeError("Official runs need a clean repository (commit everything first)")
