"""Data root, manifest loading and integrity checks."""
import hashlib
import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "config"
ENV_VAR = "DYN_SVO_ROOT"
MODES = {"size": "size_only", "full": "full_md5"}


class DataIntegrityError(RuntimeError):
    """Raised when data files do not match the manifest."""


def data_root():
    """Return the data root from DYN_SVO_ROOT; no auto-detection."""
    root = os.environ.get(ENV_VAR)
    if not root:
        raise RuntimeError(ENV_VAR + " is not set")
    return Path(root)


def load_manifest(path=None):
    path = Path(path) if path else CONFIG_DIR / "data_manifest.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def md5_of(path, chunk=8 * 1024 * 1024):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def data_dir(manifest=None, root=None):
    manifest = manifest if manifest is not None else load_manifest()
    root = Path(root) if root else data_root()
    return root / manifest["data_dir"]


def data_file(exp, manifest=None, root=None):
    """Path of the CSV for experiment letter exp (e.g. 'A')."""
    manifest = manifest if manifest is not None else load_manifest()
    name = "exp" + exp + ".csv"
    if name not in manifest["files"]:
        raise KeyError("Unknown experiment: " + exp)
    return data_dir(manifest, root) / name


def verify_data(mode="size", manifest=None, root=None):
    """Check every manifest file; return 'size_only' or 'full_md5', raise on mismatch."""
    if mode not in MODES:
        raise ValueError("mode must be 'size' or 'full'")
    manifest = manifest if manifest is not None else load_manifest()
    folder = data_dir(manifest, root)
    problems = []
    for name, info in manifest["files"].items():
        fp = folder / name
        if not fp.is_file():
            problems.append(name + ": missing")
            continue
        size = fp.stat().st_size
        if size != info["bytes"]:
            problems.append(name + ": bytes " + str(size) + " != " + str(info["bytes"]))
            continue
        if mode == "full" and md5_of(fp) != info["md5"]:
            problems.append(name + ": md5 mismatch")
    if problems:
        raise DataIntegrityError("; ".join(problems))
    return MODES[mode]
