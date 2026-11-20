"""Experiment roles, CV folds and the test-set lock."""
from pathlib import Path

import yaml

from dyn_svo.data.manifest import CONFIG_DIR

ROLES = ("dev", "test", "all")


class TestSetLocked(PermissionError):
    """Raised when test experiments are requested without allow_test=True."""


def load_split(path=None):
    path = Path(path) if path else CONFIG_DIR / "data_split.yaml"
    with open(path, encoding="utf-8") as f:
        split = yaml.safe_load(f)
    validate_split(split)
    return split


def validate_split(split):
    exps = list(split["experiments"])
    dev = list(split["roles"]["dev"])
    test = list(split["roles"]["test"])
    if len(set(exps)) != len(exps):
        raise ValueError("Duplicate experiments")
    if set(dev) & set(test):
        raise ValueError("dev and test overlap: " + str(sorted(set(dev) & set(test))))
    if sorted(dev + test) != sorted(exps):
        raise ValueError("dev + test must cover all experiments exactly once")
    if split["cv"] != "leave_one_experiment_out":
        raise ValueError("Unsupported cv: " + str(split["cv"]))


def check_access(exps, allow_test=False, split=None):
    """Return True if any test experiment is in exps; raise if that is not allowed."""
    split = split if split is not None else load_split()
    test_hit = sorted(set(exps) & set(split["roles"]["test"]))
    if test_hit and not allow_test:
        raise TestSetLocked("Test experiments locked: " + ", ".join(test_hit))
    return bool(test_hit)


def experiments(role="dev", allow_test=False, split=None):
    """Experiments for a role; 'test' and 'all' need allow_test=True."""
    if role not in ROLES:
        raise ValueError("role must be one of " + str(ROLES))
    split = split if split is not None else load_split()
    exps = list(split["experiments"]) if role == "all" else list(split["roles"][role])
    check_access(exps, allow_test=allow_test, split=split)
    return exps


def cv_folds(split=None):
    """Leave-one-experiment-out folds inside dev: list of (train, val)."""
    split = split if split is not None else load_split()
    dev = list(split["roles"]["dev"])
    return [([e for e in dev if e != v], v) for v in dev]
