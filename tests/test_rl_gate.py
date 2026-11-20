"""Learnability gate: follow check, G4 rule, gate summary, task list, script rules."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from dyn_svo.config import load_config
from dyn_svo.rl import env as E
from dyn_svo.rl import learnability as L

import test_rl_evaluate as TV
import test_rl_policy as TP


def _script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "rl_gate.py"
    spec = importlib.util.spec_from_file_location("rl_gate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_follow_check_rows_and_zero_action(tmp_path):
    ctx = TV._ctx(tmp_path)
    rows, p99 = L.follow_check(TP._random_weights(5), "centred", ctx, "C", "loeo_C", {"phi": 0.785})
    assert len(rows) == 4 and p99 == 30.0
    assert set(rows[0]) == {"av", "policy_mean_reward", "zero_mean_reward", "speed_ratio", "last_gap"}
    assert all(np.isfinite(r["policy_mean_reward"]) for r in rows)
    assert E.action_to_accel(L.ConstantAction("centred").action(None), "centred") == 0.0
    assert E.action_to_accel(L.ConstantAction("linear").action(None), "linear") == pytest.approx(0.0, abs=1e-12)


def test_g4_and_gate_summary():
    ok = [{"speed_ratio": 0.99, "last_gap": 10.0, "policy_mean_reward": -2.0, "zero_mean_reward": -50.0}]
    assert L.g4(ok, 13.7)["pass"]
    assert not L.g4([dict(ok[0], speed_ratio=0.9)], 13.7)["pass"]
    assert not L.g4([dict(ok[0], policy_mean_reward=-60.0)], 13.7)["pass"]
    res = [{"config": "a", "g4": {"pass": True}}] * 15 + [{"config": "b", "g4": {"pass": i % 2 == 0}} for i in range(15)]
    s = L.gate_summary(res, "a")
    assert s["gate_pass"] and s["pass_counts"] == {"a": [15, 15], "b": [8, 15]} and s["pass_share_all"] == 23 / 30
    assert not L.gate_summary(res, "b")["gate_pass"] and not L.gate_summary(res[:14], "a")["gate_pass"]


def test_gate_task_list_and_rules():
    mod = _script()
    tasks = mod.gate_tasks(["A", "C", "E", "F", "G"], "rl_tune_lr0.0003_g0.98", load_config("rl"))
    assert len(tasks) == 90 and sum(t["selected"] for t in tasks) == 15
    assert all(t["fold"] == "loeo_" + t["exp"] for t in tasks)
    with pytest.raises(SystemExit):
        mod.gate_tasks(["A", "C", "E", "F", "G"], "rl_tune_nonexistent", load_config("rl"))
    ns = lambda n, o, stage="tune", sel="rl_tune_lr0.0003_g0.98": mod.argparse.Namespace(run_name=n, official=o, stage=stage, selected=sel)
    mod.check_args(ns("gate_rl_tune_v2", True))
    mod.check_args(ns("g4_rl_train_v1", True, "train", None))
    for bad in (ns("debug_x", True), ns("gate_rl_tune_v2", False), ns("gate_rl_tune_v2", True, sel=None),
                ns("g4_rl_train_v1", True, "train"), ns("g4_rl_train_v1", True, "match", None)):
        with pytest.raises(SystemExit):
            mod.check_args(bad)


def test_train_stage_tasks_and_summary():
    """ the 500 out-of-fold policies of the main training, pass shares only."""
    mod = _script()
    tasks = mod.gate_tasks(["A", "C", "E", "F", "G"], None, load_config("rl"), stage="train")
    assert len(tasks) == 500 and len({t["task_id"] for t in tasks}) == 500 and not any(t["selected"] for t in tasks)
    assert all(t["fold"] == "loeo_" + t["exp"] and t["task_id"].startswith("train__") for t in tasks)
    assert all("__all_dev__" not in t["task_id"] for t in tasks)
    g = lambda ok, sr=1.0: {"pass": ok, "mean_speed_ratio": sr, "max_last_gap": 5.0, "gap_limit": 27.4,
                            "policy_mean_reward": -2.0, "zero_mean_reward": -50.0}
    res = [{"config": "a", "exp": "A", "seed": s, "g4": g(True)} for s in range(3)]
    res += [{"config": "b", "exp": e, "seed": 0, "g4": g(e == "A", 0.7)} for e in ("A", "C")]
    s = mod.summarize(res, "train", None)
    assert s["n_policies"] == 5 and s["n_pass"] == 4 and s["pass_share_all"] == 0.8 and "gate_pass" not in s
    assert s["pass_counts"] == {"a": [3, 3], "b": [1, 2]}
    assert s["pass_counts_by_experiment"] == {"a|A": [3, 3], "b|A": [1, 1], "b|C": [0, 1]}
    assert s["not_passing"] == [{"config": "b", "exp": "C", "seed": 0, "mean_speed_ratio": 0.7, "max_last_gap": 5.0,
                                 "gap_limit": 27.4, "reward_not_below_zero_av": True}]
