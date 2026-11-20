import json

import numpy as np
import pytest

from dyn_svo import provenance as pv
from dyn_svo.calibration import runner
from dyn_svo.config import load_config
from dyn_svo.sim import idm

P = {"v0": 12.0, "T": 1.4, "s0": 2.2, "a_max": 0.9, "b": 1.6}
TINY = {"segment_s": 30, "bounds": load_config("calibration")["bounds"], "delay_steps_max": 3,
        "collision_penalty_m": 1000.0, "de": {"popsize": 5, "maxiter": 2, "tol": 1e-8, "seed": 0}}


def _seg(n_seg=2, n_step=40, phase=0.0):
    t = np.arange(n_step) * idm.DT
    vl = np.array([4.0 + np.sin(t + phase + i) for i in range(n_seg)])
    xl = 30.0 + np.cumsum(vl * idm.DT, axis=1)
    len_l = np.full(n_seg, 5.0)
    x0 = xl[:, 0] - len_l - 12.0
    x, v, _, gap, _ = idm.simulate_follower(xl, vl, len_l, x0, vl[:, 0], P)
    return {"x_l": xl, "v_l": vl, "x": x, "v": v, "gap": gap, "len_l": len_l,
            "k_start": np.arange(n_seg)}


def _bank():
    bank = {("A", 1): _seg(), ("A", 2): _seg(phase=1), ("C", 1): _seg(phase=2),
            ("E", 2): _seg(phase=3), ("F", 3): _seg(phase=4), ("G", 3): _seg(phase=5)}
    regimes = {"A": "I", "C": "I", "E": "I", "F": "II", "G": "II"}
    return bank, regimes


def _prov(commit="c1"):
    return {"commit": commit, "dirty": False, "config_md5": {"sim": "x"}, "data_check": "size_only"}


def test_folds_from_split():
    f = runner.folds()
    assert [x[0] for x in f] == ["loeo_A", "loeo_C", "loeo_E", "loeo_F", "loeo_G", "all_dev"]
    for name, train, held in f[:5]:
        assert held not in train and len(train) == 4
    assert f[5][1] == ["A", "C", "E", "F", "G"] and f[5][2] is None


def test_enumerate_tasks_deterministic_and_counts():
    bank, regimes = _bank()
    tasks = runner.enumerate_tasks(bank, regimes, runner.folds())
    assert tasks == runner.enumerate_tasks(bank, regimes, runner.folds())
    ids = [t["task_id"] for t in tasks]
    assert len(ids) == len(set(ids))
    all_dev = [t for t in tasks if t["fold"] == "all_dev"]
    # regime I: vehicles 1, 2 + pooled; regime II: vehicle 3 + pooled; x 2 variants
    assert len(all_dev) == (3 + 2) * 2
    loeo_f = [t for t in tasks if t["fold"] == "loeo_F" and t["regime"] == "II"]
    assert all(t["train"] == ["G"] and t["held_out"] == "F" for t in loeo_f)
    loeo_f_regime1 = [t for t in tasks if t["fold"] == "loeo_F" and t["regime"] == "I"]
    assert all(t["held_out"] is None for t in loeo_f_regime1)


def test_held_out_never_in_train():
    bank, regimes = _bank()
    for t in runner.enumerate_tasks(bank, regimes, runner.folds()):
        assert t["held_out"] not in t["train"]


def test_shards_partition_tasks():
    bank, regimes = _bank()
    tasks = runner.enumerate_tasks(bank, regimes, runner.folds())
    parts = [runner.shard(tasks, k, 3) for k in range(3)]
    ids = sorted(t["task_id"] for p in parts for t in p)
    assert ids == sorted(t["task_id"] for t in tasks)
    with pytest.raises(ValueError):
        runner.shard(tasks, 3, 3)


def test_task_segments_pooled_and_vehicle():
    bank, regimes = _bank()
    t = {"veh": "pooled", "train": ["A", "C"], "held_out": "E"}
    assert runner.task_segments(bank, t, "train")["x_l"].shape[0] == 6
    t = {"veh": "1", "train": ["A", "C"], "held_out": "E"}
    assert runner.task_segments(bank, t, "train")["x_l"].shape[0] == 4
    assert runner.task_segments(bank, t, "held") is None


def test_run_shard_writes_resumes_and_checks_provenance(tmp_path):
    bank, regimes = _bank()
    tasks = [t for t in runner.enumerate_tasks(bank, regimes, runner.folds())
             if t["fold"] == "loeo_E" and t["veh"] == "2"]
    calls = []

    def fake(task, b, cfg):
        calls.append(task["task_id"])
        return runner.run_task(task, b, cfg)

    n = runner.run_shard(tmp_path, tasks, bank, _prov(), TINY, log=lambda s: None, task_fn=fake)
    assert n == 2 and len(calls) == 2
    rec = json.loads((tmp_path / "tasks" / (tasks[0]["task_id"] + ".json")).read_text())
    assert rec["result"]["held_rmse_m"] is not None and rec["provenance"]["commit"] == "c1"
    assert not list((tmp_path / "tasks").glob("*.tmp"))
    n2 = runner.run_shard(tmp_path, tasks, bank, _prov(), TINY, log=lambda s: None, task_fn=fake)
    assert n2 == 0 and len(calls) == 2
    with pytest.raises(pv.ProvenanceMismatch):
        runner.run_shard(tmp_path, tasks, bank, _prov("c2"), TINY, log=lambda s: None)


def test_max_tasks_and_summarize(tmp_path):
    bank, regimes = _bank()
    tasks = runner.enumerate_tasks(bank, regimes, runner.folds())[:3]
    runner.run_shard(tmp_path, tasks, bank, _prov(), TINY, max_tasks=1, log=lambda s: None)
    df, missing = runner.summarize(tmp_path, n_expected=3)
    assert len(df) == 1 and missing == 2
    assert {"v0", "T", "s0", "a_max", "b", "delay_steps", "held_rmse_m", "commit"} <= set(df.columns)
    assert (tmp_path / "params.csv").exists()


def test_provenance_helpers():
    p = pv.provenance(("sim", "calibration"), "size_only")
    assert len(p["commit"]) == 40 and isinstance(p["dirty"], bool)
    assert set(p["config_md5"]) == {"sim", "calibration"}
    assert pv.compatible(p, dict(p))
    assert not pv.compatible(p, dict(p, commit="0" * 40))
    with pytest.raises(RuntimeError):
        pv.require_clean(dict(p, dirty=True))
    pv.require_clean(dict(p, dirty=False))
