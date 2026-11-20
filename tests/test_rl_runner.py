"""RL training runner: task enumeration, run names, one training task end to end, resumption."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from dyn_svo import provenance as pv
from dyn_svo.calibration import runner as crun
from dyn_svo.config import load_config
from dyn_svo.control import cooperation as co
from dyn_svo.rl import runner as R
from dyn_svo.rl.policy import PolicyWeights

import test_rl_env as TE

DEV = ["A", "C", "E", "F", "G"]


def _script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "rl_train.py"
    spec = importlib.util.spec_from_file_location("rl_train", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_task_counts_and_ids():
    rl = load_config("rl")
    tune = R.enumerate_tasks("tune", DEV, rl)
    assert len(tune) == 90 and len({t["task_id"] for t in tune}) == 90
    assert {t["train_set"] for t in tune} == {"loeo_" + e for e in DEV}
    assert {t["phi_cfg"]["phi"] for t in tune} == {4 * co.PI16} and {t["seed"] for t in tune} == {0, 1, 2}
    with pytest.raises(SystemExit):
        R.enumerate_tasks("train", DEV, dict(rl, selected_hp=None))
    rl2 = dict(rl, selected_hp={"learning_rate": 3e-4, "gamma": 0.99})
    train = R.enumerate_tasks("train", DEV, rl2)
    assert len(train) == 600 and len({t["task_id"] for t in train}) == 600
    assert len({t["config"] for t in train}) == 10 and "rl_dyn_W40_lo4_hi7p5" in {t["config"] for t in train}
    assert {t["train_set"] for t in train} == {"loeo_" + e for e in DEV} | {"all_dev"}
    match = R.enumerate_tasks("match", DEV, rl2, phi_match=1.09544)
    assert len(match) == 60 and {t["phi_cfg"]["phi"] for t in match} == {1.0954}
    assert [e for e in R.training_sets(DEV)[0][1]] == ["C", "E", "F", "G"]


def test_run_name_and_filter_rules():
    mod = _script()
    ap = lambda *a: mod.argparse.Namespace(run_name=a[0], official=a[1], steps=a[2], only_configs=None,
                                           only_sets=None, seeds=None, max_tasks=None)
    mod.check_args(ap("debug_x", False, 100))
    for bad in (ap("debug_x", True, None), ap("rl_tune_v1", False, None), ap("rl_tune_v1", True, 100)):
        with pytest.raises(SystemExit):
            mod.check_args(bad)


def test_one_task_end_to_end_and_resume(tmp_path):
    slots_data = {"C": TE._ring([1, 2, 3, 4], 90.0, exp="C"), "E": TE._ring([5, 6, 7], 60.0, exp="E")}
    import pandas as pd
    table = pd.DataFrame([dict(fold="loeo_A", regime="I", variant="nodelay", veh=v, **TE.P) for v in ("pooled",)])
    noise = pd.DataFrame([{"fold": "loeo_A", "regime": "I", "noise_sd": 0.3}])
    ctx = {"data": slots_data, "table": table, "noise": noise, "thresholds": TE.THR, "policy": {"I": TE.POL},
           "phi_thr": TE.PHI_THR, "t_start": 30.0, "run_dir": str(tmp_path), "steps": 256, "window_s": (5.0, 20.0)}
    task = R._task("tune", "rl_tune_lr0.0003_g0.99", {"phi": 0.785}, "loeo_A", ["C", "E"], 0, 3e-4, 0.99)
    prov = {"commit": "x", "dirty": False, "config_md5": {"training_steps": "256"}}
    n = crun.run_shard(tmp_path, [task], ctx, prov, log=lambda s: None, task_fn=R.run_task)
    rec = json.loads((tmp_path / "tasks" / (task["task_id"] + ".json")).read_text(encoding="utf-8"))
    npz = tmp_path / rec["result"]["policy_file"]
    assert n == 1 and npz.exists() and R.md5_file(npz) == rec["result"]["policy_md5"]
    assert rec["result"]["steps"] == 256 and set(rec["result"]["rl_packages"]) == set(R.RL_PACKAGES)
    assert rec["result"]["action_map"] == "centred" and rec["result"]["normalize"] == "reward"
    assert PolicyWeights.from_npz(npz).out_w.shape == (1, 64)
    assert not list((tmp_path / "policies").glob("*.partial.npz"))
    assert crun.run_shard(tmp_path, [task], ctx, prov, log=lambda s: None, task_fn=R.run_task) == 0
    with pytest.raises(pv.ProvenanceMismatch):
        crun.run_shard(tmp_path, [task], ctx, dict(prov, config_md5={"training_steps": "512"}), log=lambda s: None,
                       task_fn=R.run_task)
