"""RL evaluation: rewards equal the training environment, task lists, policy md5 check, selection rule."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dyn_svo.calibration import runner as crun
from dyn_svo.config import load_config
from dyn_svo.experiments import configs as cf
from dyn_svo.experiments import runner as erun
from dyn_svo.rl import env as E
from dyn_svo.rl import evaluate as V
from dyn_svo.rl import runner as R
from dyn_svo.rl.policy import make_rl_driver
from dyn_svo.sim import chain

import test_rl_env as TE
import test_rl_policy as TP

DEV = ["A", "C", "E", "F", "G"]


@pytest.mark.parametrize("phi_cfg", [{"phi": 0.6}, TE.DYN])
def test_evaluation_rewards_equal_training_environment(phi_cfg):
    slots = TE._slots()
    for s in slots:
        s["noise_sd"] = 0.0
    n = len(slots[0]["ep"]["t"])
    env = E.ChainEnv(slots, phi_cfg, seed=0, window_s=(5.0, 20.0))
    env.draw_window = lambda: (0, 0, n - 1)
    w = TP._random_weights(2)
    obs, _ = env.reset()
    rewards, trunc = [], False
    while not trunc:
        obs, r, term, trunc, info = env.step(w.action(obs))
        rewards.append(r)
    slot = slots[0]
    rec = V.RewardRecording(make_rl_driver(w, slot["policy"], E.make_phi_source(phi_cfg, slot), slot["s_min"]))
    sim = chain.simulate(slot["ep"], slot["params"], av_driver=rec, n_rear=8, noise_sd=0.0)
    assert np.array_equal(V.step_rewards(slot["ep"], sim, rec, slot["policy"]), np.array(rewards))


def test_eval_task_lists():
    rl = load_config("rl")
    tune = V.eval_tasks("tune", DEV, "criterion", rl=rl)
    assert len(tune) == 90 and {t["exp"] for t in tune} == set(DEV)
    assert all(t["train_task_id"].endswith("__loeo_" + t["exp"] + "__s%d" % t["train_seed"]) for t in tune)
    rl2 = dict(rl, selected_hp={"learning_rate": 3e-4, "gamma": 0.99})
    main = V.eval_tasks("train", DEV, "full", "r5", rl=rl2)
    assert len(main) == 505 and len({t["task_id"] for t in main}) == 505 and {t["variant"] for t in main} == {"r5"}
    assert [t["task_id"] for t in main[:5]] == ["hdv__" + e for e in DEV] and {t["mode"] for t in main} == {"full"}
    assert all(t["config"] != "hdv" for t in main[5:]) and all("train_task_id" not in t for t in main[:5])
    assert all(t["config"] != "hdv" for t in tune)                          # criterion mode: no all-HDV tasks
    with pytest.raises(ValueError):
        V.eval_tasks("tune", DEV, "other", rl=rl)


def _ctx(tmp_path):
    data = {"C": TE._ring([1, 2, 3, 4], 90.0, exp="C")}
    table = pd.DataFrame([dict(fold="loeo_C", regime="I", variant="nodelay", veh=v, **TE.P) for v in ("pooled",)])
    noise = pd.DataFrame([{"fold": "loeo_C", "regime": "I", "noise_sd": 0.3}])
    phi_thr = {"loeo_C": TE.PHI_THR["loeo_A"]}
    return {"data": data, "table": table, "noise": noise, "thresholds": TE.THR, "policy": {"I": TE.POL},
            "phi_thr": phi_thr, "t_start": 30.0, "folds": {"C": "loeo_C"}, "train_dir": str(tmp_path)}


def _fake_training(tmp_path, tid, w):
    (tmp_path / "policies").mkdir(exist_ok=True)
    (tmp_path / "tasks").mkdir(exist_ok=True)
    p = tmp_path / "policies" / (tid + ".npz")
    w.to_npz(p)
    rec = {"task": {"task_id": tid}, "provenance": {"commit": "abc1234", "dirty": False},
           "result": {"policy_file": "policies/" + p.name, "policy_md5": R.md5_file(p)}}
    (tmp_path / "tasks" / (tid + ".json")).write_text(json.dumps(rec), encoding="utf-8")
    return p


def test_run_task_modes_and_md5_check(tmp_path):
    ctx = _ctx(tmp_path)
    tid = "tune__rl_tune_lr0.0003_g0.99__loeo_C__s0"
    p = _fake_training(tmp_path, tid, TP._random_weights(3))
    base = {"task_id": "x", "config": "rl_tune_lr0.0003_g0.99", "exp": "C", "train_seed": 0, "train_task_id": tid,
            "phi_cfg": {"phi": 0.785}, "variant": "primary"}
    cfg = {"seeds": 2, "n_rear": 8}
    crit = V.run_task(dict(base, mode="criterion"), ctx, cfg)
    assert len(crit["rows"]) == 8 and set(crit["rows"][0]) == {"av", "seed", "mean_reward", "train_seed"}
    full = V.run_task(dict(base, mode="full"), ctx, cfg)
    assert "follower_sigma" in full["rows"][0] and "min_gap" in full["rows"][0]
    assert [r["mean_reward"] for r in full["rows"]] == [r["mean_reward"] for r in crit["rows"]]
    assert crit["train_commit"] == "abc1234" and crit["action_map"] == "centred"
    p.write_bytes(p.read_bytes() + b"x")
    with pytest.raises(RuntimeError):
        V.run_task(dict(base, mode="criterion"), ctx, cfg)


def test_hdv_task_equals_mpc_evaluation(tmp_path):
    ctx = _ctx(tmp_path)
    cfg = {"seeds": 2, "n_rear": 8}
    task = {"task_id": "hdv__C", "config": "hdv", "exp": "C", "variant": "r5", "mode": "full"}
    res = V.run_task(task, ctx, cfg)
    ref = erun.run_task({"task_id": "hdv__C", "config": cf.HDV, "exp": "C", "variant": "r5"}, ctx, cfg)
    assert res["rows"] == ref["rows"] and res["mode"] == "full" and res["variant"] == "r5" and len(res["rows"]) == 8
    with pytest.raises(RuntimeError):
        V.run_task(dict(task, mode="criterion"), ctx, cfg)


def test_tuning_table_and_selection_rule():
    res = []
    for lr, g, val in ((1e-4, 0.98, -2.0), (3e-4, 0.99, -1.5), (1e-4, 0.995, -1.5000001)):
        for e in ("A", "C"):
            res.append({"task": {"config": "rl_tune_lr%g_g%g" % (lr, g)},
                        "result": {"exp": e, "rows": [{"mean_reward": val}, {"mean_reward": val}]}})
    table = V.tuning_table(res)
    assert len(table) == 3 and table[0]["criterion"] == -2.0 and table[0]["n_runs"] == {"A": 2, "C": 2}
    assert V.select_hp(table) == {"learning_rate": 1e-4, "gamma": 0.995, "config": "rl_tune_lr0.0001_g0.995"}
    [t for t in table if t["config"] == "rl_tune_lr0.0001_g0.995"][0]["criterion"] = -1.6
    assert V.select_hp(table)["config"] == "rl_tune_lr0.0003_g0.99"


def test_script_argument_rules():
    path = Path(__file__).resolve().parents[1] / "scripts" / "rl_evaluate.py"
    spec = importlib.util.spec_from_file_location("rl_evaluate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ns = lambda **k: mod.argparse.Namespace(**dict(dict(run_name="eval_dev_rl_tune_v1", official=True, variant="primary",
                                                        mode="criterion", stage="tune", max_tasks=None), **k))
    mod.check_args(ns())
    for bad in (ns(run_name="debug_x"), ns(variant="r5"), ns(mode="full"), ns(stage="train"),
                ns(run_name="eval_dev_rl_v1", stage="train", mode="full", variant="r5"), ns(max_tasks=2)):
        with pytest.raises(SystemExit):
            mod.check_args(bad)
    full = dict(stage="train", mode="full", workers=0)
    mod.check_args(ns(run_name="eval_dev_r5_rl_v1", variant="r5", **full))
    mod.check_args(ns(run_name="eval_dev_rl_v1", summarize=True, **full))
    for bad in (ns(run_name="eval_dev_r5_rl_v1", **full),                      # primary name with a variant token
                ns(run_name="eval_dev_r5x_rl_v1", variant="r5", **full),       # substring is not enough
                ns(run_name="eval_dev_rl_v1", summarize=True, stage="train", mode="full", workers=4),
                ns(summarize=True, workers=0)):                                # summary only for stage train, mode full
        with pytest.raises(SystemExit):
            mod.check_args(bad)
    mod.check_args(ns(run_name="eval_dev_rl_v1", eligibility_report=True, g4_run="g4_rl_train_v1", **full))
    for bad in (ns(run_name="eval_dev_rl_v1", eligibility_report=True, **full),                      # needs --g4-run
                ns(run_name="eval_dev_rl_v1", eligibility_report=True, g4_run="g", summarize=True, **full),
                ns(eligibility_report=True, g4_run="g", workers=0)):                                   # stage train, mode full only
        with pytest.raises(SystemExit):
            mod.check_args(bad)
