""" analysis variants; R5 non-receding leader; primary unchanged."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dyn_svo.config import load_config
from dyn_svo.experiments import configs as cf
from dyn_svo.experiments import runner, summary
from dyn_svo.sim import chain

P = {"v0": 15.0, "T": 1.2, "s0": 2.0, "a_max": 1.0, "b": 1.5}
POL = {"s0": 3.51, "h": 1.35, "sigma_gap": 2.76, "sigma_dv": 0.55, "sigma_a": 0.29, "sigma_u": 0.73}
THR = {"I": {"min_gap": 1.0, "speed_ratio": 0.9, "mean_gap": 30.0}}


def _ring(ids, seconds=80.0, L=260.0, rollback=None):
    """Synthetic ring as in test_experiments; rollback = (veh_id, t0, t1, metres) pushes one vehicle back."""
    n = len(ids)
    k = np.arange(int(seconds * 30))
    t = k / 30.0
    fr = lambda m: L - m * L / n + 5.0 * t + 0.5 * np.sin(0.3 * t + m)
    xs = {vid: fr(i) for i, vid in enumerate(ids)}
    vs = {vid: 5.0 + 0.15 * np.cos(0.3 * t + i) for i, vid in enumerate(ids)}
    if rollback is not None:
        vid, t0, t1, d = rollback
        bump = d * np.clip((t - t0) / (t1 - t0), 0.0, 1.0) * (t < t1 + 2.0)
        xs[vid] = xs[vid] - bump - 5.0 * np.clip(t - t0, 0.0, None) + 5.0 * np.clip(t - t1 - 2.0, 0.0, None)
        vs[vid] = np.gradient(xs[vid], t)
    rows = []
    for i, vid in enumerate(ids):
        lead = ids[(i - 1) % n]
        gap = np.mod(xs[lead] - xs[vid] - 4.5, L)
        rows.append(pd.DataFrame({"exp": "A", "veh_id": vid, "regime": "I", "k": k, "t": t, "x": xs[vid],
                                  "v": vs[vid], "a": 0.0, "length": 4.5, "leader_id": lead, "gap": gap}))
    return pd.concat(rows, ignore_index=True)


def _ctx(df):
    table = pd.DataFrame([dict(fold="loeo_A", regime="I", variant="nodelay", veh=v, **P) for v in ("1", "2", "pooled")])
    noise = pd.DataFrame([{"fold": "loeo_A", "regime": "I", "noise_sd": 0.3}])
    return {"data": {"A": df}, "table": table, "noise": noise, "thresholds": THR,
            "policy": {"I": POL}, "t_start": 30.0, "folds": {"A": "loeo_A"}}


def test_variants_config():
    cfg = load_config("variants")
    assert list(cfg) == ["primary", "r5"] and runner.variant_settings("primary") == {}
    assert runner.variant_settings("r5") == {"leader_monotone": True}
    with pytest.raises(ValueError):
        runner.variant_settings("r9")
    assert "variants" in runner.CONFIGS


def test_monotone_leader_properties():
    x = np.array([0.0, 1.0, 2.0, 1.9, 1.8, 2.0, 2.5, 3.0])
    v = np.array([1.0, 1.0, 0.5, -0.1, -0.1, 0.3, 0.5, 0.5])
    xm, vm = chain.monotone_leader(x, v)
    assert np.all(np.diff(xm) >= 0) and np.all(vm >= 0)
    assert np.array_equal(xm, [0.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.5, 3.0])
    assert np.array_equal(vm[[3, 4]], [0.0, 0.0]) and vm[5] == 0.3
    with pytest.raises(ValueError):
        chain.monotone_leader(np.array([250.0, 259.0, 1.0]), np.ones(3))


def test_build_episode_primary_unchanged_and_r5_only_touches_rollback():
    df = _ring([1, 2, 3, 4], rollback=(4, 40.0, 41.0, 0.2))
    old = chain.build_episode(df, 1, t_start_s=30.0)
    same = chain.build_episode(df, 1, t_start_s=30.0, leader_monotone=False)
    for key in old:
        assert np.array_equal(np.asarray(old[key]), np.asarray(same[key])), key
    raw = df[df["veh_id"] == 4].set_index("k").loc[old["k"]]
    assert np.array_equal(old["x_l"], raw["x"].to_numpy()) and np.array_equal(old["v_l"], raw["v"].to_numpy())
    assert np.max(np.maximum.accumulate(old["x_l"]) - old["x_l"]) > 0.15        # the rollback is replayed
    r5 = chain.build_episode(df, 1, t_start_s=30.0, leader_monotone=True)
    assert np.all(np.diff(r5["x_l"]) >= 0) and np.all(r5["v_l"] >= 0)
    for key in old:
        if key not in ("x_l", "v_l"):
            assert np.array_equal(np.asarray(old[key]), np.asarray(r5[key])), key


def test_enumerate_and_run_task_record_variant_r5_noop_without_rollback():
    tasks = runner.enumerate_tasks(["A"])
    assert all(t["variant"] == "primary" for t in tasks)
    assert all(t["variant"] == "r5" for t in runner.enumerate_tasks(["A"], variant="r5"))
    ctx = _ctx(_ring([1, 2, 3, 4]))
    ecfg = {"seeds": 2, "n_rear": 8}
    a = runner.run_task({"config": cf.av_configs()[4], "exp": "A"}, ctx, ecfg)
    b = runner.run_task({"config": cf.av_configs()[4], "exp": "A", "variant": "r5"}, ctx, ecfg)
    assert a["variant"] == "primary" and b["variant"] == "r5"
    assert a["rows"] == b["rows"]          # no rollback in this ring, so R5 changes nothing


def test_run_task_r5_changes_rows_when_leader_rolls_back():
    ctx = _ctx(_ring([1, 2, 3, 4], rollback=(4, 40.0, 41.0, 0.2)))
    ecfg = {"seeds": 1, "n_rear": 8}
    a = runner.run_task({"config": cf.HDV, "exp": "A"}, ctx, ecfg)
    b = runner.run_task({"config": cf.HDV, "exp": "A", "variant": "r5"}, ctx, ecfg)
    assert a["rows"] != b["rows"]


def test_check_variant():
    df = pd.DataFrame({"variant": ["r5", "r5"]})
    summary.check_variant(df, "r5")
    with pytest.raises(ValueError):
        summary.check_variant(df, "primary")
    with pytest.raises(ValueError):
        summary.check_variant(pd.DataFrame({"variant": ["r5", "primary"]}), "r5")
    summary.check_variant(pd.DataFrame({"x": [1]}), "primary")      # old runs without the column


def _evaluate_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "evaluate.py"
    spec = importlib.util.spec_from_file_location("evaluate_script", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("name,official,variant,ok", [
    ("eval_dev_r5_v1", True, "r5", True), ("eval_dev_v3", True, "r5", False),
    ("eval_dev_r5_v1", True, "primary", False), ("eval_dev_v3", True, "primary", True),
    ("debug_eval_r5", False, "r5", True), ("debug_x", True, "primary", False)])
def test_check_run_name_variants(name, official, variant, ok):
    ev = _evaluate_module()
    if ok:
        ev.check_run_name(name, official, variant)
    else:
        with pytest.raises(SystemExit):
            ev.check_run_name(name, official, variant)
