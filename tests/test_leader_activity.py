"""v5 phi_t from leader speed activity."""
import math

import numpy as np
import pandas as pd
import pytest

from dyn_svo.config import load_config
from dyn_svo.control import cooperation as co
from dyn_svo.experiments import configs as cf
from dyn_svo.experiments import runner
from dyn_svo.sim import idm

P = {"v0": 15.0, "T": 1.2, "s0": 2.0, "a_max": 1.0, "b": 1.5}
POL = {"s0": 3.51, "h": 1.35, "sigma_gap": 2.76, "sigma_dv": 0.55, "sigma_a": 0.29, "sigma_u": 0.73}
THR = {"I": {"min_gap": 1.0, "speed_ratio": 0.9, "mean_gap": 30.0}}


def test_config_v5():
    c = load_config("cooperation")["leader_activity"]
    assert c["window_s"] == [10.0, 20.0, 40.0] and c["percentiles"] == [10, 90]
    assert c["phi_ranges_pi16"] == [[3.0, 7.0], [4.0, 7.5], [5.0, 8.0]]
    assert c["thresholds_run"] == "phi_thresholds_v1" and len(co.leader_activity_grid()) == 9


def test_rolling_sd_matches_numpy_ddof0():
    x = np.random.default_rng(1).normal(8.0, 1.5, 500)
    for n in (1, 3, 50):
        r = co.RollingSD(n)
        got = [r.push(v) for v in x]
        ref = [np.std(x[max(0, i - n + 1):i + 1]) for i in range(len(x))]
        assert np.allclose(got, ref, atol=1e-9)
    assert co.leader_activity([5.0, 5.0], 10.0)[0] == 0.0


def test_phi_mapping_clip():
    lo, hi = 3 * co.PI16, 7 * co.PI16
    assert co.phi_from_activity(0.1, 0.3, 1.0, lo, hi) == lo
    assert co.phi_from_activity(2.0, 0.3, 1.0, lo, hi) == hi
    assert co.phi_from_activity(0.65, 0.3, 1.0, lo, hi) == pytest.approx((lo + hi) / 2)


def test_weight_first_step_rate_limit_and_arguments():
    w = co.LeaderActivityWeight(10.0, 3 * co.PI16, 7 * co.PI16, 0.3, 1.0)
    phi0 = w.update({"v_leader": 5.0})
    assert phi0 == pytest.approx(3 * co.PI16)                 # one sample: activity 0
    t = np.arange(1, int(round(60.0 / idm.DT))) * idm.DT
    phis = [w.update({"v_leader": 5.0 + 2.0 * math.sin(2 * math.pi * tt / 5.0)}) for tt in t]
    assert np.max(np.abs(np.diff([phi0] + phis))) <= 0.1 * idm.DT + 1e-12
    assert phis[-1] == pytest.approx(7 * co.PI16)             # SD 1.41 > a_hi: phi_hi reached
    w.reset()
    assert w.phi is None
    with pytest.raises(ValueError):
        co.LeaderActivityWeight(10.0, 0.5, 0.4, 0.3, 1.0)
    with pytest.raises(ValueError):
        co.LeaderActivityWeight(10.0, 0.1, 0.4, 1.0, 1.0)


def test_threshold_folds_exclude_held_out():
    dev = ["A", "C", "E", "F", "G"]
    folds = co.threshold_folds(dev)
    assert folds["loeo_A"] == ["C", "E", "F", "G"] and folds["all_dev"] == dev
    rng = np.random.default_rng(2)
    s = {(e, "I" if e in "ACE" else "II", "10"): rng.normal(1.0, 0.2, 200) for e in dev}
    t1 = co.thresholds_from_samples(s, folds, [10, 90])
    s2 = dict(s)
    s2[("A", "I", "10")] = s[("A", "I", "10")] + 100.0          # change only A
    t2 = co.thresholds_from_samples(s2, folds, [10, 90])
    assert t1["loeo_A"] == t2["loeo_A"] and t1["all_dev"] != t2["all_dev"]
    assert t1["loeo_F"]["II"]["10"]["train"] == ["G"] and set(t1["loeo_A"]) == {"I", "II"}


def test_dynamic_configs_names_and_ties():
    dyn = [c for c in cf.av_configs() if c["family"] == "mpc_dynamic"]
    assert len(dyn) == 9 and len({c["name"] for c in dyn}) == 9
    assert dyn[0]["name"] == "dyn_W10_lo3_hi7" and dyn[-1]["name"] == "dyn_W40_lo5_hi8"
    assert min(dyn, key=lambda c: tuple(c["rank"]))["name"] == "dyn_W10_lo5_hi8"


def _ring(ids, seconds=80.0, L=260.0):
    n = len(ids)
    k = np.arange(int(seconds * 30))
    t = k / 30.0
    rows = []
    for i, vid in enumerate(ids):
        j = (i - 1) % n
        fr = lambda m: L - m * L / n + 5.0 * t + 0.5 * np.sin(0.3 * t + m)
        gap = np.mod(fr(j) - fr(i) - 4.5, L)
        rows.append(pd.DataFrame({"exp": "A", "veh_id": vid, "regime": "I", "k": k, "t": t, "x": fr(i),
                                  "v": 5.0 + 0.15 * np.cos(0.3 * t + i), "a": 0.0, "length": 4.5,
                                  "leader_id": ids[j], "gap": gap}))
    return pd.concat(rows, ignore_index=True)


def test_run_task_dynamic_uses_fold_thresholds():
    table = pd.DataFrame([dict(fold="loeo_A", regime="I", variant="nodelay", veh=v, **P) for v in ("1", "2", "pooled")])
    noise = pd.DataFrame([{"fold": "loeo_A", "regime": "I", "noise_sd": 0.3}])
    thr = {w: {"a_lo": 0.05, "a_hi": 0.2} for w in ("10", "20", "40")}
    ctx = {"data": {"A": _ring([1, 2, 3, 4])}, "table": table, "noise": noise, "thresholds": THR,
           "policy": {"I": POL}, "t_start": 30.0, "folds": {"A": "loeo_A"}, "phi_thr": {"loeo_A": {"I": thr}}}
    cfg = [c for c in cf.av_configs() if c["name"] == "dyn_W10_lo3_hi7"][0]
    out = runner.run_task({"config": cfg, "exp": "A"}, ctx, {"seeds": 1, "n_rear": 8})
    phis = [r["phi_mean"] for r in out["rows"]]
    assert len(phis) == 4 and all(3 * co.PI16 - 1e-9 <= p <= 7 * co.PI16 + 1e-9 for p in phis)
