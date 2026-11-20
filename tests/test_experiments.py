import math

import numpy as np
import pandas as pd
import pytest

from dyn_svo.control.platform import AVPlatform
from dyn_svo.experiments import configs as cf
from dyn_svo.experiments import runner, summary

P = {"v0": 15.0, "T": 1.2, "s0": 2.0, "a_max": 1.0, "b": 1.5}
POL = {"s0": 3.51, "h": 1.35, "sigma_gap": 2.76, "sigma_dv": 0.55, "sigma_a": 0.29, "sigma_u": 0.73}
THR = {"I": {"min_gap": 1.0, "speed_ratio": 0.9, "mean_gap": 30.0}}


def test_configurations():
    c = cf.av_configs()
    names = [x["name"] for x in c]
    assert len(c) == 20 and len(set(names)) == 20
    assert sum(x["family"] == "mpc_fixed" for x in c) == 9 and sum(x["family"] == "mpc_dynamic" for x in c) == 9
    assert c[0]["phi"] == 0.0 and c[8]["phi"] == pytest.approx(math.pi / 2)
    assert cf.all_configs()[0]["name"] == "hdv"


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


def _ctx():
    table = pd.DataFrame([dict(fold="loeo_A", regime="I", variant="nodelay", veh=v, **P) for v in ("1", "2", "pooled")])
    noise = pd.DataFrame([{"fold": "loeo_A", "regime": "I", "noise_sd": 0.3}])
    return {"data": {"A": _ring([1, 2, 3, 4])}, "table": table, "noise": noise, "thresholds": THR,
            "policy": {"I": POL}, "t_start": 30.0, "folds": {"A": "loeo_A"}}


def test_make_driver_uses_regime_settings():
    ctx = dict(_ctx(), pooled={"I": P}, phi_thr_e={"I": {"10": {"a_lo": 0.3, "a_hi": 1.0}}})   # v5
    assert cf.make_driver(cf.HDV, "I", ctx) is None
    d = cf.make_driver(cf.av_configs()[9], "I", ctx)
    assert isinstance(d, AVPlatform) and d.s_min == 1.0


def test_run_task_rows_and_reproducibility():
    ctx = _ctx()
    ecfg = {"seeds": 2, "n_rear": 8}
    hdv = runner.run_task({"config": cf.HDV, "exp": "A"}, ctx, ecfg)
    fix = runner.run_task({"config": cf.av_configs()[4], "exp": "A"}, ctx, ecfg)
    assert len(hdv["rows"]) == 4 * 2 and len(fix["rows"][0]["follower_sigma"]) == 2
    assert fix["rows"][0]["phi_mean"] == pytest.approx(cf.av_configs()[4]["phi"])
    again = runner.run_task({"config": cf.HDV, "exp": "A"}, ctx, ecfg)
    assert hdv["rows"] == again["rows"]
    assert hdv["n_pooled_slots"] == 6      # vehicles 3, 4 have no own parameters: 1 + 2 + 2 + 1 slots


def _rows():
    """Synthetic summary rows: 2 experiments x 2 AV positions x 2 seeds."""
    out = []
    spec = {"hdv": ("hdv", (0,), 1.0), "fixed_0": ("mpc_fixed", (0,), 0.9), "fixed_1": ("mpc_fixed", (1,), 0.9),
            "fixed_2": ("mpc_fixed", (2,), 0.5), "dyn_a": ("mpc_dynamic", (2, 2.0), 0.8)}
    for c, (fam, rank, scale) in spec.items():
        for e in ("A", "C"):
            for av in (1, 2):
                for seed in (0, 1):
                    out.append({"config": c, "family": fam, "rank": rank, "exp": e, "regime": "I", "av": av,
                                "seed": seed, "follower_sigma": [scale, scale], "collision": False,
                                "min_gap": 5.0, "speed_ratio": 1.0,
                                "mean_gap": 40.0 if c == "fixed_2" else 10.0,
                                "safety_share": 0.0, "clip_share": 0.0, "commit": "c", "dirty": False})
    return pd.DataFrame(out)


def test_for_selection_tie_break_and_eligibility():
    df = summary.add_log_for(_rows())
    v = summary.experiment_values(df)
    assert v[("fixed_0", "A")] == pytest.approx(np.log(0.9))
    assert v[("hdv", "C")] == pytest.approx(0.0)
    # fixed_2 is best on FOR but fails G3 (mean gap 40 > 30) -> excluded; fixed_0/fixed_1 tie -> lower rank
    assert summary.select(df, v, "mpc_fixed", ["A", "C"], THR, 1e-3) == "fixed_0"
    assert summary.select(df, v, "mpc_dynamic", ["A"], THR, 1e-3) == "dyn_a"
    cv = summary.cross_validate(df, THR, ["A", "C"], 1e-3)
    row = cv[(cv["family"] == "mpc_fixed") & (cv["held_out"] == "C")].iloc[0]
    assert row["chosen"] == "fixed_0" and row["log_for"] == pytest.approx(np.log(0.9))
    assert summary.final_selection(df, THR, ["A", "C"], 1e-3)["mpc_dynamic"] == "dyn_a"
