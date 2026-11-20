import numpy as np
import pandas as pd
import pytest

from dyn_svo.analysis import human_policy as hp
from dyn_svo.config import load_config


def test_mpc_config_matches_d043():
    c = load_config("mpc")
    assert c["horizon_s"] == 10.0 and c["block_s"] == 0.5 and c["fixed_phi_grid_points"] == 9
    assert c["leader_prediction"] == "constant_speed" and c["human_policy_run"] == "human_policy_v1"


def test_policy_fit_recovers_line_and_scales():
    rng = np.random.default_rng(0)
    n = 20000
    v = rng.uniform(0, 8, n)
    gap = 2.0 + 1.5 * v + 0.5 * rng.standard_normal(n)
    v_l = v + 0.3 * rng.standard_normal(n)
    rows = pd.DataFrame({"regime": "I", "v": v, "gap": gap, "v_l": v_l, "a": 0.2 * rng.standard_normal(n),
                         "U": v + 0.4 * rng.standard_normal(n)})
    p = hp.policy_from_rows(rows)["I"]
    assert p["s0"] == pytest.approx(2.0, abs=0.05) and p["h"] == pytest.approx(1.5, abs=0.01)
    assert p["sigma_gap"] == pytest.approx(0.5, rel=0.03) and p["sigma_dv"] == pytest.approx(0.3, rel=0.03)
    assert p["sigma_a"] == pytest.approx(0.2, rel=0.03) and p["sigma_u"] == pytest.approx(0.4, rel=0.03)


def test_follower_rows_causal_mean_and_filter():
    k = np.arange(0, 3000)
    t = k / 30.0
    rows = []
    for veh, lead, vel in ((1, 2, 5.0 + np.sin(t)), (2, 1, np.full(len(k), 4.0))):
        rows.append(pd.DataFrame({"exp": "A", "regime": "I", "veh_id": veh, "leader_id": lead, "k": k, "t": t,
                                  "x": 0.0, "v": vel, "a": 0.0, "length": 4.5, "gap": 10.0,
                                  "follower_ok": True}))
    df = pd.concat(rows, ignore_index=True)
    out = hp.follower_rows(df, t_from_s=60.0, window_s=1.0)
    assert out["t"].min() >= 60.0 and (out["k"] % 3 == 0).all()
    r = out[out["veh_id"] == 1].iloc[0]
    assert r["U"] == pytest.approx(4.0)                           # leader (veh 2) constant 4 m/s
    r2 = out[out["veh_id"] == 2].iloc[0]
    kk = int(r2["k"])
    past = [5.0 + np.sin(j / 30.0) for j in range(kk - 27, kk + 1, 3)]   # last 10 samples at 10 Hz
    assert r2["U"] == pytest.approx(np.mean(past))
