import json
import math

import numpy as np
import pandas as pd
import pytest

from dyn_svo.sim import idm
from dyn_svo.theory import checks as K
from dyn_svo.theory import gains as G
from dyn_svo.theory import linear as L

POL = {"I": {"s0": 3.51, "h": 1.35, "sigma_gap": 2.76, "sigma_dv": 0.55, "sigma_a": 0.29, "sigma_u": 0.73, "n_rows": 10}}
P = {"v0": 15.0, "T": 1.2, "s0": 2.0, "a_max": 1.0, "b": 1.5}
U = math.pi / 16
SMALL = dict(K.SETTINGS, n_grid=8, table_phis=[0.0, 4 * U, 7 * U], intervals=[[4 * U, 5 * U]],
             fixed=[0.0, 4 * U], rate_grid=4, n_reference=3)


def test_speed_stats_and_ring_sizes():
    rows = pd.DataFrame({"regime": ["I"] * 5 + ["II"] * 3, "v": [1, 2, 3, 4, 5, 10, 20, 30]})
    s = K.speed_stats(rows)
    assert s["I"]["median"] == 3.0 and s["II"]["median"] == 20.0 and s["I"]["n_rows"] == 5
    assert s["I"]["p10"] < s["I"]["median"] < s["I"]["p90"]
    assert K.ring_sizes([("A", "I", 20), ("C", "I", 21), ("F", "II", 22)]) == {"I": {"A": 20, "C": 21}, "II": {"F": 22}}


def test_pooled_idm_selects_exactly_one_row():
    base = {k: 1.0 for k in idm.PARAM_NAMES}
    rows = [dict(base, fold="all_dev", regime="I", variant="nodelay", veh="pooled", v0=15.0),
            dict(base, fold="all_dev", regime="I", variant="delay", veh="pooled", v0=99.0),
            dict(base, fold="cv1", regime="I", variant="nodelay", veh="pooled", v0=98.0),
            dict(base, fold="all_dev", regime="I", variant="nodelay", veh="7", v0=97.0)]
    assert K.pooled_idm(pd.DataFrame(rows), "I")["v0"] == 15.0
    with pytest.raises(KeyError):
        K.pooled_idm(pd.DataFrame(rows), "II")


def test_gain_table_rows_are_consistent():
    m = G.mpc_from_policy(POL["I"])
    lam_h = 0.7
    rows = K.gain_table(m, POL["I"]["h"], lam_h, [20, 22], SMALL)
    assert len(rows) == 3
    for r in rows:
        k = G.feedback_gains(m, r["phi"])
        assert r["k"] == list(k) and r["routh"]
        assert r["lambda_av"] == pytest.approx(L.lambda_av(k, POL["I"]["h"]))
        assert r["head_to_tail"]["22"] == pytest.approx(r["lambda_av"] + 20 * lam_h)
        assert r["gain_at"]["30.0"] == pytest.approx(abs(L.gain_av(k, POL["I"]["h"], 2 * math.pi / 30)))
        assert 20.0 <= r["band_peak"][1] <= 64.0


def test_run_all_small_is_complete_and_serialisable():
    out = K.run_all(POL, {"I": P}, {"I": {"median": 8.0, "p10": 5.0, "p90": 11.0, "n_rows": 10}},
                    {"I": {"A": 20}}, SMALL, log=lambda s: None)
    r = out["I"]
    assert r["signs"]["positive"] and r["signs"]["q_lower"] > 0
    iv = r["certificates"]["intervals"][0]
    for key in ("continuous_dv", "continuous_e_s", "discrete_dv", "discrete_e_s"):
        assert iv[key]["verified"] and iv[key]["gamma"] >= iv[key]["frozen_reference"] * (1 - 1e-6)
    assert iv["rate_gridded_dv"]["label"] == "gridded, not rigorous"
    assert len(r["certificates"]["fixed"]) == 2
    assert all(f["continuous_dv"]["verified"] for f in r["certificates"]["fixed"])
    assert r["human"]["median"]["routh"]
    json.dumps(out)


def test_sign_check_q_is_the_slope_of_k_u():
    m = G.mpc_from_policy(POL["I"])
    res = K.sign_check(m, SMALL)
    assert res["positive"]
    rho = 0.37
    from dyn_svo.theory import certificate as C
    pf = C.partial_fractions(m)
    q = float(np.sum(pf["b"] / (pf["lam"] + rho)))
    assert G.feedback_gains_rho(m, rho)[2] == pytest.approx(rho * q, rel=1e-10)
