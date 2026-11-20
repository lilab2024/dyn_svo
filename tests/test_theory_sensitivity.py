"""theory_v2 functions: LOEO fits, bound flags, head-to-tail crossings, critical b, gain curves."""
import json
import math

import pandas as pd
import pytest

from dyn_svo.sim import idm
from dyn_svo.theory import checks as K
from dyn_svo.theory import gains as G
from dyn_svo.theory import linear as L

POL = {"I": {"s0": 3.51, "h": 1.35, "sigma_gap": 2.76, "sigma_dv": 0.55, "sigma_a": 0.29, "sigma_u": 0.73, "n_rows": 10}}
P = {"v0": 15.0, "T": 1.2, "s0": 2.0, "a_max": 1.0, "b": 1.5}
BOUNDS = {"v0": [5.0, 30.0], "T": [0.1, 3.0], "s0": [0.1, 5.0], "a_max": [0.1, 4.0], "b": [0.1, 5.0]}
U = math.pi / 16
SMALL = dict(K.SENSITIVITY, b_grid=[0.1, 1.0], curve_periods=[10.0, 600.0, 7], curve_phis=[0.0, 4 * U])


def _table():
    rows = []
    for fold, held, v0 in (("all_dev", None, 15.0), ("loeo_A", "A", 16.0), ("loeo_C", "C", 17.0), ("loeo_F", None, 15.0)):
        rows.append(dict(P, fold=fold, regime="I", variant="nodelay", veh="pooled", held_out=held, v0=v0))
        rows.append(dict(P, fold=fold, regime="I", variant="delay", veh="pooled", held_out=held, v0=99.0))
    rows.append(dict(P, fold="loeo_A", regime="I", variant="nodelay", veh="7", held_out="A", v0=98.0))
    return pd.DataFrame(rows)


def test_loeo_fits_take_the_folds_held_out_in_the_regime():
    fits = K.loeo_fits(_table(), "I")
    assert sorted(fits) == ["loeo_A", "loeo_C"] and fits["loeo_A"]["v0"] == 16.0 and fits["loeo_C"]["v0"] == 17.0
    dup = pd.concat([_table(), _table().iloc[[2]]], ignore_index=True)
    with pytest.raises(KeyError):
        K.loeo_fits(dup, "I")


def test_at_bounds_flags():
    p = dict(P, b=0.1, v0=29.99, T=1.0)
    f = K.at_bounds(p, BOUNDS)
    assert f["b"] == "lower" and f["v0"] == "upper" and f["T"] is None and f["s0"] is None
    assert K.at_bounds(dict(P, v0=29.9), BOUNDS)["v0"] is None


def test_head_to_tail_crossings_are_sign_changes():
    m = G.mpc_from_policy(POL["I"])
    h = POL["I"]["h"]
    phis = [i * math.pi / 160 for i in range(80)]
    lam = [L.lambda_av(G.feedback_gains(m, p), h) for p in phis]
    lam_h = -lam[40] / 18.0            # puts a sign change near 4 pi/16 for n = 20
    r = K.head_to_tail_crossings(m, h, lam_h, 20, phis)
    assert len(r["crossings"]) >= 1
    for x in r["crossings"]:
        assert L.lambda_av(G.feedback_gains(m, x), h) + 18 * lam_h == pytest.approx(0.0, abs=1e-6)
    none = K.head_to_tail_crossings(m, h, 1e6, 20, phis)
    assert none["crossings"] == [] and none["value_first"] > 0 and none["value_last"] > 0


def test_critical_b_zeroes_lambda_h():
    v = 6.0
    c = K.critical_b(P, v, 0.1, 5.0)
    assert c["b"] is not None
    if c["b"] is not None:
        assert L.lambda_human(L.human_partials(dict(P, b=c["b"]), v)) == pytest.approx(0.0, abs=1e-8)
        assert c["lambda_h_lo"] * c["lambda_h_hi"] <= 0
    narrow = K.critical_b(P, v, 4.9, 5.0)
    assert (narrow["b"] is None) == (narrow["lambda_h_lo"] * narrow["lambda_h_hi"] > 0)


def test_gain_curves_equal_the_transfer_function():
    m = G.mpc_from_policy(POL["I"])
    gc = K.gain_curves(m, POL["I"]["h"], SMALL)
    assert len(gc["periods_s"]) == 7 and gc["periods_s"][0] == pytest.approx(10.0) and gc["periods_s"][-1] == pytest.approx(600.0)
    c = gc["curves"][1]
    assert c["phi_over_pi16"] == pytest.approx(4.0)
    k = G.feedback_gains(m, 4 * U)
    assert c["gain"][3] == pytest.approx(abs(L.gain_av(k, POL["I"]["h"], 2 * math.pi / gc["periods_s"][3])))


def test_run_sensitivity_small_is_complete_and_serialisable():
    speeds = {"I": {"median": 8.0, "p10": 5.0, "p90": 11.0, "n_rows": 10}}
    out = K.run_sensitivity(POL, _table(), speeds, {"I": {"A": 20, "C": 22}}, BOUNDS, SMALL, log=lambda s: None)
    r = out["I"]
    assert r["all_dev"]["params"] == K.pooled_idm(_table(), "I")
    assert sorted(r["loeo"]) == ["loeo_A", "loeo_C"] and sorted(r["all_dev"]["head_to_tail"]) == ["20", "22"]
    assert [c["b"] for c in r["b_grid"]] == [0.1, 1.0, 1.5] and [c["fitted"] for c in r["b_grid"]] == [False, False, True]
    assert r["b_grid"][0]["params"]["b"] == 0.1 and r["b_grid"][0]["at_bound"]["b"] == "lower"
    assert len(r["gain_curves"]["curves"]) == 2
    json.dumps(out)
