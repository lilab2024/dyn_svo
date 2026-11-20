import math

import numpy as np
import pytest

from dyn_svo.theory import certificate as C
from dyn_svo.theory import gains as G
from dyn_svo.theory import linear as L

POL = {"s0": 3.51, "h": 1.35, "sigma_gap": 2.76, "sigma_dv": 0.55, "sigma_a": 0.29, "sigma_u": 0.73}
U = math.pi / 16


@pytest.fixture(scope="module")
def m():
    return G.mpc_from_policy(POL)


def test_partial_fractions_equal_direct_solve(m):
    pf = C.partial_fractions(m)
    assert (pf["lam"] > 0).all()
    for rho in [0.0, 0.3, 1.0, 4.0, 20.0]:
        assert np.allclose(C.gains_pf(pf, rho), G.feedback_gains_rho(m, rho), rtol=1e-10, atol=1e-13)


def test_second_derivative_bound_holds(m):
    pf = C.partial_fractions(m)
    h = 1e-3
    for rho0 in [0.0, 1.0, 5.0]:
        bound = C.second_derivative_bound(pf, rho0)
        for rho in [rho0 + 0.01, rho0 + 0.5, rho0 + 3.0]:
            d2 = (C.gains_pf(pf, rho + h) - 2 * C.gains_pf(pf, rho) + C.gains_pf(pf, rho - h)) / h ** 2
            assert (np.abs(d2) <= bound * (1 + 1e-6) + 1e-9).all()


def test_cover_contains_the_curve_of_the_implementation(m):
    lo, hi = 3 * U, 7 * U
    cov = C.curve_cover(m, lo, hi, n_grid=20)
    rho = cov["rho"]
    for phi in np.linspace(lo, hi, 2001):
        r = math.tan(phi)
        j = min(int(np.searchsorted(rho, r, side="right")) - 1, len(rho) - 2)
        t = (r - rho[j]) / (rho[j + 1] - rho[j])
        chord = (1 - t) * cov["k"][j] + t * cov["k"][j + 1]
        half = np.maximum(cov["half"][j], cov["half"][j + 1])
        assert (np.abs(np.array(G.feedback_gains(m, phi)) - chord) <= half).all()
    assert (cov["k_lower"] > 0).all()
    assert len(cov["corners"]) == 8 * len(rho)


def test_cover_rejects_pi_half(m):
    with pytest.raises(ValueError):
        C.curve_cover(m, 5 * U, 8 * U)


def test_shifted_discrete_loop_equals_discrete_gain(m):
    k = G.feedback_gains(m, 5 * U)
    A, B, Cm, D = C.discrete_shifted(k, POL["h"])
    for w in [0.02, 0.2, 1.0]:
        z = np.exp(1j * w * L.DT)
        ma = np.mean(z ** (-np.arange(int(round(L.T_MA / L.DT)))))
        win = np.array([1.0, ma])
        g = Cm[0] @ np.linalg.solve(z * np.eye(3) - A, B @ win) + D[0] @ win
        assert abs(g - L.gain_av_discrete(k, POL["h"], w)) < 1e-12


@pytest.mark.parametrize("discrete", [False, True])
@pytest.mark.parametrize("output", [0, 1])
def test_single_point_certificate_equals_hinf_norm(m, discrete, output):
    cov = C.curve_cover(m, 4 * U, 4 * U)
    res = C.certify(C.corner_systems(cov["corners"], POL["h"], discrete), output, discrete)
    assert res["verified"]
    ref = C.hinf_norm_frozen(G.feedback_gains(m, 4 * U), POL["h"], output, discrete)
    assert res["gamma"] >= ref * (1 - 1e-6)
    assert res["gamma"] <= ref * (1 + 2e-3)


def test_interval_certificate_bounds_every_frozen_point(m):
    lo, hi = 4 * U, 7 * U
    cov = C.curve_cover(m, lo, hi, n_grid=20)
    res = C.certify(C.corner_systems(cov["corners"], POL["h"]), 0)
    assert res["verified"]
    for phi in np.linspace(lo, hi, 7):
        assert res["gamma"] >= C.hinf_norm_frozen(G.feedback_gains(m, phi), POL["h"], 0) * (1 - 1e-6)
    gridded = C.certify_rate_gridded(m, POL["h"], lo, hi, 0.1, 0, n_grid=20)
    assert gridded["label"] == "gridded, not rigorous"
    assert gridded["gamma"] <= res["gamma"] * (1 + 1e-3)


def test_verify_rejects_wrong_certificates(m):
    cov = C.curve_cover(m, 4 * U, 4 * U)
    systems = C.corner_systems(cov["corners"], POL["h"])
    res = C.certify(systems, 0)
    assert res["verified"]
    assert not C.verify(res["P"], 0.5 * res["gamma"], systems, 0)[0]
    assert not C.verify(-res["P"], res["gamma"], systems, 0)[0]
