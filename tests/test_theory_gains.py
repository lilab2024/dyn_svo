import math

import numpy as np
import pytest

from dyn_svo.control import mpc as M
from dyn_svo.theory import gains as G

POL = {"s0": 3.51, "h": 1.35, "sigma_gap": 2.76, "sigma_dv": 0.55, "sigma_a": 0.29, "sigma_u": 0.73}
POL2 = {"s0": 2.75, "h": 0.82, "sigma_gap": 2.83, "sigma_dv": 0.98, "sigma_a": 0.84, "sigma_u": 2.21}


@pytest.mark.parametrize("pol", [POL, POL2])
def test_gains_reproduce_first_move_of_plan(pol):
    m = G.mpc_from_policy(pol)
    rng = np.random.default_rng(0)
    phis = [0.0, math.pi / 4, math.pi / 2] + list(rng.uniform(0.0, math.pi / 2, 20))
    for phi in phis:
        ks, kv, ku = G.feedback_gains(m, phi)
        for _ in range(10):
            gap, v, vl, u = rng.uniform(1, 40), rng.uniform(0, 30), rng.uniform(0, 30), rng.uniform(0, 30)
            e_s = gap - pol["s0"] - pol["h"] * v
            a = ks * e_s + kv * (vl - v) - ku * (v - u)
            assert a == pytest.approx(m.plan(gap, v, vl, u, phi)[0], rel=1e-9, abs=1e-9)


def test_endpoints():
    m = G.mpc_from_policy(POL)
    ks, kv, ku = G.feedback_gains(m, 0.0)
    assert ku == 0.0 and ks > 0 and kv > 0
    ks, kv, ku = G.feedback_gains(m, math.pi / 2)
    assert abs(ks) < 1e-9 and abs(kv) < 1e-9 and ku > 0


def test_gains_depend_on_tan_phi_only():
    m = G.mpc_from_policy(POL)
    for phi in np.linspace(0.05, 1.45, 15):
        a = np.array(G.feedback_gains(m, phi))
        b = np.array(G.feedback_gains_rho(m, math.tan(phi)))
        assert np.allclose(a, b, rtol=1e-6, atol=1e-12)


def test_gain_curve_matches_pointwise():
    m = G.mpc_from_policy(POL2)
    phis = M.fixed_phi_grid()
    k = G.gain_curve(m, phis)
    assert k.shape == (len(phis), 3)
    for row, phi in zip(k, phis):
        assert tuple(row) == G.feedback_gains(m, phi)


@pytest.mark.parametrize("pol", [POL, POL2])
def test_gain_signs_on_grid_for_test_policies(pol):
    # regression for these test policies only; the calibrated policy is verified by the theory checks
    m = G.mpc_from_policy(pol)
    k = G.gain_curve(m, np.linspace(0.0, 7.9 * math.pi / 16, 80))
    assert (k[:, 0] > 0).all() and (k[:, 1] > 0).all() and (k[:, 2] >= 0).all()
