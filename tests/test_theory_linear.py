import math

import numpy as np
import pytest

from dyn_svo.control import cooperation as co
from dyn_svo.control import mpc as M
from dyn_svo.control.estimator import MeanSpeedEstimator
from dyn_svo.control.platform import AVPlatform
from dyn_svo.sim import chain, idm
from dyn_svo.theory import gains as G
from dyn_svo.theory import linear as L

POL = {"s0": 3.51, "h": 1.35, "sigma_gap": 2.76, "sigma_dv": 0.55, "sigma_a": 0.29, "sigma_u": 0.73}
P = {"v0": 15.0, "T": 1.2, "s0": 2.0, "a_max": 1.0, "b": 1.5}
PHIS = [0.0, 3 * math.pi / 16, 5 * math.pi / 16, 7 * math.pi / 16]


def _episode(v_star=10.0, n_const=600, n_wave=900, amp=0.2, period=30.0):
    t = np.arange(n_const + n_wave) * idm.DT
    vl = np.full(len(t), v_star)
    vl[n_const:] += amp * np.sin(2 * math.pi * (t[n_const:] - t[n_const]) / period)
    xl = 300 + np.concatenate([[0.0], np.cumsum(0.5 * (vl[1:] + vl[:-1]) * idm.DT)])
    gap0 = POL["s0"] + POL["h"] * v_star
    x0 = np.array([xl[0] - 4.5 - gap0, xl[0] - 4.5 - gap0 - 4.5 - 20.0])
    return {"exp": "A", "regime": "I", "av_id": 1, "leader_id": 0, "order": [1, 2],
            "lengths": np.full(2, 4.5), "len_l": 4.5, "t": t, "x_l": xl, "v_l": vl, "x0": x0,
            "v0": np.full(2, v_star)}


@pytest.mark.parametrize("phi", PHIS)
def test_discrete_model_equals_simulator(phi):
    v_star = 10.0
    ep = _episode(v_star)
    m = M.MPC(POL, M.FixedPhi(phi))
    plat = AVPlatform(m, s_min=1.5)
    out = chain.simulate(ep, [P, P], av_driver=plat)
    st = plat.stats()
    assert st["safety_share"] == 0.0 and st["clip_share"] == 0.0
    Ad, Bd, Cd = L.discrete_matrices(G.feedback_gains(m, phi), POL["h"])
    est = MeanSpeedEstimator()
    z = np.zeros(3)
    n = len(ep["t"])
    dv_lin = np.empty(n)
    for i in range(n):
        dv_lin[i] = z[1]
        if i + 1 == n:
            break
        u_mean = est.update(ep["v_l"][i])
        w = np.array([ep["v_l"][i] - v_star, u_mean - v_star, (ep["x_l"][i + 1] - ep["x_l"][i]) / idm.DT - v_star])
        z = Ad @ z + Bd @ w
    assert np.max(np.abs(out["v"][:, 0] - v_star - dv_lin)) < 1e-9
    assert np.max(np.abs(out["v"][:, 0] - v_star)) > 0.05     # the wave reached the AV


@pytest.mark.parametrize("phi", PHIS)
def test_discrete_and_continuous_gains_agree(phi):
    m = G.mpc_from_policy(POL)
    k = G.feedback_gains(m, phi)
    w = 2 * math.pi / np.geomspace(20.0, 300.0, 40)     # wave band and longer (Euler lag differs below 20 s)
    gc = L.gain_av(k, POL["h"], w)
    gd = L.gain_av_discrete(k, POL["h"], w)
    assert np.max(np.abs(np.abs(gc) - np.abs(gd))) < 0.01


@pytest.mark.parametrize("phi", PHIS)
def test_lambda_av_is_the_low_frequency_coefficient(phi):
    m = G.mpc_from_policy(POL)
    k = G.feedback_gains(m, phi)
    w = 1e-3
    numeric = (1 - abs(L.gain_av(k, POL["h"], w)) ** 2) / w ** 2
    assert numeric == pytest.approx(L.lambda_av(k, POL["h"]), rel=2e-2, abs=0.05)


def test_routh_hurwitz_matches_eigenvalues():
    m = G.mpc_from_policy(POL)
    for phi in np.linspace(0.0, 7.9 * math.pi / 16, 30):
        k = G.feedback_gains(m, phi)
        A, _, _ = L.continuous_matrices(k, POL["h"])
        assert L.routh_hurwitz_stable(k, POL["h"]) == bool(np.linalg.eigvals(A).real.max() < 0)
        assert L.routh_hurwitz_stable(k, POL["h"])
    assert not L.routh_hurwitz_stable((0.0, 0.0, 0.3), POL["h"])    # no gap feedback (phi = pi/2)


def test_continuous_transfer_matches_state_space():
    m = G.mpc_from_policy(POL)
    k = G.feedback_gains(m, 5 * math.pi / 16)
    A, B, C = L.continuous_matrices(k, POL["h"])
    for w in [0.01, 0.1, 0.5]:
        lam = 1j * w
        x = np.linalg.solve(lam * np.eye(3) - A, B @ np.array([1.0, L.moving_average_continuous(lam)]))
        assert C[0] @ x == pytest.approx(L.gain_av(k, POL["h"], w), rel=1e-10)


def test_human_linearisation():
    v = 10.0
    f = L.human_partials(P, v)
    s = idm.equilibrium_gap(v, **P)
    assert float(idm.idm_accel(v, s, 0.0, **P)) == pytest.approx(0.0, abs=1e-12)
    assert f[0] > 0 and f[1] < 0 and f[2] < 0
    w = np.array([0.05, 0.2, 1.0])
    assert np.allclose(np.abs(L.gain_human(f, w)), co.transfer_gain(*f, w), rtol=1e-12)
    wq = 1e-3
    numeric = (1 - abs(L.gain_human(f, wq)) ** 2) / wq ** 2
    assert numeric == pytest.approx(L.lambda_human(f), rel=1e-3)
