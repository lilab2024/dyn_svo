import math

import numpy as np
import pytest

from dyn_svo.config import load_config
from dyn_svo.control import cooperation as co
from dyn_svo.sim import idm

P = {"v0": 15.0, "T": 1.2, "s0": 2.0, "a_max": 1.0, "b": 1.5}


def test_config_matches_d042():
    c = load_config("cooperation")
    assert c["wave_band_s"] == [20.0, 64.0] and c["phi_rate_max"] == 0.1
    assert c["grid"]["K"] == [2, 4, 8] and c["grid"]["tau_f_s"] == [2.0, 5.0, 10.0]


def test_partials_match_finite_differences():
    s, v, dv, h = 9.0, 5.0, 0.3, 1e-6
    a0 = idm.idm_accel(v, s, dv, **P)
    fd = ((idm.idm_accel(v, s + h, dv, **P) - a0) / h, (idm.idm_accel(v + h, s, dv, **P) - a0) / h,
          (idm.idm_accel(v, s, dv + h, **P) - a0) / h)
    assert np.allclose(co.idm_partials(s, v, dv, P), fd, rtol=1e-4)


def test_gain_matches_simulated_amplitude_ratio():
    v = 5.0
    se = float(idm.equilibrium_gap(v, **P))
    w = 2 * np.pi / 30.0
    t = np.arange(0, 900, idm.DT)
    vl = v + 0.05 * np.sin(w * t)
    xl = 100 + np.concatenate([[0.0], np.cumsum(0.5 * (vl[1:] + vl[:-1]) * idm.DT)])
    _, vv, _, _, _ = idm.simulate_follower(xl[None], vl[None], 4.5, np.array([100 - 4.5 - se]), np.array([v]), P)
    n = len(t) // 2
    ratio = vv[0, n:].std() / vl[n:].std()
    assert ratio == pytest.approx(co.transfer_gain(*co.idm_partials(se, v, 0.0, P), w), rel=0.01)


def test_gamma_formula():
    assert co.gamma_from_gains([1.2, 0.5]) == pytest.approx(1.2 + 0.6)
    assert co.gamma_from_gains([]) == 0.0


def _obs(gaps, speeds):
    return {"rear_gap": np.asarray(gaps, dtype=float), "rear_v": np.asarray(speeds, dtype=float)}


def test_phi_range_uses_only_k_vehicles():
    cw = co.CooperationWeight(P, K=2, tau_f=5.0)
    phi = cw.update(_obs([8, 9, 10, 11], [5, 5, 5, 5]))
    g = co.peak_gains([8, 9], [5, 5], P, co.band_omegas())
    assert cw.last_gamma == pytest.approx(co.gamma_from_gains(g))
    assert phi == pytest.approx(math.atan(cw.last_gamma)) and 0 <= phi < math.pi / 2


def test_filter_and_rate_limit():
    cw = co.CooperationWeight(P, K=1, tau_f=2.0)
    cw.update(_obs([], []))                       # no rear vehicle: Gamma 0, phi 0
    assert cw.phi == 0.0
    phis = [cw.update(_obs([8.0], [5.0])) for _ in range(300)]
    steps = np.diff([0.0] + phis)
    assert steps.max() <= 0.1 * idm.DT + 1e-12    # rate limit 0.1 rad/s
    assert cw.gamma_f == pytest.approx(cw.last_gamma, rel=1e-3)   # filter converged
    assert phis[-1] == pytest.approx(math.atan(cw.last_gamma), rel=1e-3)
