import math

import numpy as np
import pytest
from scipy.optimize import minimize

from dyn_svo.control import mpc as M
from dyn_svo.control.cooperation import CooperationWeight
from dyn_svo.control.platform import AVPlatform
from dyn_svo.sim import chain, idm

POL = {"s0": 3.51, "h": 1.35, "sigma_gap": 2.76, "sigma_dv": 0.55, "sigma_a": 0.29, "sigma_u": 0.73}
P = {"v0": 15.0, "T": 1.2, "s0": 2.0, "a_max": 1.0, "b": 1.5}


def _ep(vl, n_veh=2):
    steps = len(vl)
    t = np.arange(steps) * idm.DT
    xl = 300 + np.concatenate([[0.0], np.cumsum(0.5 * (vl[1:] + vl[:-1]) * idm.DT)])
    x0 = xl[0] - 4.5 - 10.0 - np.arange(n_veh) * (4.5 + 15.0)
    return {"exp": "A", "regime": "I", "av_id": 1, "leader_id": 0, "order": list(range(1, n_veh + 1)),
            "lengths": np.full(n_veh, 4.5), "len_l": 4.5, "t": t, "x_l": xl, "v_l": vl, "x0": x0,
            "v0": np.full(n_veh, vl[0])}


def _run(phi, vl):
    return chain.simulate(_ep(vl), [P, P], av_driver=AVPlatform(M.MPC(POL, M.FixedPhi(phi)), s_min=1.5))


def test_fixed_phi_grid():
    g = M.fixed_phi_grid()
    assert len(g) == 9 and g[0] == 0.0 and g[-1] == pytest.approx(math.pi / 2)
    assert np.allclose(np.diff(g), math.pi / 16)


def test_closed_form_matches_numerical_optimum():
    m = M.MPC(POL, M.FixedPhi(0.7))
    args = (9.0, 5.0, 5.5, 5.2, 0.7)
    u = m.plan(*args)
    r = minimize(lambda x: m.cost(x, *args), np.zeros(20), method="BFGS", options={"gtol": 1e-10})
    assert m.cost(u, *args) <= r.fun + 1e-9 and np.abs(u - r.x).max() < 1e-5


def test_phi0_converges_to_desired_gap():
    out = _run(0.0, np.full(3000, 5.0))
    assert out["gap"][-1, 0] == pytest.approx(POL["s0"] + POL["h"] * 5.0, abs=1e-3)
    assert out["v"][-1, 0] == pytest.approx(5.0, abs=1e-3) and not out["collided"].any()


def test_phi_trades_gap_variation_for_speed_smoothness():
    t = np.arange(3000) * idm.DT
    vl = 5.0 + np.sin(2 * np.pi * t / 30.0)
    n0 = 600
    gap_std, v_std = [], []
    for phi in M.fixed_phi_grid():
        o = _run(phi, vl)
        assert not o["collided"].any()
        gap_std.append(o["gap"][n0:, 0].std())
        v_std.append(o["v"][n0:, 0].std())
    assert np.all(np.diff(gap_std) > 0)                       # more phi -> more gap used as buffer
    assert v_std[7] < v_std[0] < vl[n0:].std()                # smoother than the leader, more so at high phi
    assert v_std[8] < 0.5 * v_std[0]


def test_dynamic_phi_runs_in_chain():
    t = np.arange(1500) * idm.DT
    vl = 5.0 + np.sin(2 * np.pi * t / 30.0)
    ctrl = M.MPC(POL, CooperationWeight(P, K=2, tau_f=5.0))
    out = chain.simulate(_ep(vl, n_veh=3), [P] * 3, av_driver=AVPlatform(ctrl, s_min=1.5), n_rear=2)
    assert not out["collided"].any() and 0.0 <= ctrl.last_phi < math.pi / 2


def test_reset_clears_estimator():
    m = M.MPC(POL, M.FixedPhi(0.3))
    obs = {"v": 5.0, "gap": 10.0, "v_leader": 6.0, "rear_v": np.array([]), "rear_gap": np.array([])}
    m.act(obs)
    m.reset({})
    m.act(dict(obs, v_leader=2.0))
    assert m.last_U == pytest.approx(2.0)
