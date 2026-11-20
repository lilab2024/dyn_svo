import numpy as np
import pytest

from dyn_svo.config import config_md5, load_config
from dyn_svo.sim import idm

P = {"v0": 15.0, "T": 1.2, "s0": 2.0, "a_max": 1.0, "b": 1.5}


def _const_leader(v_lead, gap0, n_seg=2, n_step=300, len_l=5.0):
    t = np.arange(n_step) * idm.DT
    x_l = np.tile(gap0 + len_l + v_lead * t, (n_seg, 1))
    v_l = np.full((n_seg, n_step), v_lead)
    return x_l, v_l, len_l


def test_sim_config_matches_d029():
    c = load_config("sim")
    assert c["dt"] == 0.1 and c["data_hz"] == 30 and c["idm_delta"] == 4.0
    assert c["accel_floor"] == -9.0
    assert round(c["dt"] * c["data_hz"]) == 3
    assert len(config_md5("sim")) == 32


def test_accel_zero_at_equilibrium():
    v = 5.0
    s_e = idm.equilibrium_gap(v, **P)
    assert idm.idm_accel(v, s_e, 0.0, **P) == pytest.approx(0.0, abs=1e-12)


def test_free_road_accel():
    assert idm.idm_accel(5.0, 1e9, 0.0, **P) == pytest.approx(P["a_max"] * (1 - (5 / 15) ** 4))


def test_accel_floor():
    assert idm.idm_accel(10.0, 0.01, 5.0, **P) == idm.ACCEL_FLOOR


def test_ballistic_constant_accel():
    x, v = idm.ballistic_step(0.0, 2.0, 0.5, dt=0.1)
    assert x == pytest.approx(2.0 * 0.1 + 0.5 * 0.5 * 0.01)
    assert v == pytest.approx(2.05)


def test_ballistic_stop_within_step():
    x, v = idm.ballistic_step(0.0, 0.3, -9.0, dt=0.1)
    assert v == 0.0
    assert x == pytest.approx(0.3 ** 2 / (2 * 9.0))


def test_equilibrium_is_stationary():
    v_lead = 5.0
    s_e = idm.equilibrium_gap(v_lead, **P)
    x_l, v_l, len_l = _const_leader(v_lead, s_e)
    x, v, a, gap, col = idm.simulate_follower(x_l, v_l, len_l, np.zeros(2), np.full(2, v_lead), P)
    assert np.abs(gap - s_e).max() < 1e-9
    assert np.abs(v - v_lead).max() < 1e-9
    assert not col.any()


def test_matches_hand_computation_two_steps():
    x_l = np.array([[20.0, 20.5, 21.0]])
    v_l = np.array([[5.0, 5.0, 5.0]])
    x, v, a, gap, _ = idm.simulate_follower(x_l, v_l, 5.0, np.array([0.0]), np.array([4.0]), P)
    s, dv = 20.0 - 5.0 - 0.0, 4.0 - 5.0
    s_star = P["s0"] + max(0.0, 4.0 * P["T"] + 4.0 * dv / (2 * np.sqrt(P["a_max"] * P["b"])))
    a0 = P["a_max"] * (1 - (4.0 / P["v0"]) ** 4 - (s_star / s) ** 2)
    assert a[0, 0] == pytest.approx(a0)
    assert x[0, 1] == pytest.approx(4.0 * 0.1 + 0.5 * a0 * 0.01)
    assert v[0, 1] == pytest.approx(4.0 + a0 * 0.1)


def test_delay_postpones_response():
    v_lead, d = 5.0, 5
    s_e = idm.equilibrium_gap(v_lead, **P)
    x_l, v_l, len_l = _const_leader(v_lead, s_e, n_seg=1, n_step=50)
    v_l = v_l.copy()
    v_l[:, 1:] = 4.0
    x_l = x_l.copy()
    x_l[:, 1:] = x_l[:, :1] + 4.0 * np.arange(1, 50) * idm.DT
    _, v0d, _, _, _ = idm.simulate_follower(x_l, v_l, len_l, np.zeros(1), np.full(1, v_lead), P)
    _, vd, _, _, _ = idm.simulate_follower(x_l, v_l, len_l, np.zeros(1), np.full(1, v_lead), P,
                                           delay_steps=d)
    assert v0d[0, 2] < v_lead - 1e-6
    assert np.allclose(vd[0, :d + 1], v_lead)
    assert vd[0, d + 2] < v_lead - 1e-6


def test_no_negative_speed_and_collision_flag():
    x_l, v_l, len_l = _const_leader(0.0, 3.0, n_seg=1, n_step=100)
    x, v, a, gap, col = idm.simulate_follower(x_l, v_l, len_l, np.zeros(1), np.full(1, 10.0), P)
    assert v.min() >= 0.0
    assert col[0]


def test_segments_are_independent():
    v_lead = 5.0
    s_e = idm.equilibrium_gap(v_lead, **P)
    x_l, v_l, len_l = _const_leader(v_lead, s_e, n_seg=3)
    x0 = np.array([0.0, -1.0, 0.0])
    _, v_all, _, _, _ = idm.simulate_follower(x_l, v_l, len_l, x0, np.full(3, v_lead), P)
    _, v_one, _, _, _ = idm.simulate_follower(x_l[1:2], v_l[1:2], len_l, x0[1:2], np.full(1, v_lead), P)
    assert np.allclose(v_all[1], v_one[0])
