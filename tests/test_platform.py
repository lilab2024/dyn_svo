import numpy as np
import pytest

from dyn_svo.config import load_config
from dyn_svo.control import platform as pf
from dyn_svo.sim import chain, idm

DT = idm.DT
P = {"v0": 15.0, "T": 1.2, "s0": 2.0, "a_max": 1.0, "b": 1.5}


class Fixed:
    """Test controller returning a fixed command."""

    def __init__(self, **out):
        self.out = out

    def reset(self, info):
        pass

    def act(self, obs):
        return dict(self.out)


def _obs(v=5.0, gap=30.0, v_leader=5.0):
    return {"t": 0.0, "v": v, "gap": gap, "v_leader": v_leader, "rear_v": np.array([]), "rear_gap": np.array([])}


def _run(platform, obs, steps):
    return [platform.act(obs) for _ in range(steps)]


def test_av_config_matches_d040():
    c = load_config("av")
    assert (c["accel_min"], c["accel_max"], c["jerk_max"]) == (-3.5, 2.0, 2.5)
    assert c["actuation_lag_s"] == 0.3 and c["speed_tracking_tau_s"] == 1.0
    assert c["safety"]["emergency_decel"] == 6.0 and c["safety"]["reaction_s"] == 0.5


def test_velocity_command_and_lag():
    p = pf.AVPlatform(Fixed(v=6.0), s_min=1.0)
    a = _run(p, _obs(v=5.0), 60)                 # target accel (6-5)/1.0 = 1.0, jerk-limited ramp
    assert a[-1] == pytest.approx(1.0, abs=1e-3)
    p2 = pf.AVPlatform(Fixed(a=1.0), s_min=1.0)
    a2 = _run(p2, _obs(), 3)
    assert a2[0] == pytest.approx(0.25 * DT / 0.3)    # first jerk-limited command 0.25, lag dt/0.3


def test_accel_and_jerk_limits():
    p = pf.AVPlatform(Fixed(a=5.0), s_min=1.0)
    a = _run(p, _obs(), 200)
    assert max(a) <= 2.0 + 1e-9 and a[-1] == pytest.approx(2.0, abs=1e-6)
    assert p.a_cmd_prev == 2.0 and p.stats()["clip_share"] > 0
    q = pf.AVPlatform(Fixed(a=-10.0), s_min=1.0)
    assert min(_run(q, _obs(), 200)) >= -3.5 - 1e-9          # far gap: safety inactive


def test_gipps_safe_speed_properties():
    assert pf.gipps_safe_speed(1.0, 0.0, 1.0, 3.5, 3.5, 0.5) == 0.0
    v1 = pf.gipps_safe_speed(10.0, 5.0, 1.0, 3.5, 3.5, 0.5)
    v2 = pf.gipps_safe_speed(20.0, 5.0, 1.0, 3.5, 3.5, 0.5)
    v3 = pf.gipps_safe_speed(10.0, 8.0, 1.0, 3.5, 3.5, 0.5)
    assert 0 < v1 < v2 and v1 < v3


def test_filter_only_more_conservative_and_inactive_when_far():
    rng = np.random.default_rng(0)
    for _ in range(200):
        obs = _obs(v=rng.uniform(0, 8), gap=rng.uniform(0.5, 40), v_leader=rng.uniform(0, 8))
        a_req = rng.uniform(-3.5, 2.0)
        p = pf.AVPlatform(Fixed(a=a_req), s_min=1.0)
        p.act(obs)
        jerk_limited = min(max(a_req, -2.5 * DT), 2.5 * DT)      # first step from a_cmd_prev = 0
        assert p.a_cmd_prev <= jerk_limited + 1e-12               # the filter never raises the command
    far = pf.AVPlatform(Fixed(a=0.0), s_min=1.0)
    _run(far, _obs(v=5.0, gap=40.0, v_leader=5.0), 50)
    assert far.stats()["safety_share"] == 0.0


def test_no_collision_when_leader_brakes_hard():
    steps = 400
    t = np.arange(steps) * DT
    v_l = np.maximum(0.0, 8.0 - 3.5 * np.maximum(t - 5.0, 0.0))     # brakes at 3.5 m/s^2 from 5 s
    x_l = 100.0 + np.concatenate([[0.0], np.cumsum(0.5 * (v_l[1:] + v_l[:-1]) * DT)])
    ep = {"exp": "A", "regime": "I", "av_id": 1, "leader_id": 0, "order": [1, 2], "lengths": np.full(2, 4.5),
          "len_l": 4.5, "t": t, "x_l": x_l, "v_l": v_l, "x0": np.array([100 - 4.5 - 12.0, 100 - 9.0 - 30.0]),
          "v0": np.array([8.0, 8.0])}
    plat = pf.AVPlatform(Fixed(a=0.5), s_min=1.5)                    # controller keeps pushing
    out = chain.simulate(ep, [P, P], av_driver=plat)
    assert out["gap"][:, 0].min() > 0.0 and not out["collided"][0]
    assert plat.stats()["safety_share"] > 0


def test_platform_gets_no_human_noise_and_keeps_crn():
    steps = 300
    t = np.arange(steps) * DT
    v_l = 5.0 + np.sin(t / 3.0)
    x_l = 200.0 + np.concatenate([[0.0], np.cumsum(0.5 * (v_l[1:] + v_l[:-1]) * DT)])
    ep = {"exp": "A", "regime": "I", "av_id": 1, "leader_id": 0, "order": [1, 2, 3], "lengths": np.full(3, 4.5),
          "len_l": 4.5, "t": t, "x_l": x_l, "v_l": v_l,
          "x0": 200.0 - np.arange(1, 4) * 20.0, "v0": np.full(3, 5.0)}
    a = chain.simulate(ep, [P] * 3, av_driver=pf.AVPlatform(Fixed(a=0.0), s_min=1.5), noise_sd=0.5,
                       rng=chain.episode_rng(0, "A", 1))
    b = chain.simulate(ep, [P] * 3, noise_sd=0.5, rng=chain.episode_rng(0, "A", 1))
    assert np.array_equal(a["noise"], b["noise"])
    assert np.allclose(a["a"][:, 0], 0.0, atol=1e-12)


def test_s_min_for_regime():
    thr = {"I": {"min_gap": 1.49}, "II": {"min_gap": 0.80}}
    assert pf.s_min_for_regime(thr, "II") == 0.80
