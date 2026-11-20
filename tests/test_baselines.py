import numpy as np
import pytest

from dyn_svo.config import load_config
from dyn_svo.control import baselines as bl
from dyn_svo.control.estimator import MeanSpeedEstimator
from dyn_svo.control.platform import AVPlatform
from dyn_svo.sim import chain, idm

C = load_config("baselines")
FS, PI = C["follower_stopper"], C["pi_saturation"]
P = {"v0": 15.0, "T": 1.2, "s0": 2.0, "a_max": 1.0, "b": 1.5}


def test_config_matches_stern_2018():
    assert FS["dx0"] == [4.5, 5.25, 6.0] and FS["d"] == [1.5, 1.0, 0.5]
    assert (PI["g_l"], PI["g_u"], PI["v_catch"], PI["gamma"]) == (7.0, 30.0, 1.0, 2.0)
    assert (PI["dxs_time_s"], PI["dxs_min"], PI["update_hz"]) == (2.0, 4.0, 30)
    assert C["mean_speed_window_s"] == 38


def test_fs_boundaries_example_from_paper():
    assert bl.fs_boundaries(-3.0, FS["dx0"], FS["d"]) == pytest.approx([7.5, 9.75, 15.0])
    assert bl.fs_boundaries(+2.0, FS["dx0"], FS["d"]) == pytest.approx([4.5, 5.25, 6.0])


def test_fs_command_regions():
    U = 7.0
    assert bl.fs_command(4.0, 5.0, 5.0, U, FS["dx0"], FS["d"]) == 0.0
    assert bl.fs_command((4.5 + 5.25) / 2, 5.0, 5.0, U, FS["dx0"], FS["d"]) == pytest.approx(2.5)
    assert bl.fs_command((5.25 + 6.0) / 2, 5.0, 5.0, U, FS["dx0"], FS["d"]) == pytest.approx(6.0)
    assert bl.fs_command(20.0, 5.0, 5.0, U, FS["dx0"], FS["d"]) == U
    assert bl.fs_command(5.5, 5.0, 9.0, U, FS["dx0"], FS["d"]) <= U      # v = min(v_lead, U)


def test_pi_weights_and_target():
    a, b, dxs = bl.pi_weights(5.0, 5.0, 5.0, PI)
    assert dxs == 4.0 and a == pytest.approx(0.5) and b == pytest.approx(0.75)
    assert bl.pi_weights(10.0, 5.0, 8.0, PI)[2] == pytest.approx(6.0)      # 2 s * 3 m/s
    assert bl.pi_target(7.0, 5.0, PI) == pytest.approx(5.0)
    assert bl.pi_target(18.5, 5.0, PI) == pytest.approx(5.5)
    assert bl.pi_target(40.0, 5.0, PI) == pytest.approx(6.0)


def test_pi_update_three_substeps():
    pi = bl.PISaturation()
    pi.reset({})
    obs = {"v": 5.0, "gap": 5.0, "v_leader": 6.0}
    out = pi.act(obs)["v"]
    a, b, _ = bl.pi_weights(5.0, 5.0, 6.0, PI)
    vt = bl.pi_target(5.0, 5.0, PI)
    v = 5.0
    for _ in range(3):
        v = b * (a * vt + (1 - a) * 6.0) + (1 - b) * v
    assert pi.substeps == 3 and out == pytest.approx(v)


def test_estimator_is_causal_and_windowed():
    e = MeanSpeedEstimator(window_s=1.0, dt=0.1)
    vals = [e.update(x) for x in range(20)]
    assert vals[0] == 0.0 and vals[4] == pytest.approx(2.0)      # mean of 0..4 so far
    assert vals[19] == pytest.approx(np.mean(range(10, 20)))     # last 10 samples only


def _const_leader_ep(v_lead=5.0, gap0=12.0, steps=1500):
    t = np.arange(steps) * idm.DT
    x_l = 300.0 + v_lead * t
    return {"exp": "A", "regime": "I", "av_id": 1, "leader_id": 0, "order": [1, 2, 3],
            "lengths": np.full(3, 4.5), "len_l": 4.5, "t": t, "x_l": x_l, "v_l": np.full(steps, v_lead),
            "x0": 300.0 - 4.5 - gap0 - np.array([0.0, 4.5 + 15.0, 2 * (4.5 + 15.0)]), "v0": np.full(3, v_lead)}


@pytest.mark.parametrize("ctrl", [bl.FollowerStopper, bl.PISaturation])
def test_closed_loop_follows_constant_leader(ctrl):
    ep = _const_leader_ep()
    out = chain.simulate(ep, [P] * 3, av_driver=AVPlatform(ctrl(), s_min=1.5))
    assert not out["collided"].any()
    assert out["v"][-200:, 0].mean() == pytest.approx(5.0, abs=0.2)
