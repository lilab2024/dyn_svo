import numpy as np
import pandas as pd
import pytest

from dyn_svo.calibration import params as cp
from dyn_svo.sim import chain, idm

P = {"v0": 15.0, "T": 1.2, "s0": 2.0, "a_max": 1.0, "b": 1.5}
L = 260.0


def _ring(ids, v=5.0, seconds=60.0, wobble=0.0):
    """Loader-format ring: ids[0] leads ids[1] ...; ids[0] follows ids[-1]. x = front bumper."""
    n = len(ids)
    lengths = [4.5 + 0.3 * i for i in range(n)]
    k = np.arange(int(seconds * 30))
    t = k / 30.0
    fronts = [L - i * L / n + v * t + wobble * np.sin(0.3 * t + i) for i in range(n)]
    rows = []
    for i, vid in enumerate(ids):
        j = (i - 1) % n
        gap = np.mod(fronts[j] - fronts[i] - lengths[j], L)
        vel = v + wobble * 0.3 * np.cos(0.3 * t + i)
        rows.append(pd.DataFrame({"exp": "A", "veh_id": vid, "regime": "I", "k": k, "t": t,
                                  "x": fronts[i], "v": vel, "a": 0.0, "length": lengths[i],
                                  "leader_id": ids[j], "gap": gap}))
    return pd.concat(rows, ignore_index=True)


def test_chain_order():
    df = _ring([3, 7, 1, 9])
    leader, order = chain.chain_order(df, 1)
    assert leader == 7 and order == [1, 9, 3]


def test_initial_positions_reproduce_data_gaps():
    df = _ring([3, 7, 1, 9], wobble=0.5)
    ep = chain.build_episode(df, 1, t_start_s=30.0)
    lead_x = np.concatenate([[ep["x_l"][0]], ep["x0"][:-1]])
    lead_len = np.concatenate([[ep["len_l"]], ep["lengths"][:-1]])
    assert np.allclose(lead_x - lead_len - ep["x0"], ep["real_gap"][0])
    assert len(ep["t"]) == len(np.arange(900, 1800, 3)) and ep["t"][0] == pytest.approx(30.0)


def test_single_follower_matches_calibration_simulator():
    df = _ring([5, 6], wobble=0.8)
    ep = chain.build_episode(df, 6, t_start_s=30.0)
    out = chain.simulate(ep, [P])
    _, v_ref, _, gap_ref, _ = idm.simulate_follower(ep["x_l"][None, :], ep["v_l"][None, :], ep["len_l"],
                                                     ep["x0"][:1], ep["v0"][:1], P)
    assert np.allclose(out["v"][:, 0], v_ref[0]) and np.allclose(out["gap"][:, 0], gap_ref[0])


def test_equilibrium_chain_is_stationary():
    n, v = 5, 5.0
    s_e = float(idm.equilibrium_gap(v, **P))
    lengths = np.full(n - 1, 4.5)
    t = np.arange(300) * idm.DT
    ep = {"exp": "A", "regime": "I", "av_id": 1, "leader_id": 0, "order": list(range(1, n)),
          "lengths": lengths, "len_l": 4.5, "t": t, "x_l": 100.0 + v * t, "v_l": np.full(300, v),
          "x0": 100.0 - np.arange(1, n) * (4.5 + s_e), "v0": np.full(n - 1, v)}
    out = chain.simulate(ep, [P] * (n - 1))
    assert np.abs(out["gap"] - s_e).max() < 1e-9 and not out["collided"].any()


def test_human_driver_in_av_slot_equals_all_hdv():
    df = _ring([3, 7, 1, 9], wobble=0.8)
    ep = chain.build_episode(df, 1, t_start_s=30.0)
    ps = [P, dict(P, T=1.5), dict(P, b=2.0)]
    base = chain.simulate(ep, ps)
    with_driver = chain.simulate(ep, ps, av_driver=chain.HumanDriver(ps[0]))
    assert np.allclose(base["v"], with_driver["v"]) and np.allclose(base["gap"], with_driver["gap"])


def test_collision_flag_and_run_continues():
    t = np.arange(100) * idm.DT
    ep = {"exp": "A", "regime": "I", "av_id": 1, "leader_id": 0, "order": [1, 2], "lengths": np.full(2, 4.5),
          "len_l": 4.5, "t": t, "x_l": np.full(100, 20.0), "v_l": np.zeros(100),
          "x0": np.array([10.0, 0.0]), "v0": np.array([10.0, 10.0])}
    out = chain.simulate(ep, [P, P])
    assert out["collided"][0] and out["first_collision_step"][0] >= 0
    assert out["v"].shape == (100, 2) and out["v"].min() >= 0.0


def _table():
    rows = []
    for veh in ("3", "7", "pooled"):
        rows.append(dict(fold="loeo_A", regime="I", variant="nodelay", veh=veh, **P))
        rows.append(dict(fold="loeo_A", regime="I", variant="delay", veh=veh, **dict(P, T=9.9)))
    return pd.DataFrame(rows)


def test_vehicle_params_with_fallback():
    out = cp.vehicle_params(_table(), "loeo_A", "I", [3, 11])
    assert out[3][1] == "vehicle" and out[11][1] == "pooled"
    assert out[3][0]["T"] == 1.2      # nodelay row, not the delay row


def test_fold_for_experiment():
    assert cp.fold_for_experiment("C") == "loeo_C"
    assert cp.fold_for_experiment("H") == "all_dev"
    with pytest.raises(KeyError):
        cp.fold_for_experiment("Z")
