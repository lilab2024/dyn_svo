import numpy as np
import pandas as pd
import pytest

from dyn_svo.calibration import fit, segments
from dyn_svo.config import load_config
from dyn_svo.sim import idm

P_TRUE = {"v0": 12.0, "T": 1.4, "s0": 2.2, "a_max": 0.9, "b": 1.6}
FAST = {"segment_s": 30, "bounds": load_config("calibration")["bounds"], "delay_steps_max": 12,
        "collision_penalty_m": 1000.0, "de": {"popsize": 10, "maxiter": 40, "tol": 1e-8, "seed": 0}}


def _leader(n_step, phase):
    """Smooth oscillation plus sharp brake / release events (makes the delay identifiable)."""
    t = np.arange(n_step) * idm.DT
    v = 4.0 + 1.5 * np.sin(2 * np.pi * t / 25.0 + phase)
    for t_ev in (5.0 + 3 * phase, 18.0 + 3 * phase):
        v = np.where((t >= t_ev) & (t < t_ev + 1.5), v - 2.0 * (t - t_ev), v)
        v = np.where((t >= t_ev + 1.5) & (t < t_ev + 3.0), v - 2.0 * (t_ev + 3.0 - t), v)
    v = np.maximum(v, 0.0)
    x = 30.0 + np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * idm.DT)])
    return x, v


def _synthetic_segments(params, delay, n_seg=4, n_step=300):
    xl, vl = zip(*[_leader(n_step, ph) for ph in np.linspace(0, 2, n_seg)])
    xl, vl = np.array(xl), np.array(vl)
    len_l = np.full(n_seg, 5.0)
    v0 = vl[:, 0]
    s0 = np.array([idm.equilibrium_gap(v, **params) for v in v0])
    x0 = xl[:, 0] - len_l - s0
    x, v, _, gap, col = idm.simulate_follower(xl, vl, len_l, x0, v0, params, delay_steps=delay)
    assert not col.any()
    return {"x_l": xl, "v_l": vl, "x": x, "v": v, "gap": gap, "len_l": len_l,
            "k_start": np.arange(n_seg)}


def test_calibration_config_matches_d029():
    c = load_config("calibration")
    assert c["segment_s"] == 30 and c["delay_steps_max"] == 12
    assert c["bounds"]["v0"] == [5.0, 30.0] and c["bounds"]["T"] == [0.1, 3.0]
    assert c["de"]["popsize"] == 15 and c["de"]["maxiter"] == 200 and c["de"]["seed"] == 0


def test_zero_rmse_at_true_params():
    seg = _synthetic_segments(P_TRUE, 0)
    rmse, col = fit.spacing_rmse(seg, P_TRUE, 0)
    assert rmse < 1e-9 and col == 0.0


def test_recovers_parameters_without_delay():
    seg = _synthetic_segments(P_TRUE, 0)
    out = fit.fit_idm(seg, with_delay=False, cfg=FAST)
    assert out["rmse_m"] < 0.05
    assert out["params"]["T"] == pytest.approx(P_TRUE["T"], rel=0.10)
    assert out["params"]["s0"] == pytest.approx(P_TRUE["s0"], rel=0.15)
    assert out["delay_steps"] == 0


def test_recovers_delay():
    seg = _synthetic_segments(P_TRUE, 6)
    out = fit.fit_idm(seg, with_delay=True, cfg=FAST)
    assert out["rmse_m"] < 0.05
    assert abs(out["delay_steps"] - 6) <= 1


def test_fit_is_deterministic():
    seg = _synthetic_segments(P_TRUE, 0, n_seg=2, n_step=150)
    quick = dict(FAST, de=dict(FAST["de"], maxiter=8))
    a = fit.fit_idm(seg, with_delay=False, cfg=quick)
    b = fit.fit_idm(seg, with_delay=False, cfg=quick)
    assert a["params"] == b["params"] and a["rmse_m"] == b["rmse_m"]


def _pairs_frame(k_values, veh=3):
    k = np.asarray(k_values)
    return pd.DataFrame({"veh_id": veh, "k": k, "x_l": k * 0.1 + 20.0, "v_l": 3.0, "x": k * 0.1,
                         "v": 3.0, "gap": 15.0, "len_l": 5.0})


def test_segments_cut_and_downsample():
    pairs = _pairs_frame(np.arange(0, 2000))
    seg = segments.vehicle_segments(pairs, 3, segment_s=30)
    assert segments.n_segments(seg) == 2
    assert seg["x_l"].shape == (2, 300)
    assert list(seg["k_start"]) == [0, 900]
    assert seg["x"][0, 1] - seg["x"][0, 0] == pytest.approx(0.3)


def test_segments_respect_gaps_in_frames():
    k = np.concatenate([np.arange(0, 1000), np.arange(1500, 2450)])
    seg = segments.vehicle_segments(_pairs_frame(k), 3, segment_s=30)
    assert list(seg["k_start"]) == [0, 1500]


def test_stack():
    a = segments.vehicle_segments(_pairs_frame(np.arange(0, 1800)), 3, segment_s=30)
    s = segments.stack([a, a])
    assert segments.n_segments(s) == 4 and s["len_l"].shape == (4,)
