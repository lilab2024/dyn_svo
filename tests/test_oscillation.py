import numpy as np
import pytest

from dyn_svo.metrics import oscillation as osc
from dyn_svo.sim import chain, idm

DT = osc.DT


def _expected(amplitude, period_s, window_s):
    """Std of a sine minus its centred moving average: A / sqrt(2) * |1 - sinc|."""
    w = osc._odd_steps(window_s, DT) * DT
    x = np.pi * w / period_s
    return amplitude / np.sqrt(2) * abs(1.0 - np.sin(x) / x)


def test_sigma_band_sine_and_trend():
    t = np.arange(0, 600, DT)
    s = np.sin(2 * np.pi * t / 20.0)
    assert osc.sigma_band(s, window_s=70) == pytest.approx(_expected(1.0, 20.0, 70), rel=0.02)
    assert osc.sigma_band(s + 0.01 * t, window_s=70) == pytest.approx(_expected(1.0, 20.0, 70), rel=0.02)


def test_sigma_band_constant_and_columns():
    t = np.arange(0, 300, DT)
    v = np.column_stack([np.full_like(t, 4.0), 2 * np.sin(2 * np.pi * t / 30.0)])
    out = osc.sigma_band(v, window_s=70)
    assert out.shape == (2,) and out[0] == pytest.approx(0.0, abs=1e-12)
    assert out[1] == pytest.approx(_expected(2.0, 30.0, 70), rel=0.02)


def test_warmup_excluded():
    v = np.zeros(1000)
    v[:200] = np.sin(np.arange(200))          # only inside the first 20 s
    assert osc.sigma_band(v, window_s=10, warmup_s=30) == pytest.approx(0.0, abs=1e-12)


def test_amplification_slope():
    g = 1.05
    sig = 0.5 * g ** np.arange(1, 11)
    assert osc.amplification_slope(sig, 0.5) == pytest.approx(np.log(g))
    assert np.isnan(osc.amplification_slope(np.array([0.1, 0.0]), 0.5))


def test_episode_validation_on_identical_chain():
    P = {"v0": 15.0, "T": 1.2, "s0": 2.0, "a_max": 1.0, "b": 1.5}
    t = np.arange(0, 200, DT)
    v_l = 5.0 + np.sin(2 * np.pi * t / 25.0)
    x_l = 200.0 + np.concatenate([[0.0], np.cumsum(0.5 * (v_l[1:] + v_l[:-1]) * DT)])
    n = 4
    s_e = float(idm.equilibrium_gap(5.0, **P))
    ep = {"exp": "A", "regime": "I", "av_id": 1, "leader_id": 0, "order": list(range(1, n + 1)),
          "lengths": np.full(n, 4.5), "len_l": 4.5, "t": t, "x_l": x_l, "v_l": v_l,
          "x0": x_l[0] - np.arange(1, n + 1) * (4.5 + s_e), "v0": np.full(n, 5.0)}
    sim = chain.simulate(ep, [P] * n)
    ep["real_v"] = sim["v"].copy()
    row = osc.episode_validation(ep, sim)
    assert row["mean_log_ratio"] == pytest.approx(0.0, abs=1e-12)
    assert row["slope_sim"] == pytest.approx(row["slope_real"])
    s = osc.summarize_validation([row])[0]
    assert s["ok_sigma"] and s["ok_slope"] and s["ok_speed"] and s["n_collided"] == 0


def test_summary_flags():
    base = {"exp": "A", "av_id": 1, "n_veh": 2, "sigma_leader": 1.0, "mean_log_ratio": 0.0,
            "speed_real": 4.0, "n_collided": 0}
    rows = [dict(base, slope_sim=0.05, slope_real=0.01, speed_sim=4.4,
                 log_ratio_by_position=[np.log(2.0), np.log(2.0)])]
    s = osc.summarize_validation(rows)[0]
    assert s["geo_mean_sigma_ratio"] == pytest.approx(2.0)
    assert not s["ok_sigma"] and not s["ok_slope"] and not s["ok_speed"]
