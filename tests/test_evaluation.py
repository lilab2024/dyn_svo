import numpy as np
import pandas as pd
import pytest

from dyn_svo.config import load_config
from dyn_svo.metrics import evaluation as ev
from dyn_svo.metrics import oscillation as osc

DT = osc.DT


def _v(n_veh=4, steps=1200, amp=None):
    t = np.arange(steps) * DT
    amp = amp if amp is not None else [1.0] * n_veh
    return np.column_stack([5.0 + a * np.sin(2 * np.pi * t / 25.0 + i) for i, a in enumerate(amp)])


def test_metrics_config_guardrails():
    c = load_config("metrics")
    assert c["guardrail_fail_rate_max"] == 0.01
    assert c["guardrail_percentiles"] == {"min_gap": 1, "speed_ratio": 1, "mean_gap": 99}
    assert c["guardrail_run"] == "guardrails_v1"


def test_for_excludes_av_and_is_one_for_identical_runs():
    base = {"v": _v()}
    assert ev.for_seed(base, base) == pytest.approx(1.0)
    av_changed = {"v": _v(amp=[0.0, 1.0, 1.0, 1.0])}
    assert ev.for_seed(av_changed, base) == pytest.approx(1.0)       # column 0 is the AV
    half = {"v": _v(amp=[1.0, 0.5, 0.5, 0.5])}
    assert ev.for_seed(half, base) == pytest.approx(0.5, rel=1e-6)


def test_episode_log_for_pairs_by_seed():
    base = [{"v": _v()}, {"v": _v(amp=[1, 2, 2, 2])}]
    arm = [{"v": _v(amp=[1, 0.5, 0.5, 0.5])}, {"v": _v(amp=[1, 1, 1, 1])}]
    assert ev.episode_log_for(arm, base) == pytest.approx(np.log(0.5), rel=1e-6)


def test_av_window_stats_and_checks():
    steps = 1200
    ep = {"v_l": np.full(steps, 5.0)}
    sim = {"gap": np.column_stack([np.linspace(8, 12, steps), np.full(steps, 9.0)]),
           "v": np.column_stack([np.full(steps, 4.9), np.full(steps, 5.0)]), "collided": np.array([False, False])}
    s = ev.av_window_stats(ep, sim)
    n0 = int(round(osc.WARMUP_S / DT))
    assert s["min_gap"] == pytest.approx(np.linspace(8, 12, steps)[n0])
    assert s["speed_ratio"] == pytest.approx(0.98) and not s["collision"]
    thr = {"min_gap": 2.0, "speed_ratio": 0.97, "mean_gap": 15.0}
    assert ev.guardrail_checks(s, thr) == {"g1a": True, "g1b": True, "g2": True, "g3": True}
    assert not ev.guardrail_checks(dict(s, speed_ratio=0.9), thr)["g2"]
    assert not ev.guardrail_checks(dict(s, mean_gap=20.0), thr)["g3"]
    assert not ev.guardrail_checks(dict(s, collision=True), thr)["g1a"]


def test_config_eligibility_rule():
    ok = {"g1a": True, "g1b": True, "g2": True, "g3": True}
    checks = [ok] * 199 + [dict(ok, g2=False)]
    assert ev.config_eligible(checks)[0]                       # 0.5% failures
    assert not ev.config_eligible([ok] * 196 + [dict(ok, g3=False)] * 4)[0]   # 2%
    assert not ev.config_eligible([ok] * 199 + [dict(ok, g1a=False)])[0]      # any collision


def test_human_window_stats_and_thresholds():
    rows = []
    for exp, regime in (("A", "I"), ("F", "II")):
        for veh, lead, g0 in ((1, 2, 10.0), (2, 1, 14.0)):
            t = np.arange(0, 120, 1 / 30)
            rows.append(pd.DataFrame({"exp": exp, "regime": regime, "veh_id": veh, "leader_id": lead,
                                      "k": np.arange(len(t)), "t": t, "x": 5.0 * t + veh, "v": 5.0,
                                      "a": 0.0, "length": 4.5, "gap": g0 + np.sin(t),
                                      "follower_ok": t < 100}))
    df = pd.concat(rows, ignore_index=True)
    st = pd.concat([ev.human_window_stats(df[df["exp"] == e], 60.0) for e in ("A", "F")], ignore_index=True)
    assert len(st) == 4 and set(st["regime"]) == {"I", "II"}
    assert st["n"].iloc[0] == 1200                             # 60 s <= t < 100 s at 30 Hz
    thr = ev.thresholds_from_humans(st)
    assert thr["I"]["speed_ratio"] == pytest.approx(1.0)
    assert thr["I"]["min_gap"] < thr["I"]["mean_gap"]
