"""Primary metric FOR and guardrails G1-G3."""
import numpy as np
import pandas as pd

from dyn_svo.config import load_config
from dyn_svo.data import loader as ld
from dyn_svo.metrics import oscillation as osc

_M = load_config("metrics")
FAIL_RATE_MAX = float(_M["guardrail_fail_rate_max"])
PCT = _M["guardrail_percentiles"]


def follower_sigma_sum(v):
    """Sum of sigma_band over the vehicles behind the AV (chain columns 1..N-1)."""
    return float(np.sum(osc.sigma_band(np.asarray(v, dtype=float)[:, 1:])))


def for_seed(sim_arm, sim_hdv):
    """FOR of one seed: followers' sigma sum with the arm / same seed all-HDV."""
    return follower_sigma_sum(sim_arm["v"]) / follower_sigma_sum(sim_hdv["v"])


def episode_log_for(sims_arm, sims_hdv):
    """Episode value: mean over seeds of log FOR (paired by seed)."""
    return float(np.mean([np.log(for_seed(a, h)) for a, h in zip(sims_arm, sims_hdv)]))


def av_window_stats(ep, sim, warmup_s=osc.WARMUP_S, dt=osc.DT):
    """AV statistics over the evaluation window (after warm-up)."""
    n0 = int(round(warmup_s / dt))
    gap, v = sim["gap"][n0:, 0], sim["v"][n0:, 0]
    return {"collision": bool(np.any(sim["collided"])), "min_gap": float(gap.min()),
            "speed_ratio": float(v.mean() / np.mean(ep["v_l"][n0:])), "mean_gap": float(gap.mean())}


def guardrail_checks(stats, thr):
    """Pass flags of one episode-seed for G1a, G1b, G2, G3 (thr: thresholds of the regime)."""
    return {"g1a": not stats["collision"], "g1b": stats["min_gap"] >= thr["min_gap"],
            "g2": stats["speed_ratio"] >= thr["speed_ratio"], "g3": stats["mean_gap"] <= thr["mean_gap"]}


def config_eligible(checks, fail_rate_max=FAIL_RATE_MAX):
    """Zero collisions; G1b, G2, G3 each failing in at most fail_rate_max of episode-seeds."""
    c = pd.DataFrame(checks)
    rates = {g: float(1.0 - c[g].mean()) for g in ("g1b", "g2", "g3")}
    ok = bool(c["g1a"].all()) and all(r <= fail_rate_max for r in rates.values())
    return ok, {"collisions": int((~c["g1a"]).sum()), **{"fail_rate_" + g: r for g, r in rates.items()}}


def human_window_stats(df, t_from_s):
    """Per real human vehicle (follower_ok rows, t >= t_from_s): min gap, mean gap, speed ratio."""
    w = ld.with_leader(df)
    w = w[w["follower_ok"] & (w["t"] >= t_from_s)]
    rows = []
    for veh, g in w.groupby("veh_id"):
        rows.append({"exp": str(g["exp"].iloc[0]), "regime": str(g["regime"].iloc[0]), "veh_id": int(veh),
                     "n": int(len(g)), "min_gap": float(g["gap"].min()), "mean_gap": float(g["gap"].mean()),
                     "speed_ratio": float(g["v"].mean() / g["v_l"].mean())})
    return pd.DataFrame(rows)


def thresholds_from_humans(stats, pct=PCT):
    """Per regime: min_gap p1, speed_ratio p1, mean_gap p99 of the human window statistics."""
    out = {}
    for regime, g in stats.groupby("regime"):
        out[regime] = {"min_gap": float(np.percentile(g["min_gap"], pct["min_gap"])),
                       "speed_ratio": float(np.percentile(g["speed_ratio"], pct["speed_ratio"])),
                       "mean_gap": float(np.percentile(g["mean_gap"], pct["mean_gap"])),
                       "n_vehicle_windows": int(len(g))}
    return out
