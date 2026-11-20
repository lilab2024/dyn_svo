"""Oscillation metrics shared by validation and controller evaluation.

sigma_band: the trend is a centred moving average of window W (shrinking at the series ends),
computed on the whole episode; the standard deviation of the residual is taken after warm-up.
"""
import numpy as np
import pandas as pd

from dyn_svo.config import load_config

_M = load_config("metrics")
_S = load_config("sim")
DT = float(_S["dt"])
W_S = float(_M["detrend_window_s"])
WARMUP_S = float(_M["warmup_s"])


def _odd_steps(window_s, dt):
    w = int(round(window_s / dt))
    return w + 1 if w % 2 == 0 else w


def sigma_band(v, window_s=W_S, warmup_s=WARMUP_S, dt=DT):
    """Detrended speed std per column of v (T,) or (T, N)."""
    v = np.asarray(v, dtype=float)
    frame = pd.DataFrame(v.reshape(len(v), -1))
    trend = frame.rolling(_odd_steps(window_s, dt), center=True, min_periods=1).mean().to_numpy()
    resid = frame.to_numpy() - trend
    n0 = int(round(warmup_s / dt))
    out = resid[n0:].std(axis=0)
    return out if v.ndim == 2 else float(out[0])


def amplification_slope(sig_positions, sig_leader):
    """Slope of log(sigma_k / sigma_leader) against position k = 1..N (per vehicle)."""
    s = np.asarray(sig_positions, dtype=float)
    if sig_leader <= 0 or np.any(s <= 0):
        return float("nan")
    k = np.arange(1, len(s) + 1, dtype=float)
    return float(np.polyfit(k, np.log(s / sig_leader), 1)[0])


def chain_mean_speed(v, warmup_s=WARMUP_S, dt=DT):
    """Mean speed over all chain vehicles after warm-up."""
    return float(np.asarray(v, dtype=float)[int(round(warmup_s / dt)):].mean())


def episode_validation(ep, sim):
    """ per-episode comparison of an all-HDV simulated chain with the real chain."""
    sig_sim = sigma_band(sim["v"])
    sig_real = sigma_band(ep["real_v"])
    sig_l = sigma_band(ep["v_l"])
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = np.log(sig_sim / sig_real)
    return {"exp": ep["exp"], "av_id": ep["av_id"], "n_veh": len(ep["order"]),
            "sigma_leader": sig_l, "mean_log_ratio": float(np.mean(log_ratio)),
            "slope_sim": amplification_slope(sig_sim, sig_l),
            "slope_real": amplification_slope(sig_real, sig_l),
            "speed_sim": chain_mean_speed(sim["v"]), "speed_real": chain_mean_speed(ep["real_v"]),
            "n_collided": int(np.sum(sim["collided"])),
            "log_ratio_by_position": [float(x) for x in log_ratio]}


def summarize_validation(rows):
    """Per-experiment summary with adequacy flags."""
    df = pd.DataFrame(rows)
    out = []
    for exp, g in df.groupby("exp"):
        all_lr = np.concatenate([np.asarray(x, dtype=float) for x in g["log_ratio_by_position"]])
        finite = np.isfinite(all_lr)
        gm = float(np.exp(np.mean(all_lr[finite]))) if finite.any() else float("nan")
        s_sim, s_real = float(g["slope_sim"].mean()), float(g["slope_real"].mean())
        spd = float(g["speed_sim"].mean() / g["speed_real"].mean() - 1.0)
        out.append({"exp": exp, "n_episodes": int(len(g)), "geo_mean_sigma_ratio": gm,
                    "slope_sim": s_sim, "slope_real": s_real, "slope_abs_diff": abs(s_sim - s_real),
                    "speed_rel_diff": spd, "n_collided": int(g["n_collided"].sum()),
                    "n_nonfinite_ratios": int((~finite).sum()),
                    "ok_sigma": bool(2.0 / 3.0 <= gm <= 1.5),
                    "ok_slope": bool(abs(s_sim - s_real) <= 0.02),
                    "ok_speed": bool(abs(spd) <= 0.05)})
    return out


def combine_seed_rows(rows):
    """Average per-seed episode_validation rows of one episode."""
    lr = np.mean([np.asarray(r["log_ratio_by_position"], dtype=float) for r in rows], axis=0)
    out = dict(rows[0])
    out.update({"mean_log_ratio": float(np.mean([r["mean_log_ratio"] for r in rows])),
                "slope_sim": float(np.nanmean([r["slope_sim"] for r in rows])),
                "speed_sim": float(np.mean([r["speed_sim"] for r in rows])),
                "n_collided": int(sum(r["n_collided"] for r in rows)),
                "n_seeds": len(rows), "log_ratio_by_position": [float(x) for x in lr]})
    return out
