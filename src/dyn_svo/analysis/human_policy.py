"""Human following policy statistics from dev data: desired gap line and cost scales for the MPC
. Computed once by a committed script before any controller result."""
import numpy as np

from dyn_svo.data import loader as ld


def follower_rows(df, t_from_s, step_frames=3, window_s=38.0, data_hz=30):
    """Follower rows at the simulation step with leader state and U = causal mean of the leader
    speed over window_s (same estimator as the controllers, computed over the full history)."""
    w = ld.with_leader(df).sort_values(["veh_id", "k"])
    w = w[w["k"] % step_frames == 0].copy()
    n = max(1, int(round(window_s * data_hz / step_frames)))
    w["U"] = w.groupby("veh_id")["v_l"].transform(lambda s: s.rolling(n, min_periods=1).mean())
    return w[w["follower_ok"] & (w["t"] >= t_from_s)]


def policy_from_rows(rows):
    """Per regime: gap = s0 + h v (least squares) and the scales used to normalise MPC costs."""
    out = {}
    for regime, g in rows.groupby("regime"):
        v = g["v"].to_numpy()
        gap = g["gap"].to_numpy()
        X = np.column_stack([np.ones_like(v), v])
        (s0, h), *_ = np.linalg.lstsq(X, gap, rcond=None)
        out[str(regime)] = {
            "s0": float(s0), "h": float(h),
            "sigma_gap": float(np.std(gap - (s0 + h * v))),
            "sigma_dv": float(np.std(v - g["v_l"].to_numpy())),
            "sigma_a": float(np.std(g["a"].to_numpy())),
            "sigma_u": float(np.std(v - g["U"].to_numpy())),
            "n_rows": int(len(g))}
    return out
