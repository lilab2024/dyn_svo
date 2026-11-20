"""IDM car-following model with optional reaction delay and ballistic update.

Conventions: x is the front-bumper position; gap = x_leader - length_leader - x;
dv = v - v_leader is the approach rate (positive when closing in).
"""
import numpy as np

from dyn_svo.config import load_config

_CFG = load_config("sim")
DT = float(_CFG["dt"])
DELTA = float(_CFG["idm_delta"])
ACCEL_FLOOR = float(_CFG["accel_floor"])
MIN_GAP_EPS = float(_CFG["min_gap_eps"])
PARAM_NAMES = ("v0", "T", "s0", "a_max", "b")


def idm_accel(v, s, dv, v0, T, s0, a_max, b, delta=DELTA):
    """IDM acceleration, clipped below at the physical floor. All inputs broadcast."""
    v = np.asarray(v, dtype=float)
    s_star = s0 + np.maximum(0.0, v * T + v * dv / (2.0 * np.sqrt(a_max * b)))
    s_eff = np.maximum(s, MIN_GAP_EPS)
    acc = a_max * (1.0 - (v / v0) ** delta - (s_star / s_eff) ** 2)
    return np.maximum(acc, ACCEL_FLOOR)


def ballistic_step(x, v, a, dt=DT):
    """One ballistic step (constant a within the step); a vehicle that would reverse stops."""
    x = np.asarray(x, dtype=float)
    v = np.asarray(v, dtype=float)
    a = np.asarray(a, dtype=float)
    v_new = v + a * dt
    stops = v_new < 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        dx_stop = np.where(a < 0.0, -v * v / (2.0 * a), 0.0)
    dx = np.where(stops, dx_stop, v * dt + 0.5 * a * dt * dt)
    return x + dx, np.maximum(v_new, 0.0)


def simulate_follower(x_l, v_l, len_l, x_init, v_init, params, delay_steps=0, dt=DT):
    """Simulate one follower behind replayed leaders, many segments at once.

    x_l, v_l: (S, N) leader front position and speed at steps 0..N-1.
    len_l: (S,) or scalar leader length. x_init, v_init: (S,) follower state at step 0.
    params: dict with PARAM_NAMES (scalars or (S,) arrays).
    delay_steps: perception delay in steps; before step d the step-0 state is perceived.
    Returns x, v, a, gap (each (S, N)) and collided (S,) bool (gap <= 0 at any step).
    """
    x_l = np.asarray(x_l, dtype=float)
    v_l = np.asarray(v_l, dtype=float)
    n_seg, n_step = x_l.shape
    len_l = np.broadcast_to(np.asarray(len_l, dtype=float), (n_seg,))
    p = {k: np.asarray(params[k], dtype=float) for k in PARAM_NAMES}
    d = int(delay_steps)

    x = np.empty((n_seg, n_step))
    v = np.empty((n_seg, n_step))
    a = np.empty((n_seg, n_step))
    x[:, 0] = x_init
    v[:, 0] = v_init
    for n in range(n_step):
        m = max(n - d, 0)
        s_seen = x_l[:, m] - len_l - x[:, m]
        a[:, n] = idm_accel(v[:, m], s_seen, v[:, m] - v_l[:, m],
                            p["v0"], p["T"], p["s0"], p["a_max"], p["b"])
        if n + 1 < n_step:
            x[:, n + 1], v[:, n + 1] = ballistic_step(x[:, n], v[:, n], a[:, n], dt)
    gap = x_l - len_l[:, None] - x
    collided = (gap <= 0.0).any(axis=1)
    return x, v, a, gap, collided


def equilibrium_gap(v, v0, T, s0, a_max, b, delta=DELTA):
    """Steady-state IDM gap at speed v (dv = 0); requires v < v0."""
    return (s0 + v * T) / np.sqrt(1.0 - (v / v0) ** delta)
