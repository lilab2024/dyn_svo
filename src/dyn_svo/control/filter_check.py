"""Fixed synthetic braking grid for the AV safety filter.

The buffer between the filter floor and the G1b threshold is the smallest multiple of BUFFER_STEP_M
for which the worst gap margin over this grid is at least 0, using the real AVPlatform and
chain.simulate. It is never derived from evaluation results.
"""
import copy

import numpy as np

from dyn_svo.control import platform as pf
from dyn_svo.sim import chain, idm

LEADER_DECEL = (0.5, 1.0, 2.0, 3.0, 3.5)    # m/s2
V_START = (4.0, 8.0, 12.0)                  # m/s
V_END = (0.0, 0.5, 1.0)                     # m/s
BRAKE_AT_S, RECOVER_AT_S, DURATION_S = 2.0, 15.0, 40.0
GAP0 = (3.511, 1.347)                       # initial gap s0 + h v (human_policy_v1, regime I)
BUFFER_STEP_M, BUFFER_MAX_M = 0.05, 1.0
LENGTH_M = 4.5
P_SLOT = {"v0": 30.0, "T": 1.0, "s0": 2.0, "a_max": 1.0, "b": 1.5}   # IDM for slot 1; slot 0 is the AV


class HoldSpeed:
    """Ignores the gap and keeps the start speed: the most demanding controller for the filter."""

    def __init__(self, v):
        self.v = float(v)

    def reset(self, info):
        pass

    def act(self, obs):
        return {"v": self.v}


class HoldAccel:
    def reset(self, info):
        pass

    def act(self, obs):
        return {"a": 0.0}


def leader(v_start, v_end, decel, dt=None):
    """Brake at decel from v_start to v_end at BRAKE_AT_S, recover at decel/2 from RECOVER_AT_S."""
    dt = idm.DT if dt is None else dt
    n = int(round(DURATION_S / dt))
    t = np.arange(n) * dt
    v, x = np.empty(n), np.empty(n)
    v[0], x[0] = v_start, 0.0
    for k in range(n - 1):
        if BRAKE_AT_S <= t[k] < RECOVER_AT_S:
            v1 = max(v[k] - decel * dt, v_end)
        elif t[k] >= RECOVER_AT_S:
            v1 = min(v[k] + 0.5 * decel * dt, v_start)
        else:
            v1 = v[k]
        x[k + 1] = x[k] + 0.5 * (v[k] + v1) * dt
        v[k + 1] = v1
    return t, x, v


def episode(v_start, v_end, decel):
    t, x, v = leader(v_start, v_end, decel)
    x_l = 200.0 + x
    gap0 = GAP0[0] + GAP0[1] * v_start
    x0 = x_l[0] - LENGTH_M - gap0
    return {"exp": "SYN", "regime": "I", "av_id": 1, "leader_id": 0, "order": [1, 2],
            "lengths": np.full(2, LENGTH_M), "len_l": LENGTH_M, "t": t, "x_l": x_l, "v_l": v,
            "x0": np.array([x0, x0 - LENGTH_M - gap0]), "v0": np.array([v_start, v_start])}


def worst_margin(threshold, buffer_m, cfg=None):
    """Smallest (AV gap - threshold) over the grid; -inf if the AV collides."""
    c = copy.deepcopy(cfg if cfg is not None else pf._C)
    c["safety"]["s_min_buffer_m"] = float(buffer_m)
    worst = np.inf
    for decel in LEADER_DECEL:
        for v_s in V_START:
            for v_e in V_END:
                ep = episode(v_s, v_e, decel)
                for ctrl in (HoldSpeed(v_s), HoldAccel()):
                    out = chain.simulate(ep, [P_SLOT, P_SLOT], av_driver=pf.AVPlatform(ctrl, s_min=threshold, cfg=c))
                    if out["collided"][0]:
                        return -np.inf
                    worst = min(worst, float(out["gap"][:, 0].min()) - float(threshold))
    return worst


def smallest_buffer(threshold, cfg=None):
    """Smallest multiple of BUFFER_STEP_M with worst_margin >= 0."""
    k = 0
    while k * BUFFER_STEP_M <= BUFFER_MAX_M + 1e-9:
        b = round(k * BUFFER_STEP_M, 2)
        if worst_margin(threshold, b, cfg) >= 0.0:
            return b
        k += 1
    raise ValueError("no buffer up to BUFFER_MAX_M keeps the margin non-negative")
