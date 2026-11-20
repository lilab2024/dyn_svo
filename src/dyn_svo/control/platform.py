"""AV platform shared by every controller: command conversion, Gipps safety filter, actuation
.

controller.act(obs) returns {"a": acceleration} or {"v": target speed}. The platform is itself an
AV-slot driver for sim.chain (human = False, so no human noise is added to it).
"""
import math

import numpy as np

from dyn_svo.config import load_config
from dyn_svo.sim import idm

_C = load_config("av")


def gipps_safe_speed(gap, v_leader, s_min, decel_av, decel_leader, reaction_s):
    """Highest speed from which the AV can still stop behind a leader that brakes at decel_leader
    (Gipps 1981, deceleration part, positive decelerations). gap = bumper-to-bumper gap."""
    d, tau = decel_av, reaction_s
    rad = d * d * tau * tau + d * (2.0 * (gap - s_min) + v_leader * v_leader / decel_leader)
    return max(0.0, -d * tau + math.sqrt(rad)) if rad > 0 else 0.0


class AVPlatform:
    """Wraps a controller; act(obs) returns the actual AV acceleration for this step."""

    human = False

    def __init__(self, controller, s_min, cfg=None, dt=None):
        self.c = cfg if cfg is not None else _C
        self.ctrl = controller
        self.s_min = float(s_min)                     # G1b threshold (guardrail), unchanged
        self.s_floor = self.s_min + float(self.c["safety"].get("s_min_buffer_m", 0.0))
        self.t_conv = float(self.c["safety"]["reaction_s"]) - float(self.c["actuation_lag_s"])
        if self.t_conv <= 0.0:
            raise ValueError("reaction_s must exceed actuation_lag_s ")
        self.dt = idm.DT if dt is None else float(dt)
        self.reset_state()

    def reset_state(self):
        self.a_act = 0.0
        self.a_cmd_prev = 0.0
        self.n = 0
        self.n_safety = 0
        self.n_clip = 0

    def reset(self, info):
        self.reset_state()
        self.ctrl.reset(info)

    def command(self, obs):
        out = self.ctrl.act(obs)
        if "a" in out:
            return float(out["a"])
        return (float(out["v"]) - obs["v"]) / float(self.c["speed_tracking_tau_s"])

    def act(self, obs):
        c, s = self.c, self.c["safety"]
        a_ctrl = self.command(obs)
        a_b = min(max(a_ctrl, c["accel_min"]), c["accel_max"])
        clipped = a_b != a_ctrl
        step = c["jerk_max"] * self.dt
        a_j = min(max(a_b, self.a_cmd_prev - step), self.a_cmd_prev + step)
        v_safe = gipps_safe_speed(obs["gap"], obs["v_leader"], self.s_floor, s["decel_av"],
                                  s["decel_leader"], s["reaction_s"])
        # reach v_safe within reaction_s including the actuation lag
        a_safe = max((v_safe - obs["v"]) / self.t_conv, -s["emergency_decel"])
        if a_safe < a_j:          # the filter can only lower the jerk-limited command
            a_cmd = a_safe
            self.n_safety += 1
        else:
            a_cmd = a_j
            clipped = clipped or a_cmd != a_b
        self.n_clip += int(clipped)
        self.a_cmd_prev = a_cmd
        self.a_act += (a_cmd - self.a_act) * min(1.0, self.dt / c["actuation_lag_s"])
        self.n += 1
        return self.a_act

    def stats(self):
        n = max(self.n, 1)
        return {"steps": self.n, "safety_share": self.n_safety / n, "clip_share": self.n_clip / n}


def s_min_for_regime(thresholds, regime):
    """Minimum gap of the safety filter = G1b threshold of the regime."""
    return float(thresholds[regime]["min_gap"])


def load_thresholds(run_name=None):
    import json
    from dyn_svo.data import manifest as mf
    run_name = run_name or load_config("metrics")["guardrail_run"]
    p = mf.data_root() / "dyn_svo" / "results" / "metrics" / run_name / "thresholds.json"
    return json.loads(p.read_text(encoding="utf-8"))["thresholds"]
