"""FollowerStopper and PI with saturation, as published in Stern et al. (2018) Sec. 3.

Both return velocity commands; the shared AV platform turns them into accelerations.
obs: v (AV speed), gap (bumper-to-bumper, m), v_leader.
"""
from dyn_svo.config import load_config
from dyn_svo.control.estimator import MeanSpeedEstimator
from dyn_svo.sim import idm

_B = load_config("baselines")


def fs_boundaries(dv, dx0, d):
    """Region boundaries dx_k = dx0_k + dv_minus^2 / (2 d_k), dv = v_lead - v_AV."""
    dvm = min(dv, 0.0)
    return [x0 + dvm * dvm / (2.0 * dk) for x0, dk in zip(dx0, d)]


def fs_command(gap, v_av, v_lead, U, dx0, d):
    """FollowerStopper commanded velocity, Stern et al. (2018) eq. (1)-(2)."""
    x1, x2, x3 = fs_boundaries(v_lead - v_av, dx0, d)
    v = min(max(v_lead, 0.0), U)
    if gap <= x1:
        return 0.0
    if gap <= x2:
        return v * (gap - x1) / (x2 - x1)
    if gap <= x3:
        return v + (U - v) * (gap - x2) / (x3 - x2)
    return U


class FollowerStopper:
    def __init__(self, cfg=None, estimator=None):
        self.c = (cfg or _B)["follower_stopper"]
        self.est = estimator or MeanSpeedEstimator()
        self.last_U = None

    def reset(self, info):
        self.est.reset()

    def act(self, obs):
        self.last_U = self.est.update(obs["v_leader"])
        return {"v": fs_command(obs["gap"], obs["v"], obs["v_leader"], self.last_U,
                                self.c["dx0"], self.c["d"])}


def pi_weights(gap, v_av, v_lead, c):
    """alpha, beta and the safety distance of the PI controller, eq. (5) and Sec. 3.2."""
    dxs = max(c["dxs_time_s"] * (v_lead - v_av), c["dxs_min"])
    alpha = min(max((gap - dxs) / c["gamma"], 0.0), 1.0)
    return alpha, 1.0 - 0.5 * alpha, dxs


def pi_target(gap, U, c):
    """Target velocity, eq. (3)."""
    return U + c["v_catch"] * min(max((gap - c["g_l"]) / (c["g_u"] - c["g_l"]), 0.0), 1.0)


class PISaturation:
    def __init__(self, cfg=None, estimator=None, dt=None):
        self.c = (cfg or _B)["pi_saturation"]
        self.est = estimator or MeanSpeedEstimator()
        dt = idm.DT if dt is None else float(dt)
        self.substeps = max(1, int(round(self.c["update_hz"] * dt)))
        self.v_cmd = None
        self.last_U = None

    def reset(self, info):
        self.est.reset()
        self.v_cmd = None

    def act(self, obs):
        self.last_U = self.est.update(obs["v"])
        if self.v_cmd is None:
            self.v_cmd = float(obs["v"])
        alpha, beta, _ = pi_weights(obs["gap"], obs["v"], obs["v_leader"], self.c)
        v_target = pi_target(obs["gap"], self.last_U, self.c)
        for _ in range(self.substeps):          # inputs held within the 0.1 s step (eq. 4)
            self.v_cmd = beta * (alpha * v_target + (1.0 - alpha) * obs["v_leader"]) + (1.0 - beta) * self.v_cmd
        return {"v": self.v_cmd}
