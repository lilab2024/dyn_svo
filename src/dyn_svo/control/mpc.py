"""Unconstrained quadratic MPC with cooperation weight phi.

Prediction at the end of each 0.5 s block over 10 s; leader keeps its current speed; AV is a
double integrator with blockwise-constant acceleration u. Residuals are affine in u:
  v = v0 + db L u,  gap = gap0 + (vL - v0) db k - Mg u,  Mg[k, i] = db^2 (k - i + 1/2), i <= k.
Cost = cos(phi) [ ((gap - s0 - h v)/sg)^2 + ((v - vL)/sdv)^2 ]
     + sin(phi) [ (u/sa)^2 + ((v - U)/su)^2 ], summed over the blocks; closed-form minimiser.
Only the first move is returned (acceleration command for the shared platform).
"""
import json
import math

import numpy as np

from dyn_svo.config import load_config
from dyn_svo.control.estimator import MeanSpeedEstimator

_C = load_config("mpc")
RIDGE = 1e-9


def load_policy(run_name=None):
    from dyn_svo.data import manifest as mf
    run_name = run_name or _C["human_policy_run"]
    p = mf.data_root() / "dyn_svo" / "results" / "mpc" / run_name / "policy.json"
    return json.loads(p.read_text(encoding="utf-8"))["policy"]


def fixed_phi_grid(n=None):
    n = int(_C["fixed_phi_grid_points"]) if n is None else int(n)
    return [i * (math.pi / 2) / (n - 1) for i in range(n)]


class FixedPhi:
    def __init__(self, phi):
        self.phi = float(phi)

    def reset(self):
        pass

    def update(self, obs):
        return self.phi


class MPC:
    def __init__(self, policy, phi_source, horizon_s=None, block_s=None, estimator=None):
        self.p = dict(policy)
        self.phi_source = phi_source
        self.est = estimator or MeanSpeedEstimator()
        self.db = float(_C["block_s"] if block_s is None else block_s)
        n = int(round(float(_C["horizon_s"] if horizon_s is None else horizon_s) / self.db))
        k = np.arange(1, n + 1, dtype=float)
        self.k = k
        self.L = np.tril(np.ones((n, n)))
        self.Mg = np.tril(self.db ** 2 * (k[:, None] - k[None, :] + 0.5))
        self.A_g = -(self.Mg + self.p["h"] * self.db * self.L)
        self.A_v = self.db * self.L
        sg, sdv, sa, su = (self.p[s] for s in ("sigma_gap", "sigma_dv", "sigma_a", "sigma_u"))
        self.w = (1.0 / sg ** 2, 1.0 / sdv ** 2, 1.0 / sa ** 2, 1.0 / su ** 2)
        self.H_ego = self.w[0] * self.A_g.T @ self.A_g + self.w[1] * self.A_v.T @ self.A_v
        self.H_soc = self.w[2] * np.eye(n) + self.w[3] * self.A_v.T @ self.A_v
        self.last_phi = None
        self.last_U = None

    def reset(self, info):
        self.est.reset()
        self.phi_source.reset()

    def plan(self, gap, v0, vL, U, phi):
        """Optimal block accelerations (closed form)."""
        b_g = gap + (vL - v0) * self.db * self.k - self.p["s0"] - self.p["h"] * v0
        b_v = np.full(len(self.k), v0 - vL)
        b_u = np.full(len(self.k), v0 - U)
        c, s = math.cos(phi), math.sin(phi)
        H = c * self.H_ego + s * self.H_soc + RIDGE * np.eye(len(self.k))
        g = c * (self.w[0] * self.A_g.T @ b_g + self.w[1] * self.A_v.T @ b_v) + s * (self.w[3] * self.A_v.T @ b_u)
        return np.linalg.solve(H, -g)

    def cost(self, u, gap, v0, vL, U, phi):
        """Objective value (for tests)."""
        v = v0 + self.A_v @ u
        s = gap + (vL - v0) * self.db * self.k - self.Mg @ u
        ego = self.w[0] * np.sum((s - self.p["s0"] - self.p["h"] * v) ** 2) + self.w[1] * np.sum((v - vL) ** 2)
        soc = self.w[2] * np.sum(u ** 2) + self.w[3] * np.sum((v - U) ** 2)
        return math.cos(phi) * ego + math.sin(phi) * soc

    def act(self, obs):
        self.last_U = self.est.update(obs["v_leader"])
        self.last_phi = self.phi_source.update(obs)
        u = self.plan(obs["gap"], obs["v"], obs["v_leader"], self.last_U, self.last_phi)
        return {"a": float(u[0])}
