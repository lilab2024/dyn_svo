"""Closed-form feedback law of the MPC (Proposition A).

control.mpc.MPC minimises a quadratic in the block accelerations; its minimiser is linear in
    e_s = gap - s0 - h v,   dv = v_L - v,   e_U = v - U,
so the first move is
    a_cmd = k_s(phi) e_s + k_v(phi) dv - k_u(phi) e_U,
with k = -e_1' H(phi)^-1 G(phi) built from the matrices of the MPC instance and the same H as
MPC.plan (ridge included). Dividing the cost by cos(phi) shows that, apart from the ridge, the gains
depend on phi only through rho = tan(phi); feedback_gains_rho gives that ridge-free form.
"""
import math

import numpy as np

from dyn_svo.control import mpc as M


def mpc_from_policy(policy):
    """MPC instance for gain computations (the phi source is irrelevant here)."""
    return M.MPC(policy, M.FixedPhi(0.0))


def _rhs(m, c, s):
    one = np.ones(len(m.k))
    return np.column_stack([
        c * m.w[0] * m.A_g.T @ one,
        c * (m.w[0] * m.db * m.A_g.T @ m.k - m.w[1] * m.A_v.T @ one),
        s * m.w[3] * m.A_v.T @ one,
    ])


def _first_move(H, G):
    x = np.linalg.solve(H, G)[0]
    return float(-x[0]), float(-x[1]), float(x[2])


def feedback_gains(m, phi):
    """(k_s, k_v, k_u) of the first move of MPC.plan at weight phi (ridge as in MPC.plan)."""
    c, s = math.cos(phi), math.sin(phi)
    H = c * m.H_ego + s * m.H_soc + M.RIDGE * np.eye(len(m.k))
    return _first_move(H, _rhs(m, c, s))


def feedback_gains_rho(m, rho):
    """Ridge-free gains as a function of rho = tan(phi) >= 0 (cost divided by cos(phi))."""
    return _first_move(m.H_ego + rho * m.H_soc, _rhs(m, 1.0, rho))


def gain_curve(m, phis):
    """Array of shape (len(phis), 3) with the columns k_s, k_v, k_u."""
    return np.array([feedback_gains(m, p) for p in phis], dtype=float)
