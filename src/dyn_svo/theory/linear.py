"""Linear closed loops of the MPC-driven AV and the IDM humans.

AV (exactly linear while the platform bounds, jerk limit and safety filter are inactive and v > 0),
with the gains (k_s, k_v, k_u) of theory.gains and deviations from an equilibrium at speed v*:
  a_cmd = k_s (ds - h dv) + k_v (dv_L - dv) - k_u (dv - dU)
Continuous model: ds' = dv_L - dv, dv' = a, tau a' = -a + a_cmd; state (ds, dv, a), inputs
(dv_L, dU), outputs (dv, e_s = ds - h dv).
Exact discrete model of the implementation (platform Euler lag lam = min(1, dt / tau), ballistic
step): state (ds_n, dv_n, a_(n-1)), inputs (dv_L_n, dU_n, dd_n) with d_n = (x_L(n+1) - x_L(n)) / dt,
  a_n = (1 - lam) a_(n-1) + lam a_cmd_n,  v_(n+1) = v_n + dt a_n,
  s_(n+1) = s_n + dt d_n - dt v_n - dt^2 a_n / 2.
U is the causal mean of the leader speed over T = 38 s (control.estimator), M(l) = (1 - exp(-l T)) / (l T).
"""
import math

import numpy as np

from dyn_svo.config import load_config
from dyn_svo.control.cooperation import idm_partials
from dyn_svo.sim import idm

TAU = float(load_config("av")["actuation_lag_s"])
DT = idm.DT
T_MA = float(load_config("baselines")["mean_speed_window_s"])


def continuous_matrices(k, h, tau=TAU):
    """A (3x3), B (3x2: dv_L, dU), C (2x3: dv, e_s) of the continuous AV loop."""
    ks, kv, ku = k
    A = np.array([[0.0, -1.0, 0.0],
                  [0.0, 0.0, 1.0],
                  [ks / tau, -(ks * h + kv + ku) / tau, -1.0 / tau]])
    B = np.array([[1.0, 0.0], [0.0, 0.0], [kv / tau, ku / tau]])
    C = np.array([[0.0, 1.0, 0.0], [1.0, -h, 0.0]])
    return A, B, C


def discrete_matrices(k, h, tau=TAU, dt=DT):
    """Ad (3x3), Bd (3x3: dv_L, dU, dd), Cd (2x3: dv, e_s) of the exact discrete AV loop."""
    ks, kv, ku = k
    lam = min(1.0, dt / tau)
    a_row = np.array([lam * ks, -lam * (ks * h + kv + ku), 1.0 - lam])
    a_in = np.array([lam * kv, lam * ku, 0.0])
    v_row = np.array([0.0, 1.0, 0.0]) + dt * a_row
    v_in = dt * a_in
    s_row = np.array([1.0, -dt, 0.0]) - 0.5 * dt * dt * a_row
    s_in = np.array([0.0, 0.0, dt]) - 0.5 * dt * dt * a_in
    Ad = np.vstack([s_row, v_row, a_row])
    Bd = np.vstack([s_in, v_in, a_in])
    Cd = np.array([[0.0, 1.0, 0.0], [1.0, -h, 0.0]])
    return Ad, Bd, Cd


def characteristic_coefficients(k, h, tau=TAU):
    """(tau, 1, k_s h + k_v + k_u, k_s): tau l^3 + l^2 + c1 l + c0."""
    ks, kv, ku = k
    return tau, 1.0, ks * h + kv + ku, ks


def routh_hurwitz_stable(k, h, tau=TAU):
    """Hurwitz test of the cubic: all coefficients positive and c2 c1 > c3 c0."""
    c3, c2, c1, c0 = characteristic_coefficients(k, h, tau)
    return bool(c3 > 0 and c2 > 0 and c1 > 0 and c0 > 0 and c2 * c1 > c3 * c0)


def moving_average_continuous(lam, T=T_MA):
    lam = np.asarray(lam, dtype=complex)
    return (1.0 - np.exp(-lam * T)) / (lam * T)


def gain_av(k, h, omega, tau=TAU, T=T_MA):
    """Complex G(j omega) from leader speed to AV speed (continuous, exact moving average)."""
    ks, kv, ku = k
    lam = 1j * np.asarray(omega, dtype=float)
    num = ks + kv * lam + ku * lam * moving_average_continuous(lam, T)
    den = tau * lam ** 3 + lam ** 2 + (ks * h + kv + ku) * lam + ks
    return num / den


def gain_av_discrete(k, h, omega, tau=TAU, dt=DT, T=T_MA):
    """Complex G(e^(j omega dt)) of the exact discrete loop; the leader is ballistic within a step,
    so d = (1 + z) v_L / 2, and U is the mean of the last round(T / dt) leader speeds."""
    Ad, Bd, Cd = discrete_matrices(k, h, tau, dt)
    n_ma = max(1, int(round(T / dt)))
    out = []
    for w in np.atleast_1d(np.asarray(omega, dtype=float)):
        z = np.exp(1j * w * dt)
        ma = np.mean(z ** (-np.arange(n_ma)))
        win = np.array([1.0, ma, 0.5 * (1.0 + z)])
        x = np.linalg.solve(z * np.eye(3) - Ad, Bd @ win)
        out.append(Cd[0] @ x)
    out = np.array(out)
    return out if np.ndim(omega) else out[0]


def lambda_av(k, h, T=T_MA):
    """Long-wave coefficient: |G|^2 = 1 - Lambda omega^2 + O(omega^4) (closed form)."""
    ks, kv, ku = k
    return h * h + 2.0 * h * (kv + ku) / ks - (2.0 + ku * T) / ks


def human_partials(p, v):
    """(f_s, f_v, f_dv) of the IDM at its equilibrium gap for speed v (dv = v - v_leader = 0)."""
    s = float(idm.equilibrium_gap(v, p["v0"], p["T"], p["s0"], p["a_max"], p["b"]))
    f_s, f_v, f_dv = idm_partials(s, v, 0.0, p)
    return float(f_s), float(f_v), float(f_dv)


def gain_human(f, omega):
    """Complex G(j omega) from leader speed to follower speed of the linearised IDM."""
    f_s, f_v, f_dv = f
    lam = 1j * np.asarray(omega, dtype=float)
    return (f_s - lam * f_dv) / (lam * lam - lam * (f_v + f_dv) + f_s)


def lambda_human(f):
    """Long-wave coefficient of the linearised IDM: (f_v^2 + 2 f_v f_dv - 2 f_s) / f_s^2."""
    f_s, f_v, f_dv = f
    return (f_v * f_v + 2.0 * f_v * f_dv - 2.0 * f_s) / (f_s * f_s)
