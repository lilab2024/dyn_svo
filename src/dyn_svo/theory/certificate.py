"""Rigorous LMI certificates for the AV loop under a time-varying weight (Theorem D).

1. Gains as partial fractions. With H_ego V = H_soc V diag(lam), V' H_soc V = I
   (all lam > 0), the ridge-free gains are
     k_s(rho) = -sum a_i / (lam_i + rho),  k_v(rho) = -sum c_i / (lam_i + rho),
     k_u(rho) = rho sum b_i / (lam_i + rho) = sum b_i - sum b_i lam_i / (lam_i + rho),
   so |k''| on [rho_j, rho_(j+1)] is bounded termwise at rho_j, and the curve lies in the chord
   expanded by drho^2 / 8 times that bound. Boxes around the grid points contain the curve; the
   corners of the boxes are the vertices for the LMIs, since (A, B) is affine in k.
2. Continuous bounded-real lemma at every corner with one P (rate independent, any measurable phi_t
   in the interval), or the Schur-complement form for the exact discrete loop. In
   discrete time the leader is ballistic within a step, d_n = (v_L(n) + v_L(n+1)) / 2; the shifted
   gap state s~_n = s_n - dt v_L(n) / 2 turns this into a 3-state loop with inputs (dv_L, dU) and a
   feedthrough D, so both certificates have the same inputs.
3. Independent numpy check of P and gamma at every corner.
4. Rate-bounded refinement with P(phi) = P0 + (phi - phi_lo) P1 on grid points only: gridded, not
   rigorous.
"""
import math
import warnings

import numpy as np
import scipy.linalg as sl

from dyn_svo.control import mpc as M
from dyn_svo.theory import gains as G
from dyn_svo.theory import linear as L

DELTA = 1e-7          # strictness margin of the LMIs given to the solver
EPS = np.finfo(float).eps


def partial_fractions(m):
    """lam, a (k_s), c (k_v), b (k_u) of the ridge-free gains of MPC instance m."""
    lam, V = sl.eigh(m.H_ego, m.H_soc)
    if lam.min() <= 0:
        raise ValueError("generalized eigenvalues must be positive")
    one = np.ones(len(m.k))
    g_s = m.w[0] * m.A_g.T @ one
    g_v = m.w[0] * m.db * m.A_g.T @ m.k - m.w[1] * m.A_v.T @ one
    g_u = m.w[3] * m.A_v.T @ one
    e = V[0]
    return {"lam": lam, "a": e * (V.T @ g_s), "c": e * (V.T @ g_v), "b": e * (V.T @ g_u)}


def gains_pf(pf, rho):
    d = 1.0 / (pf["lam"] + rho)
    return np.array([-np.sum(pf["a"] * d), -np.sum(pf["c"] * d), rho * np.sum(pf["b"] * d)])


def second_derivative_bound(pf, rho0):
    """Termwise bound of |k''(rho)| for rho >= rho0 >= 0 (each term decreases in rho)."""
    q = 2.0 / (pf["lam"] + rho0) ** 3
    return np.array([np.sum(np.abs(pf["a"]) * q), np.sum(np.abs(pf["c"]) * q),
                     np.sum(np.abs(pf["b"]) * pf["lam"] * q)])


def ridge_margin(m, pf, phis):
    """Largest difference between the gains of MPC.plan (ridge) and the ridge-free gains on the grid,
    times 10 plus 1e-12."""
    d = max(np.max(np.abs(np.array(G.feedback_gains(m, p)) - gains_pf(pf, math.tan(p)))) for p in phis)
    return 10.0 * d + 1e-12


def curve_cover(m, phi_lo, phi_hi, n_grid=400):
    """Grid points of K in rho = tan(phi), a per-point box half-width (3,) and the box corners."""
    if not 0.0 <= phi_lo <= phi_hi < math.pi / 2:
        raise ValueError("need 0 <= phi_lo <= phi_hi < pi/2 ")
    pf = partial_fractions(m)
    if phi_hi == phi_lo:
        rhos = np.array([math.tan(phi_lo)])
        half = np.zeros((1, 3))
    else:
        rhos = np.linspace(math.tan(phi_lo), math.tan(phi_hi), int(n_grid) + 1)
        dr = rhos[1] - rhos[0]
        seg = np.array([dr * dr / 8.0 * second_derivative_bound(pf, r) for r in rhos[:-1]])
        half = np.zeros((len(rhos), 3))
        half[:-1] = np.maximum(half[:-1], seg)
        half[1:] = np.maximum(half[1:], seg)
    ks = np.array([gains_pf(pf, r) for r in rhos])
    half = half + ridge_margin(m, pf, np.arctan(rhos))
    signs = np.array([1.0, -1.0])
    corners = np.array([k + half_j * np.array([sx, sy, sz])
                        for k, half_j in zip(ks, half) for sx in signs for sy in signs for sz in signs])
    lower = (ks - half).min(axis=0)
    return {"rho": rhos, "k": ks, "half": half, "corners": corners, "k_lower": lower, "pf": pf}


def discrete_shifted(k, h, tau=L.TAU, dt=L.DT):
    """(A, B, C, D) of the exact discrete loop with a ballistic leader, state (s~, dv, a_prev),
    s~ = ds - dt dv_L / 2, inputs (dv_L, dU), outputs (dv, e_s)."""
    ks, kv, ku = k
    lam = min(1.0, dt / tau)
    a_row = np.array([lam * ks, -lam * (ks * h + kv + ku), 1.0 - lam])
    a_in = np.array([lam * (kv + ks * dt / 2.0), lam * ku])
    v_row = np.array([0.0, 1.0, 0.0]) + dt * a_row
    v_in = dt * a_in
    s_row = np.array([1.0, -dt, 0.0]) - 0.5 * dt * dt * a_row
    s_in = np.array([dt, 0.0]) - 0.5 * dt * dt * a_in
    A = np.vstack([s_row, v_row, a_row])
    B = np.vstack([s_in, v_in, a_in])
    C = np.array([[0.0, 1.0, 0.0], [1.0, -h, 0.0]])
    D = np.array([[0.0, 0.0], [dt / 2.0, 0.0]])
    return A, B, C, D


def _bmat_cont(P, g2, A, B, C):
    import cvxpy as cp
    nu = B.shape[1]
    return cp.bmat([[A.T @ P + P @ A + C.T @ C, P @ B], [B.T @ P, -g2 * np.eye(nu)]])


def _num_cont(P, g2, A, B, C):
    nu = B.shape[1]
    return np.block([[A.T @ P + P @ A + C.T @ C, P @ B], [B.T @ P, -g2 * np.eye(nu)]])


def _bmat_disc(P, g2, A, B, C, D):
    import cvxpy as cp
    n, nu, ny = A.shape[0], B.shape[1], C.shape[0]
    Z = np.zeros
    return cp.bmat([[P, Z((n, nu)), A.T @ P, C.T],
                    [Z((nu, n)), g2 * np.eye(nu), B.T @ P, D.T],
                    [P @ A, P @ B, P, Z((n, ny))],
                    [C, D, Z((ny, n)), np.eye(ny)]])


def _num_disc(P, g2, A, B, C, D):
    n, nu, ny = A.shape[0], B.shape[1], C.shape[0]
    Z = np.zeros
    return np.block([[P, Z((n, nu)), A.T @ P, C.T],
                     [Z((nu, n)), g2 * np.eye(nu), B.T @ P, D.T],
                     [P @ A, P @ B, P, Z((n, ny))],
                     [C, D, Z((ny, n)), np.eye(ny)]])


def corner_systems(corners, h, discrete=False):
    out = []
    for k in corners:
        if discrete:
            out.append(discrete_shifted(tuple(k), h))
        else:
            A, B, C = L.continuous_matrices(tuple(k), h)
            out.append((A, B, C, np.zeros((C.shape[0], B.shape[1]))))
    return out


def verify(P, gamma, systems, output, discrete=False):
    """Independent numpy check: P > 0 and every corner LMI definite beyond the
    floating-point bound at the reported gamma. Returns (ok, worst slack)."""
    P = 0.5 * (P + P.T)
    ev = np.linalg.eigvalsh(P)
    worst = ev.min() - len(P) * EPS * np.abs(ev).max()
    ok = worst > 0
    g2 = gamma * gamma
    for A, B, C, D in systems:
        c, d = C[output:output + 1], D[output:output + 1]
        Mx = _num_disc(P, g2, A, B, c, d) if discrete else -_num_cont(P, g2, A, B, c)
        Mx = 0.5 * (Mx + Mx.T)
        e = np.linalg.eigvalsh(Mx)
        slack = e.min() - len(Mx) * EPS * np.abs(e).max()
        worst = min(worst, slack)
        ok = ok and slack > 0
    return bool(ok), float(worst)


INFLATIONS = (1e-6, 1e-4, 1e-3, 1e-2)


def certify(systems, output, discrete=False):
    """Common-P bounded-real certificate over the corner systems for one output (0: dv, 1: e_s).
    The solver's gamma is inflated by the first factor in INFLATIONS at which the numpy check passes;
    returns status, gamma (verified), the inflation used, P and the worst slack."""
    import cvxpy as cp
    n = systems[0][0].shape[0]
    P = cp.Variable((n, n), symmetric=True)
    g2 = cp.Variable(nonneg=True)
    cons = [P >> DELTA * np.eye(n)]
    for A, B, C, D in systems:
        c, d = C[output:output + 1], D[output:output + 1]
        if discrete:
            Mx = _bmat_disc(P, g2, A, B, c, d)
            cons.append(0.5 * (Mx + Mx.T) >> DELTA * np.eye(Mx.shape[0]))
        else:
            Mx = _bmat_cont(P, g2, A, B, c)
            cons.append(0.5 * (Mx + Mx.T) << -DELTA * np.eye(Mx.shape[0]))
    prob = cp.Problem(cp.Minimize(g2), cons)
    res = {"status": None, "gamma": None, "gamma_solver": None, "inflation": None, "P": None,
           "verified": False, "slack": None}
    try:
        with warnings.catch_warnings():   # "may be inaccurate": the status is recorded and numpy decides
            warnings.filterwarnings("ignore", message="Solution may be inaccurate")
            prob.solve(solver="CLARABEL")
    except cp.error.SolverError as e:
        res["status"] = "solver_error: " + str(e)
        return res
    res["status"] = prob.status
    if P.value is None or g2.value is None:
        return res
    g0 = math.sqrt(max(float(g2.value), 0.0))
    res["gamma_solver"], res["P"] = g0, np.array(P.value)
    for f in INFLATIONS:
        ok, slack = verify(P.value, g0 * (1.0 + f), systems, output, discrete)
        res.update({"gamma": g0 * (1.0 + f), "inflation": f, "verified": ok, "slack": slack})
        if ok:
            break
    return res


def hinf_norm_frozen(k, h, output, discrete=False, omegas=None):
    """Frequency-sweep H-infinity norm of the frozen loop with the inputs of the certificate."""
    omegas = np.geomspace(1e-4, math.pi / L.DT, 4000) if omegas is None else omegas
    if discrete:
        A, B, C, D = discrete_shifted(tuple(k), h)
        zs = np.exp(1j * omegas * L.DT)
    else:
        A, B, C = L.continuous_matrices(tuple(k), h)
        D = np.zeros((C.shape[0], B.shape[1]))
        zs = 1j * omegas
    c, d = C[output:output + 1], D[output:output + 1]
    best = 0.0
    for z in zs:
        Gm = c @ np.linalg.solve(z * np.eye(A.shape[0]) - A, B) + d
        best = max(best, float(np.linalg.svd(Gm, compute_uv=False)[0]))
    return best


def certify_rate_gridded(m, h, phi_lo, phi_hi, rate, output, n_grid=60):
    """Gridded rate-bounded refinement (not rigorous): P(phi) = P0 + (phi - phi_lo) P1."""
    import cvxpy as cp
    n = 3
    P0 = cp.Variable((n, n), symmetric=True)
    P1 = cp.Variable((n, n), symmetric=True)
    g2 = cp.Variable(nonneg=True)
    cons = []
    for phi in np.linspace(phi_lo, phi_hi, int(n_grid) + 1):
        P = P0 + (phi - phi_lo) * P1
        cons.append(P >> DELTA * np.eye(n))
        A, B, C = L.continuous_matrices(G.feedback_gains(m, phi), h)
        c = C[output:output + 1]
        for dP in (rate * P1, -rate * P1):
            Mx = _bmat_cont(P, g2, A, B, c) + cp.bmat([[dP, np.zeros((n, B.shape[1]))],
                                                         [np.zeros((B.shape[1], n)), np.zeros((B.shape[1], B.shape[1]))]])
            cons.append(0.5 * (Mx + Mx.T) << -DELTA * np.eye(n + B.shape[1]))
    prob = cp.Problem(cp.Minimize(g2), cons)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Solution may be inaccurate")
        prob.solve(solver="CLARABEL")
    gamma = math.sqrt(float(g2.value)) if g2.value is not None else None
    return {"status": prob.status, "gamma": gamma, "label": "gridded, not rigorous"}
