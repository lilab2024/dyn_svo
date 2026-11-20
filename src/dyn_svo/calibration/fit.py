"""IDM calibration by closed-loop segment simulation and differential evolution."""
import time

import numpy as np
from scipy.optimize import differential_evolution, minimize

from dyn_svo.config import load_config
from dyn_svo.sim import idm


def _settings(cfg=None):
    return cfg if cfg is not None else load_config("calibration")


def simulate_segments(seg, params, delay_steps=0):
    """Simulate every segment from its real initial state with the real leader replayed."""
    return idm.simulate_follower(seg["x_l"], seg["v_l"], seg["len_l"], seg["x"][:, 0],
                                 seg["v"][:, 0], params, delay_steps=delay_steps)


def spacing_rmse(seg, params, delay_steps=0):
    """Pooled spacing RMSE (m) over all segments and steps, and share of collided segments."""
    _, _, _, gap, collided = simulate_segments(seg, params, delay_steps)
    rmse = float(np.sqrt(np.mean((gap - seg["gap"]) ** 2)))
    return rmse, float(collided.mean())


def objective(theta, seg, delay_steps, penalty):
    params = dict(zip(idm.PARAM_NAMES, theta[:5]))
    rmse, col_share = spacing_rmse(seg, params, delay_steps)
    return rmse + penalty * col_share


def fit_idm(seg, with_delay, cfg=None):
    """Calibrate IDM on stacked segments. Returns params, delay_steps, rmse, diagnostics."""
    c = _settings(cfg)
    bounds = [tuple(c["bounds"][k]) for k in idm.PARAM_NAMES]
    penalty = float(c["collision_penalty_m"])
    de = c["de"]
    t0 = time.time()
    if with_delay:
        bounds_de = bounds + [(0, int(c["delay_steps_max"]))]
        integrality = [False] * 5 + [True]

        def f_de(th):
            return objective(th, seg, int(round(th[5])), penalty)
    else:
        bounds_de = bounds
        integrality = None

        def f_de(th):
            return objective(th, seg, 0, penalty)

    res = differential_evolution(f_de, bounds_de, popsize=int(de["popsize"]),
                                 maxiter=int(de["maxiter"]), tol=float(de["tol"]),
                                 seed=int(de["seed"]), polish=False, integrality=integrality,
                                 updating="immediate", workers=1)
    delay = int(round(res.x[5])) if with_delay else 0
    pol = minimize(lambda th: objective(th, seg, delay, penalty), res.x[:5], method="L-BFGS-B",
                   bounds=bounds)
    theta = pol.x if pol.fun <= res.fun else res.x[:5]
    params = dict(zip(idm.PARAM_NAMES, (float(v) for v in theta)))
    rmse, col_share = spacing_rmse(seg, params, delay)
    return {"params": params, "delay_steps": delay, "rmse_m": rmse, "collided_share": col_share,
            "de_fun": float(res.fun), "polish_fun": float(pol.fun), "nfev": int(res.nfev + pol.nfev),
            "n_segments": int(seg["x_l"].shape[0]), "seconds": round(time.time() - t0, 2)}
