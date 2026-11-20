"""Human acceleration-noise level from calibration residuals.

Residual = observed acceleration - IDM acceleration at the observed state (same rows and step
as the calibration). The noise sd of fold f and regime r is the std of all residuals of the
training experiments of f in regime r, with the fold-f parameters (pooled fallback).
"""
import numpy as np
import pandas as pd

from dyn_svo.calibration import params as cp
from dyn_svo.calibration import segments
from dyn_svo.sim import idm


def accel_residuals(pairs, params_by_veh, step_frames=segments.STEP_FRAMES):
    """Residuals of every follower row (every step_frames frames). params_by_veh: {veh: dict}."""
    d = pairs[pairs["k"] % step_frames == 0]
    out = []
    for veh, g in d.groupby("veh_id"):
        p = params_by_veh[int(veh)]
        a_idm = idm.idm_accel(g["v"].to_numpy(), g["gap"].to_numpy(),
                              (g["v"] - g["v_l"]).to_numpy(), **p)
        out.append(g["a"].to_numpy() - a_idm)
    return np.concatenate(out) if out else np.array([])


def noise_table(fold_list, regimes_of, pairs_of, table):
    """Rows (fold, regime, noise_sd, n) for every fold and regime with training data.

    fold_list: runner.folds(); regimes_of: {exp: regime}; pairs_of: {exp: follower pairs}.
    """
    rows = []
    for fold, train, _ in fold_list:
        for regime in ("I", "II"):
            res = []
            for e in [e for e in train if regimes_of[e] == regime]:
                pairs = pairs_of[e]
                vehs = sorted(int(v) for v in pairs["veh_id"].unique())
                pv = cp.vehicle_params(table, fold, regime, vehs)
                res.append(accel_residuals(pairs, {v: pv[v][0] for v in vehs}))
            if res:
                r = np.concatenate(res)
                rows.append({"fold": fold, "regime": regime, "noise_sd": float(np.std(r)),
                             "residual_mean": float(np.mean(r)), "n": int(len(r))})
    return pd.DataFrame(rows)


def load_noise_table(run_name=cp.DEFAULT_RUN, root=None):
    from dyn_svo.data import manifest as mf
    root = mf.data_root() if root is None else root
    return pd.read_csv(root / "dyn_svo" / "results" / "calibration" / run_name / "noise_sd.csv")


def noise_sd(table, fold, regime):
    row = table[(table["fold"] == fold) & (table["regime"] == regime)]
    if len(row) != 1:
        raise KeyError("No noise sd for " + fold + "/" + regime)
    return float(row.iloc[0]["noise_sd"])
