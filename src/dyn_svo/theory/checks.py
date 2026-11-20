"""Theory checks with the calibrated parameters. Pure functions; scripts/theory_checks.py
loads the development data and parameters, calls run_all and writes theory.json."""
import math

import numpy as np

from dyn_svo.sim import idm
from dyn_svo.theory import certificate as C
from dyn_svo.theory import gains as G
from dyn_svo.theory import linear as L

U16 = math.pi / 16
SETTINGS = {
    "n_grid": 400,
    "sign_range": [0.0, 7.5 * U16],
    "sign_grid": 2000,
    "table_phis": [i * math.pi / 160 for i in range(80)],   # 0 .. 7.9 pi/16 in steps of pi/160
    "periods_s": [20.0, 30.0, 45.0, 64.0],
    "band_s": [20.0, 64.0],
    "long_s": [64.0, 600.0],
    "intervals": [[3 * U16, 7 * U16], [4 * U16, 7.5 * U16]],
    "fixed": [j * U16 for j in range(8)],
    "rate_rad_s": 0.1,
    "rate_grid": 60,
    "n_reference": 41,                                # frozen H-infinity reference points per interval
}


def speed_stats(rows):
    """Per regime: median, p10, p90 of v and the row count."""
    out = {}
    for regime, g in rows.groupby("regime"):
        v = g["v"].to_numpy(dtype=float)
        out[str(regime)] = {"median": float(np.median(v)), "p10": float(np.percentile(v, 10)),
                            "p90": float(np.percentile(v, 90)), "n_rows": int(len(v))}
    return out


def ring_sizes(experiments):
    """experiments: iterable of (exp, regime, n_vehicles) -> {regime: {exp: n}}."""
    out = {}
    for exp, regime, n in experiments:
        out.setdefault(str(regime), {})[str(exp)] = int(n)
    return out


def pooled_idm(table, regime, fold="all_dev", variant="nodelay"):
    t = table[(table["fold"] == fold) & (table["regime"] == regime) & (table["variant"] == variant)
              & (table["veh"].astype(str) == "pooled")]
    if len(t) != 1:
        raise KeyError("need exactly one pooled row for " + fold + "/" + str(regime) + "/" + variant)
    return {k: float(t.iloc[0][k]) for k in idm.PARAM_NAMES}


def human_summary(p, speeds):
    """Linearised IDM at the median, p10 and p90 speeds."""
    out = {}
    for key in ("median", "p10", "p90"):
        f = L.human_partials(p, speeds[key])
        out[key] = {"v": speeds[key], "f": list(f), "lambda_h": L.lambda_human(f),
                    "routh": bool(f[0] > 0 and -(f[1] + f[2]) > 0)}
    return out


def _band_peak(k, h, lo, hi, n=200):
    w = 2 * math.pi / np.geomspace(lo, hi, n)
    g = np.abs(L.gain_av(k, h, w))
    i = int(np.argmax(g))
    return float(g[i]), float(2 * math.pi / w[i])


def gain_table(m, h, lambda_h, ns, s=SETTINGS):
    """Rows per phi: gains, Routh-Hurwitz, Lambda_AV, head-to-tail Lambda per n, |G| at periods,
    band and long-period peaks (continuous model, exact moving average)."""
    rows = []
    for phi in s["table_phis"]:
        k = G.feedback_gains(m, phi)
        lam = L.lambda_av(k, h) if k[0] > 0 else None
        rows.append({
            "phi": phi, "phi_over_pi16": phi / U16, "k": list(k),
            "routh": L.routh_hurwitz_stable(k, h),
            "lambda_av": lam,
            "head_to_tail": {str(n): (lam + (n - 2) * lambda_h if lam is not None else None) for n in ns},
            "gain_at": {str(T): float(abs(L.gain_av(k, h, 2 * math.pi / T))) for T in s["periods_s"]},
            "band_peak": _band_peak(k, h, *s["band_s"]),
            "long_peak": _band_peak(k, h, *s["long_s"]),
        })
    return rows


def sign_check(m, s=SETTINGS):
    """Rigorous signs on the sign range: k_s > 0 and k_v > 0 from the lower
    bound of the curve cover; k_u >= 0 (zero only at phi = 0) from k_u = rho q(rho),
    q(rho) = sum b_i / (lam_i + rho) > 0, bounded with the same chord and termwise
    second-derivative argument (q'' bounded by sum 2 |b_i| / (lam_i + rho_j)^3 on each segment)."""
    cov = C.curve_cover(m, s["sign_range"][0], s["sign_range"][1], s["sign_grid"])
    pf, rho = cov["pf"], cov["rho"]
    q = np.array([np.sum(pf["b"] / (pf["lam"] + r)) for r in rho])
    dr = rho[1] - rho[0]
    q_lower = min(min(q[j], q[j + 1]) - dr * dr / 8.0 * np.sum(2.0 * np.abs(pf["b"]) / (pf["lam"] + rho[j]) ** 3)
                  for j in range(len(rho) - 1))
    lower = list(cov["k_lower"])
    return {"range": s["sign_range"], "n_grid": s["sign_grid"], "k_s_lower": lower[0], "k_v_lower": lower[1],
            "q_lower": float(q_lower), "max_half_width": list(cov["half"].max(axis=0)),
            "positive": bool(lower[0] > 0 and lower[1] > 0 and q_lower > 0)}


def _certify_all(m, h, lo, hi, n_grid, n_ref):
    cov = C.curve_cover(m, lo, hi, n_grid)
    phis = np.linspace(lo, hi, n_ref) if hi > lo else np.array([lo])
    out = {"phi": [lo, hi], "n_corners": int(len(cov["corners"])),
           "max_half_width": list(cov["half"].max(axis=0)), "k_lower": list(cov["k_lower"])}
    for disc in (False, True):
        systems = C.corner_systems(cov["corners"], h, disc)
        for o, name in ((0, "dv"), (1, "e_s")):
            r = C.certify(systems, o, disc)
            ref = max(C.hinf_norm_frozen(G.feedback_gains(m, p), h, o, disc) for p in phis)
            out[("discrete_" if disc else "continuous_") + name] = {
                "status": r["status"], "gamma": r["gamma"], "gamma_solver": r["gamma_solver"],
                "inflation": r["inflation"], "verified": r["verified"], "slack": r["slack"],
                "P": None if r["P"] is None else np.asarray(r["P"]).tolist(), "frozen_reference": ref}
    return out


def certificates(m, h, s=SETTINGS):
    out = {"intervals": [], "fixed": []}
    for lo, hi in s["intervals"]:
        c = _certify_all(m, h, lo, hi, s["n_grid"], s["n_reference"])
        for o, name in ((0, "dv"), (1, "e_s")):
            c["rate_gridded_" + name] = C.certify_rate_gridded(m, h, lo, hi, s["rate_rad_s"], o, s["rate_grid"])
        out["intervals"].append(c)
    for phi in s["fixed"]:
        out["fixed"].append(_certify_all(m, h, phi, phi, 1, 1))
    return out


def run_all(policy, idm_by_regime, speeds, rings, s=SETTINGS, log=print):
    """Everything of per regime. policy: {regime: MPC policy}; idm_by_regime: {regime: pooled
    IDM}; speeds: speed_stats; rings: ring_sizes."""
    out = {}
    for regime in sorted(policy):
        log("regime " + regime)
        m = G.mpc_from_policy(policy[regime])
        h = float(policy[regime]["h"])
        hum = human_summary(idm_by_regime[regime], speeds[regime])
        ns = sorted(set(rings[regime].values()))
        out[regime] = {
            "idm_pooled": idm_by_regime[regime], "speeds": speeds[regime], "human": hum,
            "ring_sizes": rings[regime],
            "partial_fraction_lambda": list(C.partial_fractions(m)["lam"]),
            "gain_table": gain_table(m, h, hum["median"]["lambda_h"], ns, s),
            "signs": sign_check(m, s),
        }
        log("  certificates")
        out[regime]["certificates"] = certificates(m, h, s)
    return out


# ---------------- human-model sensitivity and gain curves (theory_v2) ----------------

SENSITIVITY = {
    "b_grid": [0.1, 0.2, 0.3, 0.5, 1.0],                       # plus the fitted b of each regime
    "crossing_phis": [i * math.pi / 160 for i in range(80)],   # 0 .. 7.9 pi/16, as the gain table of theory_v1
    "bound_tol": 1e-3,                                         # at a bound: within 1e-3 of the bound range
    "curve_periods": [10.0, 600.0, 200],                       # log-spaced periods in s
    "curve_phis": [j * U16 for j in range(8)] + [7.5 * U16],
}


def loeo_fits(table, regime, variant="nodelay"):
    """Pooled fits of the LOEO folds whose held-out experiment lies in the regime."""
    t = table[(table["regime"] == regime) & (table["variant"] == variant) & (table["veh"].astype(str) == "pooled")
              & table["fold"].astype(str).str.startswith("loeo_") & table["held_out"].notna()]
    out = {}
    for _, r in t.sort_values("fold").iterrows():
        if str(r["fold"]) in out:
            raise KeyError("more than one pooled row for " + str(r["fold"]) + "/" + str(regime))
        out[str(r["fold"])] = {k: float(r[k]) for k in idm.PARAM_NAMES}
    return out


def at_bounds(p, bounds, tol=SENSITIVITY["bound_tol"]):
    """'lower' / 'upper' / None per parameter: within tol times the bound range of a bound."""
    out = {}
    for k in idm.PARAM_NAMES:
        lo, hi = (float(x) for x in bounds[k])
        d = tol * (hi - lo)
        out[k] = "lower" if p[k] - lo <= d else ("upper" if hi - p[k] <= d else None)
    return out


def head_to_tail_crossings(m, h, lambda_h, n, phis=SENSITIVITY["crossing_phis"]):
    """phi values at which Lambda_AV(phi) + (n - 2) Lambda_h changes sign on the grid, refined by bisection."""
    from scipy.optimize import brentq

    def f(phi):
        return L.lambda_av(G.feedback_gains(m, phi), h) + (n - 2) * lambda_h
    vals = [f(p) for p in phis]
    out = []
    for j in range(len(phis) - 1):
        if vals[j] == 0.0:
            out.append(float(phis[j]))
        elif vals[j] * vals[j + 1] < 0:
            out.append(float(brentq(f, phis[j], phis[j + 1], xtol=1e-12)))
    return {"crossings": out, "value_first": float(vals[0]), "value_last": float(vals[-1]),
            "phi_range": [float(phis[0]), float(phis[-1])]}


def critical_b(p, v, lo, hi):
    """b in [lo, hi] at which Lambda_h at speed v is zero (None if the sign does not change)."""
    from scipy.optimize import brentq

    def f(b):
        return L.lambda_human(L.human_partials(dict(p, b=b), v))
    a, z = f(lo), f(hi)
    if a * z > 0:
        return {"b": None, "lambda_h_lo": float(a), "lambda_h_hi": float(z), "range": [lo, hi]}
    return {"b": float(brentq(f, lo, hi, xtol=1e-12)), "lambda_h_lo": float(a), "lambda_h_hi": float(z),
            "range": [lo, hi]}


def sensitivity_case(m, h, p, speeds, ns, bounds, s=SENSITIVITY):
    """One parameter set: bounds, linearised humans at three speeds, head-to-tail crossings at the median."""
    hum = human_summary(p, speeds)
    lam = hum["median"]["lambda_h"]
    return {"params": p, "at_bound": at_bounds(p, bounds, s["bound_tol"]), "human": hum,
            "head_to_tail": {str(n): head_to_tail_crossings(m, h, lam, n, s["crossing_phis"]) for n in ns}}


def gain_curves(m, h, s=SENSITIVITY):
    lo, hi, n = s["curve_periods"]
    periods = np.geomspace(lo, hi, int(n))
    w = 2 * math.pi / periods
    return {"periods_s": [float(x) for x in periods],
            "curves": [{"phi": float(phi), "phi_over_pi16": float(phi / U16),
                        "gain": [float(x) for x in np.abs(L.gain_av(G.feedback_gains(m, phi), h, w))]}
                       for phi in s["curve_phis"]]}


def run_sensitivity(policy, table, speeds, rings, bounds, s=SENSITIVITY, log=print):
    """theory_v2: per regime the all_dev pooled fit, the LOEO fits, the b grid, the critical b
    the gain curves. policy: {regime: MPC policy}; speeds: speed_stats; rings: ring_sizes; bounds: calibration."""
    out = {}
    for regime in sorted(policy):
        log("regime " + regime)
        m = G.mpc_from_policy(policy[regime])
        h = float(policy[regime]["h"])
        ns = sorted(set(rings[regime].values()))
        base = pooled_idm(table, regime)
        bvals = sorted(set([float(b) for b in s["b_grid"]] + [base["b"]]))
        out[regime] = {
            "speeds": speeds[regime], "ring_sizes": rings[regime],
            "all_dev": sensitivity_case(m, h, base, speeds[regime], ns, bounds, s),
            "loeo": {f: sensitivity_case(m, h, p, speeds[regime], ns, bounds, s)
                     for f, p in loeo_fits(table, regime).items()},
            "b_grid": [dict(sensitivity_case(m, h, dict(base, b=b), speeds[regime], ns, bounds, s), b=b,
                            fitted=(b == base["b"])) for b in bvals],
            "critical_b_median": critical_b(base, speeds[regime]["median"], *[float(x) for x in bounds["b"]]),
            "gain_curves": gain_curves(m, h, s),
        }
    return out
