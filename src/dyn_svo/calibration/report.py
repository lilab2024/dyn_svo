"""Calibration report: provenance, delay-variant rule, boundary rule."""
import numpy as np
import pandas as pd

from dyn_svo.sim import idm

BOUND_RTOL = 1e-3   # a parameter within 0.1% of the bound range counts as "at the bound"


def per_vehicle(df):
    return df[df["veh"].astype(str) != "pooled"]


def check_provenance(df):
    return {"n_rows": int(len(df)), "commits": sorted(df["commit"].astype(str).unique()),
            "any_dirty": bool(df["dirty"].astype(bool).any()),
            "data_checks": sorted(df["data_check"].astype(str).unique())}


def delay_rule(df):
    """ variant with lower mean held-out spacing RMSE over LOEO per-vehicle fits.

    Only (fold, regime, vehicle) cells that have a held-out RMSE for both variants are used.
    A tie selects 'nodelay' (simpler model).
    """
    pv = per_vehicle(df)
    loeo = pv[pv["fold"].str.startswith("loeo_") & pv["held_rmse_m"].notna()]
    wide = loeo.pivot_table(index=["fold", "regime", "veh"], columns="variant",
                            values="held_rmse_m").dropna()
    means = {v: float(wide[v].mean()) for v in ("nodelay", "delay")}
    chosen = "nodelay" if means["nodelay"] <= means["delay"] else "delay"
    return {"n_pairs": int(len(wide)), "mean_held_rmse_m": means, "chosen": chosen,
            "share_pairs_delay_better": float((wide["delay"] < wide["nodelay"]).mean())}


def bound_shares(df, bounds):
    """Share of fits at the lower / upper bound, per IDM parameter."""
    out = {}
    for p in idm.PARAM_NAMES:
        lo, hi = bounds[p]
        tol = BOUND_RTOL * (hi - lo)
        out[p] = {"lower": float((df[p] <= lo + tol).mean()), "upper": float((df[p] >= hi - tol).mean())}
    return out


def boundary_rule(df, variant, bounds):
    """ triggered iff share of per-vehicle fits with b at its lower bound > 1/3 AND
    median(held-out RMSE / training RMSE) over LOEO per-vehicle fits > 1.5."""
    pv = per_vehicle(df)
    pv = pv[pv["variant"] == variant]
    lo, hi = bounds["b"]
    share = float((pv["b"] <= lo + BOUND_RTOL * (hi - lo)).mean())
    loeo = pv[pv["fold"].str.startswith("loeo_") & pv["held_rmse_m"].notna()]
    ratio = float((loeo["held_rmse_m"] / loeo["rmse_m"]).median())
    return {"variant": variant, "share_b_at_lower": share, "median_held_over_train": ratio,
            "triggered": bool(share > 1.0 / 3.0 and ratio > 1.5)}


def rmse_summary(df):
    """Quantiles of training and held-out RMSE per regime and variant (per-vehicle fits)."""
    rows = []
    pv = per_vehicle(df)
    for (reg, var), g in pv.groupby(["regime", "variant"]):
        h = g["held_rmse_m"].dropna()
        rows.append({"regime": reg, "variant": var, "n_fits": int(len(g)),
                     "train_median": float(g["rmse_m"].median()),
                     "train_p90": float(g["rmse_m"].quantile(0.9)),
                     "held_n": int(len(h)),
                     "held_median": float(h.median()) if len(h) else None,
                     "held_p90": float(h.quantile(0.9)) if len(h) else None})
    return rows


def build_report(df, bounds):
    choice = delay_rule(df)
    pv = per_vehicle(df)
    delay_fits = pv[pv["variant"] == "delay"]
    return {
        "provenance_of_results": check_provenance(df),
        "max_collided_share": float(df["collided_share"].max()),
        "delay_rule": choice,
        "delay_steps_counts": {int(k): int(v) for k, v in
                               delay_fits["delay_steps"].value_counts().sort_index().items()},
        "bound_shares_chosen_per_vehicle": bound_shares(pv[pv["variant"] == choice["chosen"]], bounds),
        "boundary_rule": boundary_rule(df, choice["chosen"], bounds),
        "rmse_summary": rmse_summary(df),
        "pooled": df[df["veh"].astype(str) == "pooled"][
            ["fold", "regime", "variant", "rmse_m", "held_rmse_m", "delay_steps"]
            + list(idm.PARAM_NAMES)].round(4).to_dict("records"),
        "compute_hours_sum": float(df["seconds"].sum() / 3600.0),
    }
