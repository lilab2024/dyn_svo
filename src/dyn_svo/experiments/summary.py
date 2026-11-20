"""FOR, eligibility, nested LOEO selection and final selection from evaluation tasks (
). Dev results are exploratory."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dyn_svo.config import load_config
from dyn_svo.metrics import evaluation as ev


def load_rows(run_dir):
    rows = []
    for p in sorted((Path(run_dir) / "tasks").glob("*.json")):
        rec = json.loads(p.read_text(encoding="utf-8"))
        c, r = rec["task"]["config"], rec["result"]
        for row in r["rows"]:
            rows.append(dict(row, config=c["name"], family=c["family"], rank=tuple(c["rank"]),
                             exp=r["exp"], regime=r["regime"], commit=rec["provenance"]["commit"],
                             dirty=rec["provenance"]["dirty"], variant=rec["task"].get("variant", "primary")))
    return pd.DataFrame(rows)


def check_variant(df, variant):
    """ every task of a run must belong to the requested analysis variant."""
    found = sorted(df["variant"].unique()) if "variant" in df.columns else ["primary"]
    if found != [variant]:
        raise ValueError("tasks belong to variants " + str(found) + ", expected " + repr(variant))


def add_log_for(df):
    """log FOR per (config, exp, av, seed), paired with the same-seed all-HDV row."""
    df = df.copy()
    df["sigma_sum"] = df["follower_sigma"].apply(lambda s: float(np.sum(s)))
    base = df[df["config"] == "hdv"].set_index(["exp", "av", "seed"])["sigma_sum"]
    df["log_for"] = np.log(df["sigma_sum"].to_numpy() / base.loc[list(zip(df["exp"], df["av"], df["seed"]))].to_numpy())
    return df


def experiment_values(df):
    """Mean over seeds per episode, then mean over AV positions: one value per (config, exp)."""
    ep = df.groupby(["config", "exp", "av"])["log_for"].mean().reset_index()
    return ep.groupby(["config", "exp"])["log_for"].mean()


def eligibility(df, config, exps, thresholds):
    sub = df[(df["config"] == config) & (df["exp"].isin(exps))]
    checks = [ev.guardrail_checks(r, thresholds[r["regime"]]) for r in sub.to_dict("records")]
    return ev.config_eligible(checks)


def select(df, values, family, train_exps, thresholds, tie_tol):
    """Best eligible configuration of a family on the training experiments (None if none)."""
    cands = []
    for c, rank in df[df["family"] == family][["config", "rank"]].drop_duplicates().itertuples(index=False):
        ok, _ = eligibility(df, c, train_exps, thresholds)
        if ok:
            cands.append((float(np.mean([values[(c, e)] for e in train_exps])), tuple(rank), c))
    if not cands:
        return None
    best = min(s for s, _, _ in cands)
    return min((r, c) for s, r, c in cands if s <= best + tie_tol)[1]


def cross_validate(df, thresholds, dev, tie_tol):
    """Nested LOEO: per family and held-out dev experiment k, select on the other four."""
    values = experiment_values(df)
    out = []
    for fam in sorted(df["family"].unique()):
        if fam == "hdv":
            continue
        for k in dev:
            train = [e for e in dev if e != k]
            c = select(df, values, fam, train, thresholds, tie_tol)
            row = {"family": fam, "held_out": k, "chosen": c}
            if c is not None:
                ok, rates = eligibility(df, c, [k], thresholds)
                row.update(log_for=float(values[(c, k)]), eligible_on_held_out=ok, **rates)
            out.append(row)
    return pd.DataFrame(out)


def final_selection(df, thresholds, dev, tie_tol):
    values = experiment_values(df)
    return {fam: select(df, values, fam, dev, thresholds, tie_tol)
            for fam in sorted(df["family"].unique()) if fam != "hdv"}


def build_summary(df, thresholds, dev):
    tie = float(load_config("eval")["tie_tol"])
    df = add_log_for(df)
    values = experiment_values(df)
    plat = df[df["family"] != "hdv"].groupby("config")[["safety_share", "clip_share"]].mean()
    return {"commits": sorted(df["commit"].unique()), "any_dirty": bool(df["dirty"].any()),
            "n_rows": int(len(df)), "variants": sorted(df["variant"].unique()) if "variant" in df.columns else ["primary"],
            "experiment_log_for": {f"{c}|{e}": float(v) for (c, e), v in values.items()},
            "cross_validation": cross_validate(df, thresholds, dev, tie).to_dict("records"),
            "final_selection": final_selection(df, thresholds, dev, tie),
            "platform_shares": plat.round(5).to_dict("index")}


# ---------------- test run ----------------

def plan_constants(path=None):
    """Decision constants of the evaluation (config/decision_constants.yaml)."""
    import yaml
    from dyn_svo.data.manifest import REPO_ROOT
    p = Path(path) if path is not None else REPO_ROOT / "config" / "decision_constants.yaml"
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def paired_t_interval(d, level=0.95):
    """Mean and paired-t interval of d (df = n - 1; indicative for n = 3, plan Section 9)."""
    from scipy import stats
    d = np.asarray(d, dtype=float)
    m = float(d.mean())
    if len(d) < 2:
        return m, None, None
    half = float(stats.t.ppf(0.5 + level / 2, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)))
    return m, m - half, m + half


def _paired(df, config, field):
    """(exp, av, seed) -> field of config, aligned with the same-seed all-HDV row."""
    key = ["exp", "av", "seed"]
    a = df[df["config"] == config].set_index(key)[field]
    b = df[df["config"] == "hdv"].set_index(key)[field]
    return a, b.loc[a.index]


def position_log_ratios(df, config, exp):
    """W1: L_j = mean over AV positions of the seed mean of log(sigma_j(config) / sigma_j(hdv)),
    j = 1..J with J the smallest number of followers over the episodes of exp."""
    sub = df[df["exp"] == exp]
    a, b = _paired(sub, config, "follower_sigma")
    J = int(min(len(x) for x in a))
    logs = pd.DataFrame([np.log(np.asarray(x[:J], float) / np.asarray(y[:J], float)) for x, y in zip(a, b)],
                        index=a.index)
    per_ep = logs.groupby(level=["exp", "av"]).mean()
    return per_ep.mean().to_numpy()


def _episode_mean(series):
    return float(series.groupby(level=["exp", "av"]).mean().mean())


def h1_rule(values, guard_ok, dyn, ref, exps, min_reduction):
    """H1 or the matched rule (W5): d = log FOR(dyn) - log FOR(ref) per experiment."""
    d = {e: float(values[(dyn, e)] - values[(ref, e)]) for e in exps}
    mean, lo, hi = paired_t_interval(list(d.values()))
    thr = float(np.log(1.0 - min_reduction))
    crit = {"all_negative": all(v < 0 for v in d.values()), "mean_at_most_threshold": mean <= thr,
            "dynamic_guardrails_all_experiments": all(guard_ok[(dyn, e)] for e in exps)}
    return {"reference": ref, "d": d, "mean_d": mean, "t95": [lo, hi], "threshold": thr, **crit,
            "supported": all(crit.values())}


def h2_rule(df, dyn, exps, speed_noninf):
    """H2 for the dynamic arm in every experiment."""
    per = {}
    for e in exps:
        sub = df[df["exp"] == e]
        L = position_log_ratios(df, dyn, e)
        slope = float(np.polyfit(np.arange(1, len(L) + 1), L, 1)[0]) if len(L) > 1 else float("nan")
        ga, gb = _paired(sub, dyn, "av_gap_std")
        sa, sb = _paired(sub, dyn, "chain_mean_speed")
        gap_dyn, gap_hdv = _episode_mean(ga), _episode_mean(gb)
        speed_ratio = _episode_mean(sa / sb)
        c1 = bool((L < 0).all() and slope < 0)
        c2 = gap_dyn > gap_hdv
        c3 = speed_ratio >= speed_noninf
        per[e] = {"position_log_ratio": [float(x) for x in L], "slope": slope, "followers_smoother": c1,
                  "av_gap_std": gap_dyn, "av_gap_std_hdv": gap_hdv, "av_absorbs": c2,
                  "chain_speed_ratio": speed_ratio, "speed_kept": c3}
    c = {k: all(per[e][k] for e in exps) for k in ("followers_smoother", "av_absorbs", "speed_kept")}
    return {"per_experiment": per, **c, "supported": all(c.values()),
            "smoother_at_efficiency_cost": c["followers_smoother"] and c["av_absorbs"] and not c["speed_kept"]}


def build_test_summary(df, thresholds, freeze, variant, constants=None):
    """Descriptive values of every configuration and, where a dynamic selection exists, H1, the
    matched rule and H2. No manual judgement."""
    constants = constants if constants is not None else plan_constants()
    v = freeze["variants"][variant]
    exps = list(freeze["test_experiments"])
    df = add_log_for(df)
    values = experiment_values(df)
    configs = [c for c in v["configs"] if c != "hdv"]
    guard, guard_ok = {}, {}
    for c in configs:
        for e in exps:
            ok, rates = eligibility(df, c, [e], thresholds)
            guard[f"{c}|{e}"] = dict(rates, passes=ok)
            guard_ok[(c, e)] = ok
    out = {"variant": variant, "test_experiments": exps, "commits": sorted(df["commit"].unique()),
           "any_dirty": bool(df["dirty"].any()), "n_rows": int(len(df)),
           "experiment_log_for": {f"{c}|{e}": float(values[(c, e)]) for c in configs for e in exps},
           "guardrails": guard, "constants": {k: constants[k] for k in ("min_for_reduction", "chain_speed_noninferiority")},
           "best_fixed": v["best_fixed"], "dynamic": v["dynamic"]}
    if v["dynamic"] is None:
        out.update(h1=None, matched_rule=None, h2=None,
                   note="no dynamic selection in this variant: H1, the matched rule and H2 are not examined ")
        return out
    dyn = v["dynamic"]
    out["h1"] = h1_rule(values, guard_ok, dyn, v["best_fixed"], exps, constants["min_for_reduction"])
    out["matched_rule"] = h1_rule(values, guard_ok, dyn, "fixed_match", exps, constants["min_for_reduction"])
    out["h2"] = h2_rule(df, dyn, exps, constants["chain_speed_noninferiority"])
    return out
