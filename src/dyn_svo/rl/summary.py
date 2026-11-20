"""RL-MLP development summary.

Reads the task files of one full-mode RL evaluation run (stage train: 500 out-of-fold policy tasks plus the
all-HDV tasks of the same run), checks that the run is complete and uniform, and reuses experiments.summary
for FOR, eligibility, nested LOEO selection and final selection with the families rl_fixed and rl_dynamic.
Episode value = mean over 10 training x 10 noise seeds; guardrail rates over the same runs. Additional items
that decide nothing: SD across training seeds, share of training seeds with d < 0,
dev d of the nested LOEO, phi_match of the final dynamic selection. eligibility_report: guardrail counts
per configuration, also split by the G4 result of each policy (descriptive). Dev results are exploratory.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dyn_svo.config import load_config
from dyn_svo.experiments import summary as S
from dyn_svo.metrics import evaluation as ev
from dyn_svo.rl import runner as R

FIXED, DYNAMIC, HDV = "rl_fixed", "rl_dynamic", "hdv"


def config_meta(rl=None):
    """Family and tie rank of every stage-train configuration, from config/rl.yaml: fixed rank =
    [phi in pi/16] (smaller phi wins ties), dynamic rank = [window, phi_hi - phi_lo]."""
    rl = rl or load_config("rl")
    meta = {HDV: {"family": HDV, "rank": [0.0]}}
    for p in rl["fixed_phi_pi16"]:
        meta[R.fixed_name(p)] = {"family": FIXED, "rank": [float(p)]}
    for c in rl["dynamic_configs"]:
        meta[R.dyn_name(c)] = {"family": DYNAMIC,
                               "rank": [float(c["window_s"]), float(c["phi_hi_pi16"]) - float(c["phi_lo_pi16"])]}
    return meta


def load_records(run_dir):
    out = []
    for p in sorted((Path(run_dir) / "tasks").glob("*.json")):
        out.append(json.loads(p.read_text(encoding="utf-8")))
    return out


def run_problems(records, tasks, variant, seeds):
    """Every reason why the run cannot be summarised (empty list = complete and uniform)."""
    probs = []
    want = {t["task_id"] for t in tasks}
    have = [r["task"]["task_id"] for r in records]
    missing, extra = sorted(want - set(have)), sorted(set(have) - want)
    if missing:
        probs.append("missing tasks: %d, e.g. %s" % (len(missing), missing[0]))
    if extra:
        probs.append("unexpected tasks: %d, e.g. %s" % (len(extra), extra[0]))
    if len(have) != len(set(have)):
        probs.append("duplicate task ids")
    if not records:
        return probs + ["no task files"]
    prov = [r["provenance"] for r in records]
    commits = sorted({p["commit"] for p in prov})
    if len(commits) != 1:
        probs.append("commits: " + ", ".join(c[:7] for c in commits))
    if any(p["dirty"] for p in prov):
        probs.append("dirty provenance in %d tasks" % sum(bool(p["dirty"]) for p in prov))
    cfg_md5 = {json.dumps(p["config_md5"], sort_keys=True) for p in prov}
    if len(cfg_md5) != 1:
        probs.append("config md5 differ between tasks (%d sets)" % len(cfg_md5))
    for key, want_val in (("variant", variant), ("mode", "full")):
        vals = sorted({str(r["task"].get(key)) for r in records} | {str(r["result"].get(key)) for r in records})
        if vals != [want_val]:
            probs.append("%s: %s, expected %s" % (key, vals, want_val))
    train_runs = sorted({str(p["config_md5"].get("train_run")) for p in prov})
    if len(train_runs) != 1:
        probs.append("train runs: " + ", ".join(train_runs))
    hdv = {r["result"]["exp"]: r for r in records if r["task"]["config"] == HDV}
    for r in records:
        t, res = r["task"], r["result"]
        if res["exp"] != t["exp"]:
            probs.append(t["task_id"] + ": result experiment " + str(res["exp"]))
            continue
        keys = [(int(x["av"]), int(x["seed"])) for x in res["rows"]]
        if len(keys) != len(set(keys)):
            probs.append(t["task_id"] + ": duplicate (av, seed) rows")
        if sorted({s for _, s in keys}) != list(range(seeds)):
            probs.append(t["task_id"] + ": noise seeds are not 0-%d" % (seeds - 1))
        if t["config"] != HDV:
            if any(x.get("train_seed") != t["train_seed"] for x in res["rows"]):
                probs.append(t["task_id"] + ": rows with another training seed")
            h = hdv.get(t["exp"])
            if h is not None and sorted(keys) != sorted((int(x["av"]), int(x["seed"])) for x in h["result"]["rows"]):
                probs.append(t["task_id"] + ": (av, seed) rows differ from the all-HDV task of " + t["exp"])
    return probs


def load_rows(records, meta=None):
    """Rows in the layout of experiments.summary.load_rows; RL configurations are named strings."""
    meta = meta or config_meta()
    rows = []
    for rec in records:
        t, r = rec["task"], rec["result"]
        if t["config"] not in meta:
            raise ValueError("configuration not in config/rl.yaml (stage train): " + t["config"])
        m = meta[t["config"]]
        for row in r["rows"]:
            rows.append(dict(row, config=t["config"], family=m["family"], rank=tuple(m["rank"]), exp=r["exp"],
                             regime=r["regime"], commit=rec["provenance"]["commit"], dirty=rec["provenance"]["dirty"],
                             variant=t["variant"], train_seed=row.get("train_seed")))
    return pd.DataFrame(rows)


def seed_values(df):
    """Experiment log FOR per (config, exp, training seed): mean over noise seeds per AV position, then over
    positions (df must carry log_for; all-HDV rows excluded)."""
    sub = df[df["family"] != HDV]
    ep = sub.groupby(["config", "exp", "train_seed", "av"])["log_for"].mean().reset_index()
    return ep.groupby(["config", "exp", "train_seed"])["log_for"].mean()


def seed_spread(sv, dev):
    """SD (ddof 1) across training seeds, per (config, exp) and of the mean over the dev experiments."""
    per, dev_mean = {}, {}
    for (c, e), g in sv.groupby(level=["config", "exp"]):
        per[c + "|" + e] = {"sd": float(np.std(g.to_numpy(), ddof=1)), "n_seeds": int(len(g))}
    for c, g in sv.groupby(level="config"):
        m = g.groupby(level="train_seed").mean()
        n_exp = g.groupby(level="train_seed").size()
        if (n_exp == len(dev)).all():
            dev_mean[c] = {"sd": float(np.std(m.to_numpy(), ddof=1)), "n_seeds": int(len(m))}
    return per, dev_mean


def seed_share(sv, dyn, fix, dev):
    """ (a): per training seed s, d_s = mean over dev experiments of E(dyn, e, s) - E(fix, e, s); the share
    of seeds with d_s < 0, and per experiment the number of seeds with d < 0. Seed s of two configurations shares
    the training windows and the training noise stream."""
    a, b = sv.loc[dyn], sv.loc[fix]
    d = (a - b).dropna()
    seeds = sorted({s for _, s in d.index})
    d_s = {int(s): float(np.mean([d[(e, s)] for e in dev])) for s in seeds}
    per_exp = {e: [int(sum(d[(e, s)] < 0 for s in seeds)), len(seeds)] for e in dev}
    return {"dynamic": dyn, "fixed": fix, "d_by_seed": d_s, "n_negative": int(sum(v < 0 for v in d_s.values())),
            "n_seeds": len(d_s), "share_negative": sum(v < 0 for v in d_s.values()) / max(1, len(d_s)),
            "per_experiment_negative": per_exp}


def cv_d(cv):
    """Nested LOEO dev estimate (as): per held-out experiment, log FOR of the dynamic configuration
    chosen in that fold minus that of the fixed one."""
    rows = {r["held_out"]: {} for r in cv}
    for r in cv:
        rows[r["held_out"]][r["family"]] = r
    per, reasons = {}, []
    for k, fam in rows.items():
        f, dy = fam.get(FIXED), fam.get(DYNAMIC)
        if not f or not dy or f.get("chosen") is None or dy.get("chosen") is None:
            reasons.append(k + ": no selection in " + ", ".join(x for x in (FIXED, DYNAMIC) if not fam.get(x) or fam[x].get("chosen") is None))
            continue
        per[k] = {"dynamic": dy["chosen"], "fixed": f["chosen"], "d": float(dy["log_for"] - f["log_for"]),
                  "dynamic_eligible_on_held_out": bool(dy["eligible_on_held_out"]),
                  "fixed_eligible_on_held_out": bool(f["eligible_on_held_out"])}
    if reasons:
        return {"per_fold": per, "mean_d": None, "n_negative": None, "not_computed": reasons}
    d = [v["d"] for v in per.values()]
    return {"per_fold": per, "mean_d": float(np.mean(d)), "n_negative": int(sum(x < 0 for x in d)), "n_folds": len(d)}


def phi_match(df, dyn):
    """ mean phi_mean over every out-of-fold run of the final dynamic selection, 4 decimals."""
    vals = df.loc[df["config"] == dyn, "phi_mean"].astype(float)
    if vals.empty or vals.isna().any():
        raise ValueError("no complete phi_mean values for " + dyn)
    return {"config": dyn, "phi_match": round(float(vals.mean()), 4), "phi_match_pi16": float(vals.mean()) / (np.pi / 16),
            "n_runs": int(len(vals))}


def build_rl_summary(df, thresholds, dev):
    out = S.build_summary(df, thresholds, dev)
    dfl = S.add_log_for(df)
    values = S.experiment_values(dfl)
    sv = seed_values(dfl)
    per, dev_mean = seed_spread(sv, dev)
    fs = out["final_selection"]
    dyn, fix = fs.get(DYNAMIC), fs.get(FIXED)
    out["n_runs"] = {c + "|" + e: int(n) for (c, e), n in df.groupby(["config", "exp"]).size().items()}
    out["seed_sd"] = per
    out["seed_sd_dev_mean"] = dev_mean
    out["cv_d"] = cv_d(out["cross_validation"])
    if dyn is not None and fix is not None:
        d = {e: float(values[(dyn, e)] - values[(fix, e)]) for e in dev}
        out["final_d"] = {"dynamic": dyn, "fixed": fix, "per_experiment": d, "mean_d": float(np.mean(list(d.values()))),
                          "n_negative": int(sum(v < 0 for v in d.values()))}
        out["seed_share"] = seed_share(sv, dyn, fix, dev)
    else:
        why = "no final selection in " + ", ".join(f for f in (FIXED, DYNAMIC) if fs.get(f) is None)
        out["final_d"] = {"not_computed": why}
        out["seed_share"] = {"not_computed": why}
    out["phi_match"] = phi_match(df, dyn) if dyn is not None else {"not_computed": "no final dynamic selection "}
    return out


def g4_problems(g4_records, eval_tasks, train_run):
    """ the G4 run must cover every RL task of the evaluation, from one clean commit, stage train."""
    probs = []
    want = {(t["config"], t["exp"], int(t["train_seed"])) for t in eval_tasks if t["config"] != HDV}
    have = [(r["result"]["config"], r["result"]["exp"], int(r["result"]["seed"])) for r in g4_records]
    missing = sorted(want - set(have))
    if missing:
        probs.append("G4 run lacks %d policies, e.g. %s" % (len(missing), missing[0]))
    if len(have) != len(set(have)):
        probs.append("G4 run has duplicate policies")
    if not g4_records:
        return probs + ["G4 run has no task files"]
    prov = [r["provenance"] for r in g4_records]
    if len({p["commit"] for p in prov}) != 1:
        probs.append("G4 run commits: " + ", ".join(sorted({p["commit"][:7] for p in prov})))
    if any(p["dirty"] for p in prov):
        probs.append("G4 run has dirty provenance")
    for key, val in (("stage", "train"), ("train_run", train_run)):
        got = sorted({str(p["config_md5"].get(key)) for p in prov})
        if got != [val]:
            probs.append("G4 run %s: %s, expected %s" % (key, got, val))
    return probs


def _rates(sub):
    return {"n_runs": int(len(sub)), "collisions": int((~sub["g1a"]).sum()),
            **{"fail_rate_" + g: float(1.0 - sub[g].mean()) if len(sub) else None for g in ("g1b", "g2", "g3")}}


def eligibility_report(df, thresholds, dev, g4_records):
    """ (descriptive, decides nothing): per RL configuration the collisions and G1b, G2, G3 failure rates
    on all dev experiments and per experiment, with the eligibility rule of applied to exactly those sets;
    and the same counts split by whether the policy passed G4. The G4 groups carry counts and rates
    only: no eligibility, selection, FOR or d is computed on a subset."""
    g4 = {(r["result"]["config"], r["result"]["exp"], int(r["result"]["seed"])): bool(r["result"]["g4"]["pass"]) for r in g4_records}
    sub = df[df["family"] != HDV]
    recs = sub.to_dict("records")
    chk = pd.DataFrame([ev.guardrail_checks(r, thresholds[r["regime"]]) for r in recs])
    chk["config"], chk["exp"] = sub["config"].to_numpy(), sub["exp"].to_numpy()
    chk["train_seed"] = [int(r["train_seed"]) for r in recs]
    chk["g4_pass"] = [g4[(r["config"], r["exp"], int(r["train_seed"]))] for r in recs]
    out = {}
    for c, g in chk.groupby("config"):
        ok, _ = ev.config_eligible(g[["g1a", "g1b", "g2", "g3"]].to_dict("records"))
        item = {"all_dev": dict(_rates(g), eligible=ok), "per_experiment": {}, "by_g4": {}}
        for e in dev:
            ge = g[g["exp"] == e]
            ok_e, _ = ev.config_eligible(ge[["g1a", "g1b", "g2", "g3"]].to_dict("records"))
            item["per_experiment"][e] = dict(_rates(ge), eligible=ok_e)
        for flag, name in ((True, "g4_pass"), (False, "g4_fail")):
            gg = g[g["g4_pass"] == flag]
            item["by_g4"][name] = dict(_rates(gg), n_policies=int(len(gg[["exp", "train_seed"]].drop_duplicates())))
        out[c] = item
    return {"fail_rate_max": ev.FAIL_RATE_MAX, "configs": out}
