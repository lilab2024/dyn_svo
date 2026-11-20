"""Stage-2 matched fixed control on dev.

  python scripts/matched_control.py --source-run eval_dev_r5_v5 --run-name eval_dev_r5_v5_match
phi_match = mean of phi_mean over all dev episodes of the source run's final dynamic selection, 4 decimals.
One run (one commit) evaluates hdv, that dynamic configuration and fixed_match on every dev experiment with
the source variant; tasks are written as by evaluate.py, plus match_summary.json with per-experiment log FOR,
d_match = dynamic - matched, guardrail rates and the dev rule (used only for the stopping rule).
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from dyn_svo import provenance as pv
from dyn_svo.calibration import runner as crun
from dyn_svo.control import platform as pf
from dyn_svo.data import manifest as mf
from dyn_svo.data import split as sp
from dyn_svo.experiments import configs as cf
from dyn_svo.experiments import runner, summary


def results_dir(run_name):
    return mf.data_root() / "dyn_svo" / "results" / "eval" / run_name


def matched_phi(source_dir, config_name):
    """Mean phi_mean over every dev episode of one configuration, rounded to 4 decimals."""
    df = summary.load_rows(source_dir)
    vals = df.loc[df["config"] == config_name, "phi_mean"].astype(float)
    if vals.empty or vals.isna().any():
        raise SystemExit("No complete phi_mean values for " + config_name)
    return round(float(vals.mean()), 4), int(len(vals))


def dev_rule(d_by_exp):
    """ state dependence on dev iff mean d_match < 0 and d_match < 0 in at least 4 of 5 experiments."""
    d = np.asarray(list(d_by_exp.values()), dtype=float)
    return bool(d.mean() < 0 and int((d < 0).sum()) >= 4)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-run", required=True)
    ap.add_argument("--run-name", required=True)
    args = ap.parse_args(argv)
    src, out_dir = results_dir(args.source_run), results_dir(args.run_name)
    if args.run_name.startswith("debug_") or "match" not in args.run_name.split("_"):
        raise SystemExit("Official matched runs need a name with the token match and no debug_ prefix.")
    if out_dir.exists():
        raise SystemExit("Run dir exists, refusing: " + str(out_dir))
    s = json.loads((src / "summary.json").read_text(encoding="utf-8"))
    variant, dyn = s.get("variant", "primary"), s["final_selection"].get("mpc_dynamic")
    if dyn is None:
        raise SystemExit("No final dynamic selection in " + args.source_run + "; no matched control.")
    if variant != "primary" and variant not in args.run_name.split("_"):
        raise SystemExit("Run name must contain the variant " + variant + ".")
    phi, n_ep = matched_phi(src, dyn)
    data_check = mf.verify_data("full")
    prov = pv.provenance(runner.CONFIGS, data_check)
    pv.require_clean(prov)
    dev = sp.experiments("dev")
    by_name = {c["name"]: c for c in cf.all_configs()}
    match_cfg = {"name": "fixed_match", "family": "matched", "phi": phi, "rank": [0]}
    ctx = runner.build_context(dev)
    print("variant", variant, "| dynamic", dyn, "| phi_match", phi, "(", round(phi / (math.pi / 16), 3), "pi/16 ) from", n_ep, "episodes", flush=True)
    for c in (by_name["hdv"], by_name[dyn], match_cfg):
        for e in dev:
            task = {"task_id": c["name"] + "__" + e, "config": c, "exp": e, "variant": variant}
            res = runner.run_task(task, ctx)
            crun.write_json_atomic(out_dir / "tasks" / (task["task_id"] + ".json"), {"task": task, "result": res, "provenance": prov})
            print("done", task["task_id"], flush=True)
    df = summary.add_log_for(summary.load_rows(out_dir))
    summary.check_variant(df, variant)
    vals = summary.experiment_values(df)
    thr = pf.load_thresholds()
    per = {e: {"log_for_dynamic": float(vals[(dyn, e)]), "log_for_matched": float(vals[("fixed_match", e)]),
               "d_match": float(vals[(dyn, e)] - vals[("fixed_match", e)])} for e in dev}
    d = {e: per[e]["d_match"] for e in dev}
    elig = {}
    for c in (dyn, "fixed_match"):
        ok, rates = summary.eligibility(df, c, dev, thr)
        elig[c] = dict(rates, eligible_all_dev=ok)
    out = {"source_run": args.source_run, "variant": variant, "dynamic": dyn, "phi_match": phi,
           "phi_match_pi16": phi / (math.pi / 16), "n_episodes_for_phi": n_ep, "per_experiment": per,
           "mean_d_match": float(np.mean(list(d.values()))), "n_negative": int(sum(v < 0 for v in d.values())),
           "state_dependence_on_dev": dev_rule(d), "eligibility": elig, "provenance": prov}
    crun.write_json_atomic(out_dir / "match_summary.json", out)
    print(json.dumps({k: out[k] for k in ("dynamic", "phi_match", "per_experiment", "mean_d_match", "n_negative",
                                          "state_dependence_on_dev", "eligibility")}, indent=1))


if __name__ == "__main__":
    main()
