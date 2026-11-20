"""Evaluation runs of RL policies.

  python scripts/rl_evaluate.py --run-name eval_dev_rl_tune_v1 --train-run rl_tune_v1 --stage tune --mode criterion --official --workers 20
  python scripts/rl_evaluate.py --run-name eval_dev_rl_v1 --train-run rl_train_v1 --stage train --mode full --official --workers 20
  python scripts/rl_evaluate.py --run-name eval_dev_r5_rl_v1 --train-run rl_train_v1 --stage train --mode full --variant r5 --official --workers 20
  python scripts/rl_evaluate.py --run-name eval_dev_rl_v1 --train-run rl_train_v1 --stage train --mode full --official --summarize
  python scripts/rl_evaluate.py --run-name eval_dev_rl_v1 --train-run rl_train_v1 --stage train --mode full --official --eligibility-report --g4-run g4_rl_train_v1
Mode full also runs the all-HDV tasks hdv__<exp>; --summarize writes summary.json;
--eligibility-report writes eligibility.json (descriptive guardrail counts, also split by G4).
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dyn_svo import provenance as pv
from dyn_svo.calibration import runner as crun
from dyn_svo.control import platform as pf
from dyn_svo.config import load_config
from dyn_svo.data import manifest as mf
from dyn_svo.data import split as sp
from dyn_svo.experiments import runner as erun
from dyn_svo.rl import evaluate as V
from dyn_svo.rl import summary as RS

CONFIGS = tuple(erun.CONFIGS) + ("rl",)


def results_dir(run_name):
    return mf.data_root() / "dyn_svo" / "results" / "eval" / run_name


def train_dir(train_run):
    return mf.data_root() / "dyn_svo" / "results" / "rl" / train_run


def check_args(args):
    if args.official == args.run_name.startswith("debug_"):
        raise SystemExit("Official runs must not start with debug_; other runs must.")
    tokens = args.run_name.split("_")                     # variant names as tokens, as in evaluate.py
    if args.variant != "primary" and args.variant not in tokens:
        raise SystemExit("A non-primary run name must contain the variant name.")
    if args.variant == "primary" and any(v in tokens for v in erun.variant_names() if v != "primary"):
        raise SystemExit("A primary run name must not contain a variant name.")
    if (args.mode == "criterion") != (args.stage == "tune"):
        raise SystemExit("Mode criterion belongs to stage tune and stage tune to mode criterion.")
    if args.stage == "tune" and args.variant != "primary":
        raise SystemExit("The tuning evaluation runs in the primary environment only.")
    if args.official and args.max_tasks:
        raise SystemExit("Official runs evaluate every task.")
    if getattr(args, "summarize", False):
        if args.stage != "train" or args.mode != "full":
            raise SystemExit("--summarize is defined for stage train in mode full.")
        if args.workers or args.max_tasks:
            raise SystemExit("--summarize runs no tasks (no --workers, no --max-tasks).")
    if getattr(args, "eligibility_report", False):
        if getattr(args, "summarize", False):
            raise SystemExit("--eligibility-report and --summarize are separate calls.")
        if args.stage != "train" or args.mode != "full":
            raise SystemExit("--eligibility-report is defined for stage train in mode full.")
        if args.workers or args.max_tasks:
            raise SystemExit("--eligibility-report runs no tasks (no --workers, no --max-tasks).")
        if not getattr(args, "g4_run", None):
            raise SystemExit("--eligibility-report needs --g4-run.")


def check_training(tasks, tdir):
    """Every needed training task must be finished; its provenance is shown once."""
    tasks = [t for t in tasks if t["config"] != V.HDV]          # all-HDV tasks need no policy
    missing = [t["train_task_id"] for t in tasks if not (tdir / "tasks" / (t["train_task_id"] + ".json")).exists()]
    if missing:
        raise SystemExit("%d training tasks are not finished, e.g. %s" % (len(missing), missing[0]))
    commits = set()
    for t in tasks:
        rec = json.loads((tdir / "tasks" / (t["train_task_id"] + ".json")).read_text(encoding="utf-8"))
        commits.add((rec["provenance"]["commit"][:7], rec["provenance"]["dirty"]))
    return sorted(commits)


def child_args(args, k, data_check):
    cmd = [sys.executable, __file__, "--run-name", args.run_name, "--train-run", args.train_run, "--stage", args.stage,
           "--mode", args.mode, "--variant", args.variant, "--shard", str(k) + "/" + str(args.workers),
           "--progress-s", str(args.progress_s), "--parent-check", data_check]
    for flag, val in (("--max-tasks", args.max_tasks), ("--phi-match", args.phi_match)):
        if val is not None:
            cmd += [flag, str(val)]
    if args.official:
        cmd.append("--official")
    return cmd


def launch(args, n_tasks):
    run_dir = results_dir(args.run_name)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    data_check = mf.verify_data("full" if args.official else "size")
    procs = []
    for k in range(args.workers):
        log = open(run_dir / "logs" / ("shard_" + str(k) + ".log"), "a", encoding="utf-8")
        procs.append((subprocess.Popen(child_args(args, k, data_check), stdout=log, stderr=subprocess.STDOUT), log))
    t0 = time.time()
    next_report = t0 + args.progress_s
    try:
        while any(p.poll() is None for p, _ in procs):
            time.sleep(5)
            if time.time() >= next_report:
                n_done = len(list((run_dir / "tasks").glob("*.json")))
                print("[%.0f s] task files: %d of %d" % (time.time() - t0, n_done, n_tasks), flush=True)
                next_report += args.progress_s
    except KeyboardInterrupt:
        for p, _ in procs:
            p.terminate()
        for p, _ in procs:
            p.wait()
        print("Interrupted: all shard processes terminated.")
        raise
    finally:
        for _, log in procs:
            log.close()
    codes = [p.returncode for p, _ in procs]
    print("shard exit codes:", codes, "| wall %.0f s" % (time.time() - t0))
    if any(codes):
        raise SystemExit("Some shards failed; see " + str(run_dir / "logs"))


def summarize(args, tasks, dev):
    """ refuse incomplete or mixed runs, then write summary.json."""
    run_dir = results_dir(args.run_name)
    records = RS.load_records(run_dir)
    probs = RS.run_problems(records, tasks, args.variant, int(load_config("eval")["seeds"]))
    train_runs = {r["provenance"]["config_md5"].get("train_run") for r in records}
    if train_runs and train_runs != {args.train_run}:
        probs.append("train run of the tasks: %s, expected %s" % (sorted(map(str, train_runs)), args.train_run))
    if probs:
        print("Nothing written. Problems:")
        for p in probs:
            print(" -", p)
        raise SystemExit("summary refused (%d problems)" % len(probs))
    df = RS.load_rows(records)
    out = RS.build_rl_summary(df, pf.load_thresholds(), dev)
    out.update(variant=args.variant, run_name=args.run_name, train_run=args.train_run, n_tasks=len(records),
               train_commits=sorted({r["result"]["train_commit"] for r in records if r["task"]["config"] != V.HDV}),
               summary_provenance=pv.provenance(CONFIGS, "not_applicable"))
    crun.write_json_atomic(run_dir / "summary.json", out)
    keys = ("commits", "any_dirty", "n_rows", "n_tasks", "train_commits", "final_selection", "cv_d", "phi_match")
    print(json.dumps({k: out[k] for k in keys}, indent=1))


def eligibility(args, tasks, dev):
    """ refuse incomplete or mixed runs (evaluation and G4), then write eligibility.json (descriptive)."""
    run_dir = results_dir(args.run_name)
    records = RS.load_records(run_dir)
    probs = RS.run_problems(records, tasks, args.variant, int(load_config("eval")["seeds"]))
    train_runs = {r["provenance"]["config_md5"].get("train_run") for r in records}
    if train_runs and train_runs != {args.train_run}:
        probs.append("train run of the tasks: %s, expected %s" % (sorted(map(str, train_runs)), args.train_run))
    g4 = RS.load_records(train_dir(args.g4_run))
    probs += RS.g4_problems(g4, tasks, args.train_run)
    if probs:
        print("Nothing written. Problems:")
        for p in probs:
            print(" -", p)
        raise SystemExit("eligibility report refused (%d problems)" % len(probs))
    out = RS.eligibility_report(RS.load_rows(records), pf.load_thresholds(), dev, g4)
    out.update(variant=args.variant, run_name=args.run_name, train_run=args.train_run, g4_run=args.g4_run, n_tasks=len(records),
               eval_commits=sorted({r["provenance"]["commit"] for r in records}),
               g4_commits=sorted({r["provenance"]["commit"] for r in g4}),
               report_provenance=pv.provenance(CONFIGS, "not_applicable"))
    crun.write_json_atomic(results_dir(args.run_name) / "eligibility.json", out)
    f = lambda x: "NA" if x is None else "%.4f" % x
    for c, v in sorted(out["configs"].items()):
        line = c + " | eligible on all dev: " + str(v["all_dev"]["eligible"])
        for name, x in (("all", v["all_dev"]), ("G4 pass", v["by_g4"]["g4_pass"]), ("G4 fail", v["by_g4"]["g4_fail"])):
            line += " | %s: runs %d%s coll %d g1b %s g2 %s g3 %s" % (name, x["n_runs"], (" pol %d" % x["n_policies"]) if "n_policies" in x else "",
                                                                     x["collisions"], f(x["fail_rate_g1b"]), f(x["fail_rate_g2"]), f(x["fail_rate_g3"]))
        print(line, flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--train-run", required=True)
    ap.add_argument("--stage", required=True, choices=("tune", "train", "match"))
    ap.add_argument("--mode", required=True, choices=V.MODES)
    ap.add_argument("--variant", default="primary")
    ap.add_argument("--official", action="store_true")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--max-tasks", type=int, default=None)
    ap.add_argument("--phi-match", type=float, default=None)
    ap.add_argument("--progress-s", type=int, default=300)
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--eligibility-report", action="store_true")
    ap.add_argument("--g4-run", default=None)
    ap.add_argument("--parent-check", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    erun.variant_settings(args.variant)
    check_args(args)
    dev = sp.experiments("dev")
    tasks = V.eval_tasks(args.stage, dev, args.mode, args.variant, phi_match=args.phi_match)
    tdir = train_dir(args.train_run)
    if args.summarize:
        summarize(args, tasks, dev)
        return
    if args.eligibility_report:
        eligibility(args, tasks, dev)
        return
    if args.workers > 0:
        print("tasks:", len(tasks), "| training commits (commit, dirty):", check_training(tasks, tdir),
              "| workers:", args.workers, flush=True)
        launch(args, len(tasks))
        return
    if args.parent_check:
        mf.verify_data("size")
        data_check = args.parent_check + " (parent)"
    else:
        data_check = mf.verify_data("full" if args.official else "size")
    prov = pv.provenance(CONFIGS, data_check)
    prov["config_md5"]["train_run"] = args.train_run         # compared on resume
    prov["config_md5"]["eval_mode"] = args.mode
    prov["variant"] = args.variant
    if args.official:
        pv.require_clean(prov)
    k, n = (int(s) for s in args.shard.split("/"))
    ctx = erun.build_context(dev)
    ctx["train_dir"] = str(tdir)
    mine = crun.shard(tasks, k, n)
    print("tasks:", len(tasks), "| shard", args.shard, "| mine", len(mine), flush=True)
    crun.run_shard(results_dir(args.run_name), mine, ctx, prov, max_tasks=args.max_tasks,
                   progress_s=args.progress_s, log=lambda s: print(s, flush=True), task_fn=V.run_task)


if __name__ == "__main__":
    main()
