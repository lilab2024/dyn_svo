"""Learnability gate of RL: G4 for every policy of a tuning run; the gate decides on the
policies of the selected candidate, the rest is descriptive. Stage train: G4 for the 500
out-of-fold policies of the main training on their held-out experiments, pass shares only, no gate. No FOR.

  python scripts/rl_gate.py --run-name gate_rl_tune_v2 --train-run rl_tune_v2 --selected rl_tune_lr0.0003_g0.98 --official --workers 12
  python scripts/rl_gate.py --run-name g4_rl_train_v1 --train-run rl_train_v1 --stage train --official --workers 12
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
from dyn_svo.config import load_config
from dyn_svo.data import manifest as mf
from dyn_svo.data import split as sp
from dyn_svo.experiments import runner as erun
from dyn_svo.rl import evaluate as V
from dyn_svo.rl import learnability as L
from dyn_svo.rl import runner as R


def out_dir(run_name):
    return mf.data_root() / "dyn_svo" / "results" / "rl" / run_name


def gate_tasks(dev, selected, rl=None, stage="tune"):
    """Stage tune: all tuning policies, 15 of them selected. Stage train: the out-of-fold policies (loeo_k sets
    only), none selected."""
    tasks = []
    for t in R.enumerate_tasks(stage, dev, rl):
        if t["train_set"] == "all_dev":
            continue
        tasks.append({"task_id": t["task_id"], "config": t["config"], "exp": t["train_set"][len("loeo_"):],
                      "fold": t["train_set"], "phi_cfg": t["phi_cfg"], "seed": t["seed"],
                      "selected": stage == "tune" and t["config"] == selected})
    if stage == "tune" and sum(t["selected"] for t in tasks) != 15:
        raise SystemExit("The selected candidate must have 15 tuning policies: " + str(selected))
    return tasks


def run_task(task, ctx, cfg=None):
    w, trec = V.load_policy(ctx["train_dir"], task["task_id"])
    amap = load_config("rl")["action_map"]
    rows, p99 = L.follow_check(w, amap, ctx, task["exp"], task["fold"], task["phi_cfg"])
    return {"config": task["config"], "exp": task["exp"], "seed": task["seed"], "selected": task["selected"],
            "action_map": amap, "train_commit": trec["provenance"]["commit"], "policy_md5": trec["result"]["policy_md5"],
            "rows": rows, "g4": L.g4(rows, p99)}


def check_args(args):
    if args.official == args.run_name.startswith("debug_"):
        raise SystemExit("Official runs must not start with debug_; other runs must.")
    if args.stage not in ("tune", "train"):
        raise SystemExit("--stage must be tune or train.")
    if args.stage == "tune" and not args.selected:
        raise SystemExit("Stage tune needs --selected.")
    if args.stage == "train" and args.selected:
        raise SystemExit("Stage train has no selected candidate and no gate.")


def summarize(results, stage, selected):
    return L.gate_summary(results, selected) if stage == "tune" else L.train_summary(results)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--train-run", required=True)
    ap.add_argument("--stage", default="tune", choices=("tune", "train"))
    ap.add_argument("--selected", default=None)
    ap.add_argument("--official", action="store_true")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--parent-check", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    check_args(args)
    dev = sp.experiments("dev")
    tasks = gate_tasks(dev, args.selected, stage=args.stage)
    tdir = mf.data_root() / "dyn_svo" / "results" / "rl" / args.train_run
    missing = [t["task_id"] for t in tasks if not (tdir / "tasks" / (t["task_id"] + ".json")).exists()]
    if missing:
        raise SystemExit("%d training tasks missing, e.g. %s" % (len(missing), missing[0]))
    odir = out_dir(args.run_name)
    if args.workers > 0:
        (odir / "logs").mkdir(parents=True, exist_ok=True)
        data_check = mf.verify_data("full" if args.official else "size")
        print("tasks:", len(tasks), "| stage:", args.stage, "| selected:", args.selected, "| workers:", args.workers, flush=True)
        procs = []
        for k in range(args.workers):
            cmd = [sys.executable, __file__, "--run-name", args.run_name, "--train-run", args.train_run, "--stage",
                   args.stage, "--shard", "%d/%d" % (k, args.workers), "--parent-check", data_check]
            if args.selected:
                cmd += ["--selected", args.selected]
            if args.official:
                cmd.append("--official")
            log = open(odir / "logs" / ("shard_%d.log" % k), "a", encoding="utf-8")
            procs.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log))
        t0 = time.time()
        while any(p.poll() is None for p, _ in procs):
            time.sleep(5)
        for _, log in procs:
            log.close()
        codes = [p.returncode for p, _ in procs]
        print("shard exit codes:", codes, "| wall %.0f s" % (time.time() - t0), flush=True)
        if any(codes):
            raise SystemExit("Some shards failed; see " + str(odir / "logs"))
        results = [json.loads(f.read_text(encoding="utf-8"))["result"] for f in sorted((odir / "tasks").glob("*.json"))]
        if len(results) != len(tasks):
            raise SystemExit("task files %d of %d; no summary written" % (len(results), len(tasks)))
        summary = summarize(results, args.stage, args.selected)
        (odir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
        if args.stage == "tune":
            print("GATE", "PASSED" if summary["gate_pass"] else "FAILED", json.dumps(summary), flush=True)
        else:
            print("G4 (descriptive, no gate)", json.dumps({k: summary[k] for k in ("n_policies", "n_pass", "pass_share_all", "pass_counts")}), flush=True)
        return
    if args.parent_check:
        mf.verify_data("size")
        data_check = args.parent_check + " (parent)"
    else:
        data_check = mf.verify_data("full" if args.official else "size")
    prov = pv.provenance(R.CONFIGS, data_check)
    prov["config_md5"]["train_run"] = args.train_run
    prov["config_md5"]["selected"] = args.selected
    prov["config_md5"]["stage"] = args.stage
    if args.official:
        pv.require_clean(prov)
    ctx = erun.build_context(dev)
    ctx["train_dir"] = str(tdir)
    k, n = (int(s) for s in args.shard.split("/"))
    crun.run_shard(odir, crun.shard(tasks, k, n), ctx, prov, log=lambda s: print(s, flush=True), task_fn=run_task)


if __name__ == "__main__":
    main()
