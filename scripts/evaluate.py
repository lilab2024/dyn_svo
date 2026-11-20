"""Episode evaluation on dev experiments and dev tuning summary.

  python scripts/evaluate.py --run-name debug_eval --max-tasks 2
  python scripts/evaluate.py --run-name eval_dev_v1 --official --workers 8
  python scripts/evaluate.py --run-name eval_dev_v1 --official --summarize
  python scripts/evaluate.py --run-name eval_dev_r5_v1 --official --workers 8 --variant r5
  python scripts/evaluate.py --run-name eval_test_v5 --official --workers 12 --stage test
  python scripts/evaluate.py --run-name eval_test_r5_v5 --official --workers 12 --stage test --variant r5
  python scripts/evaluate.py --run-name eval_test_v5 --official --stage test --summarize
The test stage loads B, D and H only for an official run whose config/test_freeze.yaml matches the
development results it records; every other run keeps the test lock.
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
from dyn_svo.data import manifest as mf
from dyn_svo.data import split as sp
from dyn_svo.experiments import freeze as fz
from dyn_svo.experiments import runner, summary


def results_dir(run_name):
    return mf.data_root() / "dyn_svo" / "results" / "eval" / run_name


def check_run_name(name, official, variant="primary", stage="dev"):
    if official == name.startswith("debug_"):
        raise SystemExit("Official runs must not start with debug_; other runs must.")
    if stage == "test" and not (official and name.startswith("eval_test_")):
        raise SystemExit("The test stage needs --official and a run name starting with eval_test_.")
    if stage == "dev" and "test" in name.split("_"):
        raise SystemExit("Development run names must not contain the token test.")
    if official:
        tokens = name.split("_")
        others = [v for v in runner.variant_names() if v != "primary"]
        if variant != "primary" and variant not in tokens:
            raise SystemExit("Official runs of variant " + variant + " must contain it in the run name.")
        if variant == "primary" and any(v in tokens for v in others):
            raise SystemExit("Primary run names must not contain a variant name.")


def launch(args):
    run_dir = results_dir(args.run_name)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    data_check = mf.verify_data("full" if args.official else "size")
    procs = []
    for k in range(args.workers):
        cmd = [sys.executable, __file__, "--run-name", args.run_name, "--shard", str(k) + "/" + str(args.workers),
               "--progress-s", str(args.progress_s), "--parent-check", data_check]
        if args.official:
            cmd.append("--official")
        cmd += ["--variant", args.variant, "--stage", args.stage]
        if args.max_tasks is not None:
            cmd += ["--max-tasks", str(args.max_tasks)]
        log = open(run_dir / "logs" / ("shard_" + str(k) + ".log"), "a", encoding="utf-8")
        procs.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log))
    t0 = time.time()
    next_report = t0 + args.progress_s
    try:
        while any(p.poll() is None for p, _ in procs):
            time.sleep(5)
            if time.time() >= next_report:
                n_done = len(list((run_dir / "tasks").glob("*.json")))
                print("[%.0f s] task files: %d" % (time.time() - t0, n_done), flush=True)
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
    print("shard exit codes:", codes)
    if any(codes):
        raise SystemExit("Some shards failed; see " + str(run_dir / "logs"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--official", action="store_true")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--max-tasks", type=int, default=None)
    ap.add_argument("--progress-s", type=int, default=300)
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--parent-check", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--variant", default="primary")
    ap.add_argument("--stage", choices=("dev", "test"), default="dev")
    args = ap.parse_args(argv)
    runner.variant_settings(args.variant)
    check_run_name(args.run_name, args.official, args.variant, args.stage)
    if args.stage == "test":
        freeze = fz.load_freeze()
        problems = fz.verify_freeze(freeze, mf.data_root() / "dyn_svo" / "results")
        if problems:
            raise SystemExit("config/test_freeze.yaml does not match the development results: " + "; ".join(problems))
        if args.variant not in freeze["variants"]:
            raise SystemExit("variant " + args.variant + " is not frozen for the test run")
        exps = sp.experiments("test", allow_test=True)
        if sorted(exps) != sorted(freeze["test_experiments"]):
            raise SystemExit("test experiments differ from the freeze file")
        tasks = fz.test_tasks(freeze, args.variant)
        configs = runner.CONFIGS + ("test_freeze",)
    else:
        exps = sp.experiments("dev")
        tasks = runner.enumerate_tasks(exps, variant=args.variant)
        configs = runner.CONFIGS
    dev = exps
    if args.summarize and args.stage == "test":
        df = summary.load_rows(results_dir(args.run_name))
        summary.check_variant(df, args.variant)
        n_done = df[["config", "exp"]].drop_duplicates().shape[0] if len(df) else 0
        if n_done != len(tasks):
            raise SystemExit("Run incomplete: %d of %d tasks" % (n_done, len(tasks)))
        out = summary.build_test_summary(df, pf.load_thresholds(), freeze, args.variant)
        out["summary_provenance"] = pv.provenance(configs, "not_applicable")
        crun.write_json_atomic(results_dir(args.run_name) / "summary.json", out)
        print(json.dumps({k: out[k] for k in ("commits", "any_dirty", "n_rows", "experiment_log_for")}, indent=1))
        for k in ("h1", "matched_rule", "h2"):
            print(k, "supported:", None if out.get(k) is None else out[k]["supported"])
        return
    if args.summarize:
        df = summary.load_rows(results_dir(args.run_name))
        summary.check_variant(df, args.variant)
        n_done = df[["config", "exp"]].drop_duplicates().shape[0] if len(df) else 0
        if n_done != len(tasks):
            raise SystemExit("Run incomplete: %d of %d tasks" % (n_done, len(tasks)))
        out = summary.build_summary(df, pf.load_thresholds(), dev)
        out["variant"] = args.variant
        out["summary_provenance"] = pv.provenance(runner.CONFIGS, "not_applicable")
        crun.write_json_atomic(results_dir(args.run_name) / "summary.json", out)
        print(json.dumps({k: out[k] for k in ("commits", "any_dirty", "n_rows", "final_selection")}, indent=1))
        return
    if args.workers > 0:
        launch(args)
        return
    k, n = (int(s) for s in args.shard.split("/"))
    if args.parent_check:
        mf.verify_data("size")
        data_check = args.parent_check + " (parent)"
    else:
        data_check = mf.verify_data("full" if args.official else "size")
    prov = pv.provenance(configs, data_check)
    if args.official:
        pv.require_clean(prov)
    ctx = runner.build_context(dev, allow_test=(args.stage == "test"))
    mine = crun.shard(tasks, k, n)
    print("total tasks:", len(tasks), "| shard", args.shard, "| variant", args.variant, flush=True)
    crun.run_shard(results_dir(args.run_name), mine, ctx, prov, max_tasks=args.max_tasks,
                   progress_s=args.progress_s, log=lambda s: print(s, flush=True), task_fn=runner.run_task)


if __name__ == "__main__":
    main()
