"""PPO training runs of RL-MLP.

  python scripts/rl_train.py --run-name debug_rl_timing --stage tune --steps 50000 --only-configs rl_tune_lr0.0003_g0.99 --only-sets loeo_A --seeds 0,1 --workers 2
  python scripts/rl_train.py --run-name rl_tune_v1 --stage tune --official --workers 20
  python scripts/rl_train.py --run-name rl_train_v1 --stage train --official --workers 20
  python scripts/rl_train.py --run-name rl_match_v1 --stage match --phi-match 1.0954 --official --workers 20
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
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
from dyn_svo.rl import runner as R


def results_dir(run_name):
    return mf.data_root() / "dyn_svo" / "results" / "rl" / run_name


def check_args(args):
    if args.official == args.run_name.startswith("debug_"):
        raise SystemExit("Official runs must not start with debug_; other runs must.")
    if args.official and (args.steps is not None or args.only_configs or args.only_sets or args.seeds or args.max_tasks):
        raise SystemExit("Official runs take the steps from config/rl.yaml and run every task (no filters).")


def select(tasks, args):
    if args.only_configs:
        tasks = [t for t in tasks if t["config"] in args.only_configs.split(",")]
    if args.only_sets:
        tasks = [t for t in tasks if t["train_set"] in args.only_sets.split(",")]
    if args.seeds:
        keep = {int(s) for s in args.seeds.split(",")}
        tasks = [t for t in tasks if t["seed"] in keep]
    if not tasks:
        raise SystemExit("No task left after the filters.")
    return tasks


def steps_for(args):
    steps = load_config("rl")["training_steps"] if args.steps is None else args.steps
    if steps is None:
        raise SystemExit("config/rl.yaml training_steps is not set; debug runs pass --steps.")
    return int(steps)


def child_args(args, k, data_check):
    cmd = [sys.executable, __file__, "--run-name", args.run_name, "--stage", args.stage, "--shard", str(k) + "/" + str(args.workers),
           "--progress-s", str(args.progress_s), "--parent-check", data_check]
    for flag, val in (("--steps", args.steps), ("--only-configs", args.only_configs), ("--only-sets", args.only_sets),
                      ("--seeds", args.seeds), ("--max-tasks", args.max_tasks), ("--phi-match", args.phi_match)):
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


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--stage", required=True, choices=R.STAGES)
    ap.add_argument("--official", action="store_true")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--only-configs", default=None)
    ap.add_argument("--only-sets", default=None)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--max-tasks", type=int, default=None)
    ap.add_argument("--phi-match", type=float, default=None)
    ap.add_argument("--progress-s", type=int, default=300)
    ap.add_argument("--parent-check", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    check_args(args)
    dev = sp.experiments("dev")
    tasks = select(R.enumerate_tasks(args.stage, dev, phi_match=args.phi_match), args)
    steps = steps_for(args)
    if args.workers > 0:
        print("tasks:", len(tasks), "| steps per task:", steps, "| workers:", args.workers, flush=True)
        launch(args, len(tasks))
        return
    if args.parent_check:
        mf.verify_data("size")
        data_check = args.parent_check + " (parent)"
    else:
        data_check = mf.verify_data("full" if args.official else "size")
    prov = pv.provenance(R.CONFIGS, data_check)
    prov["config_md5"]["training_steps"] = str(steps)        # part of the compatibility check
    prov["rl_packages"] = R.rl_versions()
    if args.official:
        pv.require_clean(prov)
    k, n = (int(s) for s in args.shard.split("/"))
    ctx = erun.build_context(dev)
    ctx.update(run_dir=str(results_dir(args.run_name)), steps=steps)
    mine = crun.shard(tasks, k, n)
    print("tasks:", len(tasks), "| shard", args.shard, "| mine", len(mine), "| steps", steps, flush=True)
    crun.run_shard(results_dir(args.run_name), mine, ctx, prov, max_tasks=args.max_tasks,
                   progress_s=args.progress_s, log=lambda s: print(s, flush=True), task_fn=R.run_task)


if __name__ == "__main__":
    main()
