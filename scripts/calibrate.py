"""Run IDM calibration.

Examples
  python scripts/calibrate.py --run-name debug_pilot --max-tasks 2
  python scripts/calibrate.py --run-name idm_v1 --official --workers 8
  python scripts/calibrate.py --run-name idm_v1 --summarize
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "2"

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dyn_svo import provenance as pv
from dyn_svo.calibration import runner
from dyn_svo.config import load_config
from dyn_svo.data import manifest as mf
from dyn_svo.data import split as sp


def results_dir(run_name):
    return mf.data_root() / "dyn_svo" / "results" / "calibration" / run_name


def check_run_name(name, official):
    if official == name.startswith("debug_"):
        raise SystemExit("Official runs must not start with debug_; other runs must.")


def prepare(official, parent_check=None):
    if parent_check:
        mf.verify_data("size")
        data_check = parent_check + " (parent)"
    else:
        data_check = mf.verify_data("full" if official else "size")
    prov = pv.provenance(runner.CONFIGS, data_check)
    if official:
        pv.require_clean(prov)
    cfg = load_config("calibration")
    bank, regimes = runner.build_segment_bank(sp.experiments("dev"), cfg["segment_s"])
    tasks = runner.enumerate_tasks(bank, regimes, runner.folds())
    return prov, cfg, bank, tasks


def launch(args):
    """Start one subprocess per shard; Ctrl+C terminates all of them first."""
    run_dir = results_dir(args.run_name)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    data_check = mf.verify_data("full" if args.official else "size")
    procs = []
    for k in range(args.workers):
        cmd = [sys.executable, __file__, "--run-name", args.run_name, "--shard",
               str(k) + "/" + str(args.workers), "--progress-s", str(args.progress_s),
               "--parent-check", data_check]
        if args.official:
            cmd.append("--official")
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
    args = ap.parse_args(argv)
    check_run_name(args.run_name, args.official)
    if args.summarize:
        prov, cfg, bank, tasks = prepare(False)
        df, missing = runner.summarize(results_dir(args.run_name), n_expected=len(tasks))
        print("rows:", len(df), "| missing tasks:", missing)
        return
    if args.workers > 0:
        launch(args)
        return
    k, n = (int(s) for s in args.shard.split("/"))
    prov, cfg, bank, tasks = prepare(args.official, args.parent_check)
    mine = runner.shard(tasks, k, n)
    print("total tasks:", len(tasks), "| shard", args.shard, flush=True)
    runner.run_shard(results_dir(args.run_name), mine, bank, prov, cfg, max_tasks=args.max_tasks,
                     progress_s=args.progress_s, log=lambda s: print(s, flush=True))


if __name__ == "__main__":
    main()
