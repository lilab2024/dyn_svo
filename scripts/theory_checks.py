"""Theory checks with the calibrated parameters: gains, stability, long-wave coefficients,
signs and LMI certificates per regime, from development data and parameters only.

  python scripts/theory_checks.py --run-name theory_v1
Writes results/theory/<run>/theory.json (needs a clean repository).
"""
import argparse
import json
import sys
import time
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from dyn_svo import provenance as pv
from dyn_svo.analysis import human_policy as hp
from dyn_svo.calibration import params as cp
from dyn_svo.calibration import runner
from dyn_svo.config import load_config
from dyn_svo.control import mpc as M
from dyn_svo.data import loader as ld
from dyn_svo.data import manifest as mf
from dyn_svo.data import split as sp
from dyn_svo.theory import checks as K

CONFIGS = ("sim", "data_split", "data_manifest", "metrics", "baselines", "mpc", "av")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    args = ap.parse_args(argv)
    t0 = time.time()
    data_check = mf.verify_data("full")   # official result run
    prov = pv.provenance(CONFIGS, data_check)
    pv.require_clean(prov)
    split = sp.load_split()
    t_from = split["segments"]["drop_first_s"] + load_config("metrics")["warmup_s"]
    window = load_config("baselines")["mean_speed_window_s"]
    rows, rings = [], []
    for exp in sp.experiments("dev", split=split):
        df = ld.load_experiment(exp)
        rows.append(hp.follower_rows(df, t_from, window_s=window))
        rings.append((exp, df["regime"].iloc[0], df["veh_id"].nunique()))
    rows = pd.concat(rows, ignore_index=True)
    speeds = K.speed_stats(rows)
    policy = M.load_policy()
    table = cp.load_table()
    idm_by_regime = {r: K.pooled_idm(table, r) for r in policy}
    out = K.run_all(policy, idm_by_regime, speeds, K.ring_sizes(rings),
                    log=lambda s: print(s, "%.0f s" % (time.time() - t0), flush=True))
    path = mf.data_root() / "dyn_svo" / "results" / "theory" / args.run_name / "theory.json"
    runner.write_json_atomic(path, {
        "theory": out, "settings": K.SETTINGS, "t_from_s": t_from,
        "human_policy_run": load_config("mpc")["human_policy_run"],
        "solver_packages": {p: version(p) for p in ("cvxpy", "clarabel")},
        "elapsed_s": time.time() - t0, "provenance": prov})
    for r, v in out.items():
        print(r, "signs positive:", v["signs"]["positive"],
              "| intervals verified:", [all(c[k]["verified"] for k in c if k.startswith(("continuous_", "discrete_")))
                                        for c in v["certificates"]["intervals"]])
    print("written:", path)


if __name__ == "__main__":
    main()
