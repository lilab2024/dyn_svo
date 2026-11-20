"""Human following policy statistics for the MPC.

  python scripts/human_policy.py --run-name human_policy_v1
Writes results/mpc/<run>/policy.json.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from dyn_svo import provenance as pv
from dyn_svo.analysis import human_policy as hp
from dyn_svo.calibration import runner
from dyn_svo.config import load_config
from dyn_svo.data import loader as ld
from dyn_svo.data import manifest as mf
from dyn_svo.data import split as sp

CONFIGS = ("sim", "data_split", "data_manifest", "metrics", "baselines", "mpc")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    args = ap.parse_args(argv)
    data_check = mf.verify_data("full")   # official result run
    prov = pv.provenance(CONFIGS, data_check)
    split = sp.load_split()
    t_from = split["segments"]["drop_first_s"] + load_config("metrics")["warmup_s"]
    window = load_config("baselines")["mean_speed_window_s"]
    rows = pd.concat([hp.follower_rows(ld.load_experiment(e), t_from, window_s=window)
                      for e in sp.experiments("dev", split=split)], ignore_index=True)
    pol = hp.policy_from_rows(rows)
    out = mf.data_root() / "dyn_svo" / "results" / "mpc" / args.run_name / "policy.json"
    runner.write_json_atomic(out, {"policy": pol, "t_from_s": t_from, "provenance": prov})
    print(json.dumps(pol, indent=1))
    print("written:", out)


if __name__ == "__main__":
    main()
