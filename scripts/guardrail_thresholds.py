"""Guardrail thresholds from dev human data.

  python scripts/guardrail_thresholds.py --run-name guardrails_v1
Writes results/metrics/<run>/thresholds.json and human_window_stats.csv.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from dyn_svo import provenance as pv
from dyn_svo.calibration import runner
from dyn_svo.config import load_config
from dyn_svo.data import loader as ld
from dyn_svo.data import manifest as mf
from dyn_svo.data import split as sp
from dyn_svo.metrics import evaluation as ev

CONFIGS = ("sim", "data_split", "data_manifest", "metrics")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    args = ap.parse_args(argv)
    data_check = mf.verify_data("full")   # official result run
    prov = pv.provenance(CONFIGS, data_check)
    split = sp.load_split()
    t_from = split["segments"]["drop_first_s"] + load_config("metrics")["warmup_s"]
    stats = pd.concat([ev.human_window_stats(ld.load_experiment(e), t_from)
                       for e in sp.experiments("dev", split=split)], ignore_index=True)
    thr = ev.thresholds_from_humans(stats)
    out_dir = mf.data_root() / "dyn_svo" / "results" / "metrics" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    stats.to_csv(out_dir / "human_window_stats.csv", index=False)
    runner.write_json_atomic(out_dir / "thresholds.json", {
        "thresholds": thr, "t_from_s": t_from, "percentiles": load_config("metrics")["guardrail_percentiles"],
        "provenance": prov})
    print(json.dumps(thr, indent=1))
    print("vehicle windows per experiment:", stats.groupby("exp").size().to_dict())
    print("written:", out_dir)


if __name__ == "__main__":
    main()
