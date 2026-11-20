"""Leader-activity thresholds a_lo, a_hi for the v5 phi_t.

  python scripts/phi_thresholds.py --run-name phi_thresholds_v1
Writes results/mpc/<run>/thresholds.json: per fold (loeo_k, all_dev), regime and window W_a, the p10 and
p90 of the causal speed SD (cooperation.leader_activity, ddof 0) over all vehicles of the fold's training
experiments. Real data at the simulation step from t = drop_first_s (as chain.build_episode), values kept
from t = drop_first_s + warmup_s. The R5 variant uses the same thresholds (raw data).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from dyn_svo import provenance as pv
from dyn_svo.calibration import runner
from dyn_svo.config import load_config
from dyn_svo.control import cooperation as co
from dyn_svo.data import loader as ld
from dyn_svo.data import manifest as mf
from dyn_svo.data import split as sp
from dyn_svo.sim import chain

CONFIGS = ("sim", "data_split", "data_manifest", "metrics", "cooperation")


def activity_samples(df, window_s, t0, t_from):
    """Activity of every vehicle's speed (each vehicle is some AV's leader), kept from t_from."""
    k0 = int(round(t0 * chain.DATA_HZ))
    ks = np.arange(k0, int(df["k"].max()) + 1, chain.STEP_FRAMES)
    piv = df.pivot(index="k", columns="veh_id", values="v").loc[ks]
    keep = ks / chain.DATA_HZ >= t_from
    return np.concatenate([co.leader_activity(piv[c].to_numpy(dtype=float), window_s)[keep] for c in piv.columns])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    args = ap.parse_args(argv)
    out = mf.data_root() / "dyn_svo" / "results" / "mpc" / args.run_name / "thresholds.json"
    if out.exists():
        raise SystemExit("Result exists, refusing to overwrite: " + str(out))
    data_check = mf.verify_data("full")   # official result run
    prov = pv.provenance(CONFIGS, data_check)
    pv.require_clean(prov)
    split = sp.load_split()
    dev = sp.experiments("dev", split=split)
    t0 = float(split["segments"]["drop_first_s"])
    t_from = t0 + float(load_config("metrics")["warmup_s"])
    cfg = load_config("cooperation")["leader_activity"]
    samples = {}
    for e in dev:
        df = ld.load_experiment(e)
        regime = str(df["regime"].iloc[0])
        for w in cfg["window_s"]:
            samples[(e, regime, co.window_key(w))] = activity_samples(df, w, t0, t_from)
    thr = co.thresholds_from_samples(samples, co.threshold_folds(dev), cfg["percentiles"])
    runner.write_json_atomic(out, {"thresholds": thr, "percentiles": cfg["percentiles"], "t_from_s": t_from,
                                   "dt": chain.STEP_FRAMES / chain.DATA_HZ, "provenance": prov})
    for fold, node in thr.items():
        for regime, ws in node.items():
            print(fold, regime, {w: (round(v["a_lo"], 3), round(v["a_hi"], 3), v["n"]) for w, v in ws.items()})
    print("written:", out)


if __name__ == "__main__":
    main()
