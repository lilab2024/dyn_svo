""" level-2 validation: all-HDV open chains vs real data on dev experiments.

  python scripts/validate_chain.py --run-name chainval_v1 --no-noise          (deterministic IDM)
  python scripts/validate_chain.py --run-name chainval_v2 --seeds 5           (R0 = IDM + noise)
Writes results/validation/<run>/episodes.csv and summary.json.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from dyn_svo import provenance as pv
from dyn_svo.calibration import noise as nz
from dyn_svo.calibration import params as cp
from dyn_svo.calibration import runner
from dyn_svo.data import loader as ld
from dyn_svo.data import manifest as mf
from dyn_svo.data import split as sp
from dyn_svo.metrics import oscillation as osc
from dyn_svo.sim import chain

CONFIGS = ("sim", "data_split", "data_manifest", "metrics", "calibration")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--params-run", default=cp.DEFAULT_RUN)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--no-noise", action="store_true")
    args = ap.parse_args(argv)
    seeds = [0] if args.no_noise else list(range(args.seeds))
    data_check = mf.verify_data("full")   # official result run
    prov = pv.provenance(CONFIGS, data_check)
    split = sp.load_split()
    table = cp.load_table(args.params_run)
    params_path = mf.data_root() / "dyn_svo" / "results" / "calibration" / args.params_run / "params.csv"
    ntab = None if args.no_noise else nz.load_noise_table(args.params_run)
    rows, n_pooled = [], 0
    for exp in sp.experiments("dev", split=split):
        df = ld.load_experiment(exp)
        fold = cp.fold_for_experiment(exp, split)
        for av in sorted(int(v) for v in df["veh_id"].unique()):
            ep = chain.build_episode(df, av, t_start_s=split["segments"]["drop_first_s"])
            pvals = cp.vehicle_params(table, fold, ep["regime"], ep["order"])
            n_pooled += sum(src == "pooled" for _, src in pvals.values())
            sd = 0.0 if ntab is None else nz.noise_sd(ntab, fold, ep["regime"])
            per_seed = []
            for s in seeds:
                rng = None if sd == 0.0 else chain.episode_rng(s, exp, av)
                sim = chain.simulate(ep, [pvals[v][0] for v in ep["order"]], noise_sd=sd, rng=rng)
                per_seed.append(osc.episode_validation(ep, sim))
            row = osc.combine_seed_rows(per_seed)
            row["noise_sd"] = sd
            row["n_pooled_params"] = sum(src == "pooled" for _, src in pvals.values())
            rows.append(row)
        print("done", exp, flush=True)
    summary = osc.summarize_validation(rows)
    out_dir = mf.data_root() / "dyn_svo" / "results" / "validation" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).drop(columns=["log_ratio_by_position"]).to_csv(out_dir / "episodes.csv", index=False)
    runner.write_json_atomic(out_dir / "summary.json", {
        "summary": summary, "episodes": rows, "params_run": args.params_run,
        "noise": not args.no_noise, "seeds": seeds,
        "params_csv_md5": hashlib.md5(params_path.read_bytes()).hexdigest(),
        "n_pooled_params_total": int(n_pooled), "provenance": prov})
    print(pd.DataFrame(summary).round(4).to_string(index=False))
    print("pooled-parameter vehicles over all episodes:", n_pooled, "| episodes:", len(rows))
    print("written:", out_dir)


if __name__ == "__main__":
    main()
