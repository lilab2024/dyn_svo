"""Acceleration-noise sd per (fold, regime) from calibration residuals.

  python scripts/noise_sd.py --params-run idm_v1
Writes results/calibration/<params_run>/noise_sd.csv and noise_sd.json (provenance).
"""
import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dyn_svo import provenance as pv
from dyn_svo.calibration import noise, runner
from dyn_svo.calibration import params as cp
from dyn_svo.data import loader as ld
from dyn_svo.data import manifest as mf
from dyn_svo.data import split as sp

CONFIGS = ("sim", "data_split", "data_manifest", "calibration")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--params-run", default=cp.DEFAULT_RUN)
    args = ap.parse_args(argv)
    data_check = mf.verify_data("full")   # official result run
    prov = pv.provenance(CONFIGS, data_check)
    table = cp.load_table(args.params_run)
    regimes, pairs = {}, {}
    for e in sp.experiments("dev"):
        df = ld.load_experiment(e)
        regimes[e] = str(df["regime"].iloc[0])
        pairs[e] = ld.follower_pairs(df)
    out = noise.noise_table(runner.folds(), regimes, pairs, table)
    run_dir = mf.data_root() / "dyn_svo" / "results" / "calibration" / args.params_run
    out.to_csv(run_dir / "noise_sd.csv", index=False)
    runner.write_json_atomic(run_dir / "noise_sd.json", {
        "rows": out.to_dict("records"),
        "params_csv_md5": hashlib.md5((run_dir / "params.csv").read_bytes()).hexdigest(),
        "provenance": prov})
    print(out.round(4).to_string(index=False))
    print("written:", run_dir / "noise_sd.csv")


if __name__ == "__main__":
    main()
