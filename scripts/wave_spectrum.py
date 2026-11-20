"""Wave spectrum of dev real data and the detrend window W.

  python scripts/wave_spectrum.py --run-name spectrum_v1
Writes results/spectrum/<run>/spectrum.json; exits with an error if the rule result differs
from config/metrics.yaml.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dyn_svo import provenance as pv
from dyn_svo.analysis import spectrum as spc
from dyn_svo.calibration import runner
from dyn_svo.config import load_config
from dyn_svo.data import loader as ld
from dyn_svo.data import manifest as mf
from dyn_svo.data import split as sp

CONFIGS = ("sim", "data_split", "data_manifest", "metrics")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    args = ap.parse_args(argv)
    cfg = load_config("metrics")
    s = cfg["spectrum"]
    data_check = mf.verify_data("full")   # official result run
    rows = []
    for exp in sp.experiments("dev"):
        f, psd, n_veh, n_samp = spc.experiment_spectrum(
            ld.load_experiment(exp), segment_s=s["segment_s"], overlap=s["overlap"],
            nfft_factor=s["nfft_factor"])
        period, ratio, edge = spc.dominant_period(f, psd, tuple(s["period_range_s"]))
        rows.append({"exp": exp, "n_veh": n_veh, "min_samples_per_veh": n_samp,
                     "dominant_period_s": round(period, 2), "peak_over_median": round(ratio, 2),
                     "at_band_edge": edge})
    used = [r["dominant_period_s"] for r in rows if not r["at_band_edge"]]
    w = spc.detrend_window(used)
    out = {"rows": rows, "experiments_used": [r["exp"] for r in rows if not r["at_band_edge"]],
           "detrend_window_s": w, "config_detrend_window_s": cfg["detrend_window_s"],
           "provenance": pv.provenance(CONFIGS, data_check)}
    path = mf.data_root() / "dyn_svo" / "results" / "spectrum" / args.run_name / "spectrum.json"
    runner.write_json_atomic(path, out)
    print(json.dumps({k: out[k] for k in ("rows", "experiments_used", "detrend_window_s",
                                          "config_detrend_window_s")}, indent=1))
    print("written:", path)
    if w != cfg["detrend_window_s"]:
        raise SystemExit("Rule result differs from config/metrics.yaml - stop")


if __name__ == "__main__":
    main()
