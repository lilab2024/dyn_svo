"""Summarize an IDM calibration run and apply the pre-registered rules.

  python scripts/calibration_report.py --run-name idm_v1 --expected 534
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dyn_svo import provenance as pv
from dyn_svo.calibration import report, runner
from dyn_svo.config import load_config
from dyn_svo.data import manifest as mf


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--expected", type=int, required=True)
    args = ap.parse_args(argv)
    run_dir = mf.data_root() / "dyn_svo" / "results" / "calibration" / args.run_name
    df, missing = runner.summarize(run_dir, n_expected=args.expected)
    if missing:
        raise SystemExit("Run incomplete: " + str(missing) + " tasks missing")
    rep = report.build_report(df, load_config("calibration")["bounds"])
    rep["report_provenance"] = pv.provenance(runner.CONFIGS, "not_applicable")
    out = run_dir / "report.json"
    runner.write_json_atomic(out, rep)
    print(json.dumps(rep, indent=1, default=str))
    print("written:", out)


if __name__ == "__main__":
    main()
