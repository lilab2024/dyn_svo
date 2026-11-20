"""Stage-2 matched fixed control."""
import importlib.util
import json
from pathlib import Path

import pytest

from dyn_svo.control import mpc as M
from dyn_svo.control.platform import AVPlatform
from dyn_svo.experiments import configs as cf

POL = {"s0": 3.51, "h": 1.35, "sigma_gap": 2.76, "sigma_dv": 0.55, "sigma_a": 0.29, "sigma_u": 0.73}
THR = {"I": {"min_gap": 1.0, "speed_ratio": 0.9, "mean_gap": 30.0}}


def _script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "matched_control.py"
    spec = importlib.util.spec_from_file_location("matched_control_script", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _task(tmp, name, fam, phis):
    rows = [{"av": i, "seed": 0, "phi_mean": p, "follower_sigma": [1.0]} for i, p in enumerate(phis)]
    rec = {"task": {"config": {"name": name, "family": fam, "rank": [0]}, "variant": "r5"},
           "result": {"exp": "A", "regime": "I", "rows": rows}, "provenance": {"commit": "c", "dirty": False}}
    (tmp / "tasks").mkdir(parents=True, exist_ok=True)
    (tmp / "tasks" / (name + "__A.json")).write_text(json.dumps(rec), encoding="utf-8")


def test_matched_phi_mean_and_rounding(tmp_path):
    mc = _script()
    _task(tmp_path, "dyn_x", "mpc_dynamic", [1.0, 1.1, 1.23456])
    _task(tmp_path, "fixed_0", "mpc_fixed", [0.0, 0.0, 0.0])
    phi, n = mc.matched_phi(tmp_path, "dyn_x")
    assert n == 3 and phi == round((1.0 + 1.1 + 1.23456) / 3, 4)


@pytest.mark.parametrize("d,ok", [({"A": -0.01, "C": -0.02, "E": -0.01, "F": 0.01, "G": -0.01}, True),
                                  ({"A": -0.05, "C": -0.02, "E": 0.01, "F": 0.01, "G": -0.01}, False),
                                  ({"A": -0.01, "C": -0.01, "E": -0.01, "F": -0.01, "G": 0.10}, False)])
def test_dev_rule(d, ok):
    assert _script().dev_rule(d) is ok


def test_make_driver_matched_is_fixed_phi():
    cfg = {"name": "fixed_match", "family": "matched", "phi": 1.1, "rank": [0]}
    d = cf.make_driver(cfg, "I", {"thresholds": THR, "policy": {"I": POL}})
    assert isinstance(d, AVPlatform) and isinstance(d.ctrl.phi_source, M.FixedPhi) and d.ctrl.phi_source.phi == 1.1


def test_main_refuses_without_dynamic_selection(tmp_path, monkeypatch):
    mc = _script()
    src = tmp_path / "dyn_svo" / "results" / "eval" / "eval_dev_v5"
    src.mkdir(parents=True)
    (src / "summary.json").write_text(json.dumps({"variant": "primary", "final_selection": {"mpc_dynamic": None}}), encoding="utf-8")
    monkeypatch.setattr(mc, "results_dir", lambda name: tmp_path / "dyn_svo" / "results" / "eval" / name)
    with pytest.raises(SystemExit, match="no matched control"):
        mc.main(["--source-run", "eval_dev_v5", "--run-name", "eval_dev_v5_match"])
    with pytest.raises(SystemExit, match="token match"):
        mc.main(["--source-run", "eval_dev_v5", "--run-name", "eval_dev_v5_x"])
