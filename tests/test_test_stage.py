"""Test stage of the evaluation: lock, run names, freeze file, new field, decision rules.
Synthetic data only; no test experiment is ever loaded here."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dyn_svo.data import split as sp
from dyn_svo.experiments import configs as cf
from dyn_svo.experiments import freeze as fz
from dyn_svo.experiments import runner, summary

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("evaluate_script", ROOT / "scripts" / "evaluate.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)

P = {"v0": 15.0, "T": 1.2, "s0": 2.0, "a_max": 1.0, "b": 1.5}
THR = {"I": {"min_gap": 1.0, "speed_ratio": 0.9, "mean_gap": 30.0}}
CONST = {"min_for_reduction": 0.05, "chain_speed_noninferiority": 0.98}


def _no_files(monkeypatch):
    for mod, name in ((runner.cp, "load_table"), (runner.nz, "load_noise_table"), (runner.pf, "load_thresholds"),
                      (runner.M, "load_policy"), (runner.co, "load_phi_thresholds")):
        monkeypatch.setattr(mod, name, lambda *a, **k: {})


def test_lock_holds_outside_the_test_stage(monkeypatch):
    _no_files(monkeypatch)
    for e in sp.load_split()["roles"]["test"]:
        with pytest.raises(sp.TestSetLocked):
            runner.build_context([e])                    # the real loader refuses before reading any file
    with pytest.raises(sp.TestSetLocked):
        sp.experiments("test")
    seen = []
    monkeypatch.setattr(runner.ld, "load_experiment", lambda e, allow_test=False: seen.append((e, allow_test)) or "df")
    runner.build_context(["B"], allow_test=True)
    runner.build_context(["A"])
    assert seen == [("B", True), ("A", False)]


@pytest.mark.parametrize("name, official, stage, ok", [
    ("eval_test_v5", True, "test", True), ("eval_test_r5_v5", True, "test", True),
    ("eval_dev_v5", True, "test", False), ("debug_test_x", False, "test", False),
    ("eval_test_v5", True, "dev", False), ("eval_dev_v5", True, "dev", True)])
def test_run_names(name, official, stage, ok):
    variant = "r5" if "r5" in name.split("_") else "primary"
    if ok:
        ev.check_run_name(name, official, variant, stage)
    else:
        with pytest.raises(SystemExit):
            ev.check_run_name(name, official, variant, stage)


def test_test_stage_refuses_unofficial_runs():
    with pytest.raises(SystemExit):
        ev.main(["--run-name", "debug_x", "--stage", "test"])


def test_freeze_file_and_tasks():
    f = fz.load_freeze()
    assert f["test_experiments"] == ["B", "D", "H"] and f["parameter_fold"] == "all_dev"
    assert f["test_experiments"] == sp.load_split()["roles"]["test"]
    p, r = f["variants"]["primary"], f["variants"]["r5"]
    assert p["dynamic"] is None and p["matched"] is None and p["best_fixed"] == "fixed_0"
    assert r["dynamic"] == "dyn_W40_lo4_hi7p5" and r["best_fixed"] == "fixed_7" and r["matched"]["phi"] == 1.0954
    names = {c["name"] for c in cf.all_configs()}
    for v in f["variants"].values():
        assert set(v["configs"]) - {fz.MATCHED} <= names
    tp, tr = fz.test_tasks(f, "primary"), fz.test_tasks(f, "r5")
    assert len(tp) == 4 * 3 and len(tr) == 6 * 3
    m = [t for t in tr if t["config"]["name"] == fz.MATCHED]
    assert len(m) == 3 and all(t["config"]["family"] == "matched" and t["config"]["phi"] == 1.0954 for t in m)
    assert all(t["variant"] == "r5" and t["stage"] == "test" for t in tr)


def _write(root, rel, obj):
    p = Path(root) / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj), encoding="utf-8")
    return hashlib.md5(p.read_bytes()).hexdigest()


def _freeze(tmp):
    s0 = _write(tmp, "eval/dev/summary.json", {"variant": "primary", "final_selection": {"mpc_fixed": "fixed_0", "mpc_dynamic": None}})
    s1 = _write(tmp, "eval/dev_r5/summary.json", {"variant": "r5", "final_selection": {"mpc_fixed": "fixed_7", "mpc_dynamic": "dyn_a"}})
    s2 = _write(tmp, "eval/dev_r5_match/match_summary.json", {"dynamic": "dyn_a", "phi_match": 1.09541})
    return {"test_experiments": ["B"], "dev_sources": {"eval/dev/summary.json": s0, "eval/dev_r5/summary.json": s1,
                                                       "eval/dev_r5_match/match_summary.json": s2},
            "variants": {"primary": {"selection_source": "eval/dev/summary.json", "best_fixed": "fixed_0", "dynamic": None,
                                     "matched": None, "configs": ["hdv", "fixed_0"]},
                         "r5": {"selection_source": "eval/dev_r5/summary.json", "best_fixed": "fixed_7", "dynamic": "dyn_a",
                                "matched": {"source": "eval/dev_r5_match/match_summary.json", "phi": 1.0954},
                                "configs": ["hdv", "fixed_7", "dyn_a", "fixed_match"]}}}


def test_verify_freeze_detects_every_mismatch(tmp_path):
    f = _freeze(tmp_path)
    assert fz.verify_freeze(f, tmp_path) == []
    g = json.loads(json.dumps(f)); g["variants"]["r5"]["best_fixed"] = "fixed_6"
    assert any("selection differs" in x for x in fz.verify_freeze(g, tmp_path))
    g = json.loads(json.dumps(f)); g["variants"]["r5"]["matched"]["phi"] = 1.0955
    assert any("matched control differs" in x for x in fz.verify_freeze(g, tmp_path))
    g = json.loads(json.dumps(f)); g["variants"]["r5"]["configs"] = ["hdv", "fixed_7", "dyn_a"]
    assert any("configs miss" in x for x in fz.verify_freeze(g, tmp_path))
    (tmp_path / "eval/dev/summary.json").write_text("{}", encoding="utf-8")
    assert fz.verify_freeze(f, tmp_path) == ["md5 differs eval/dev/summary.json"]


def _ring(ids, seconds=80.0, L=260.0):
    n = len(ids)
    t = np.arange(int(seconds * 30)) / 30.0
    rows = []
    for i, vid in enumerate(ids):
        j = (i - 1) % n
        fr = lambda m: L - m * L / n + 5.0 * t + 0.5 * np.sin(0.3 * t + m)
        rows.append(pd.DataFrame({"exp": "A", "veh_id": vid, "regime": "I", "k": np.arange(len(t)), "t": t, "x": fr(i),
                                  "v": 5.0 + 0.15 * np.cos(0.3 * t + i), "a": 0.0, "length": 4.5,
                                  "leader_id": ids[j], "gap": np.mod(fr(j) - fr(i) - 4.5, L)}))
    return pd.concat(rows, ignore_index=True)


def test_episode_rows_carry_chain_mean_speed():
    table = pd.DataFrame([dict(fold="loeo_A", regime="I", variant="nodelay", veh=v, **P) for v in ("1", "2", "pooled")])
    ctx = {"data": {"A": _ring([1, 2, 3, 4])}, "table": table,
           "noise": pd.DataFrame([{"fold": "loeo_A", "regime": "I", "noise_sd": 0.3}]), "thresholds": THR,
           "policy": {"I": {"s0": 3.51, "h": 1.35, "sigma_gap": 2.76, "sigma_dv": 0.55, "sigma_a": 0.29, "sigma_u": 0.73}},
           "t_start": 30.0, "folds": {"A": "loeo_A"}}
    res = runner.run_task({"config": cf.HDV, "exp": "A"}, ctx, {"seeds": 1, "n_rear": 8})
    for r in res["rows"]:
        assert 3.0 < r["chain_mean_speed"] < 7.0


def _rows(dyn_sigma=(0.9, 0.8, 0.7), dyn_speed=1.0, collide_exp=None, dyn_gap_std=2.0):
    """3 experiments x 2 AV positions x 2 seeds; hdv sigma 1 at every position."""
    spec = {"hdv": ("hdv", (0,), (1.0, 1.0, 1.0), 1.0, 1.0), "fixed_7": ("mpc_fixed", (7,), (0.95,) * 3, 1.0, 1.5),
            "dyn_a": ("mpc_dynamic", (40, 3.5), dyn_sigma, dyn_speed, dyn_gap_std),
            "fixed_match": ("matched", (0,), (0.9,) * 3, 1.0, 1.8)}
    out = []
    for c, (fam, rank, sig, spd, gstd) in spec.items():
        for e in ("B", "D", "H"):
            for av in (1, 2):
                for seed in (0, 1):
                    out.append({"config": c, "family": fam, "rank": rank, "exp": e, "regime": "I", "av": av, "seed": seed,
                                "follower_sigma": list(sig), "collision": c == "dyn_a" and e == collide_exp,
                                "min_gap": 5.0, "speed_ratio": 1.0, "mean_gap": 10.0, "av_gap_std": gstd,
                                "chain_mean_speed": 5.0 * spd, "safety_share": 0.0, "clip_share": 0.0,
                                "commit": "c", "dirty": False, "variant": "r5"})
    return pd.DataFrame(out)


FZ = {"test_experiments": ["B", "D", "H"],
      "variants": {"r5": {"best_fixed": "fixed_7", "dynamic": "dyn_a", "configs": ["hdv", "fixed_7", "dyn_a", "fixed_match"]},
                   "primary": {"best_fixed": "fixed_7", "dynamic": None, "configs": ["hdv", "fixed_7"]}}}


def test_decision_rules_supported_case():
    s = summary.build_test_summary(_rows(), THR, FZ, "r5", CONST)
    d = np.log(np.mean([0.9, 0.8, 0.7])) - np.log(0.95)
    assert s["h1"]["d"]["B"] == pytest.approx(d) and s["h1"]["supported"]
    assert s["matched_rule"]["d"]["H"] == pytest.approx(np.log(0.8) - np.log(0.9)) and s["matched_rule"]["supported"]
    h2 = s["h2"]["per_experiment"]["D"]
    assert h2["position_log_ratio"] == pytest.approx(list(np.log([0.9, 0.8, 0.7]))) and h2["slope"] < 0
    assert s["h2"]["supported"] and not s["h2"]["smoother_at_efficiency_cost"]
    assert s["h1"]["t95"][0] == pytest.approx(d) and s["h1"]["t95"][1] == pytest.approx(d)   # identical d: zero width


def test_decision_rules_failures():
    s = summary.build_test_summary(_rows(dyn_speed=0.95), THR, FZ, "r5", CONST)
    assert s["h2"]["smoother_at_efficiency_cost"] and not s["h2"]["supported"]
    s = summary.build_test_summary(_rows(collide_exp="D"), THR, FZ, "r5", CONST)
    assert not s["h1"]["dynamic_guardrails_all_experiments"] and not s["h1"]["supported"]
    assert not s["matched_rule"]["supported"]
    s = summary.build_test_summary(_rows(dyn_sigma=(0.7, 0.8, 1.05)), THR, FZ, "r5", CONST)
    assert not s["h2"]["followers_smoother"]                 # one position above 1 and rising ratios
    s = summary.build_test_summary(_rows(dyn_gap_std=1.0), THR, FZ, "r5", CONST)
    assert not s["h2"]["av_absorbs"]
    s = summary.build_test_summary(_rows(dyn_sigma=(0.94, 0.93, 0.92)), THR, FZ, "r5", CONST)   # better, but by < 5 %
    assert s["h1"]["all_negative"] and not s["h1"]["mean_at_most_threshold"] and not s["h1"]["supported"]


def test_variant_without_dynamic_selection_examines_no_hypothesis():
    df = _rows()
    df = df[df["config"].isin(["hdv", "fixed_7"])]
    s = summary.build_test_summary(df, THR, FZ, "primary", CONST)
    assert s["h1"] is None and s["h2"] is None and s["matched_rule"] is None and "not examined" in s["note"]
    assert s["experiment_log_for"]["fixed_7|B"] == pytest.approx(np.log(0.95))


def test_paired_t_interval_and_plan_constants():
    m, lo, hi = summary.paired_t_interval([-0.1, -0.05, -0.03])
    half = 4.302652729911275 * np.std([-0.1, -0.05, -0.03], ddof=1) / math.sqrt(3)
    assert m == pytest.approx(-0.06) and lo == pytest.approx(m - half) and hi == pytest.approx(m + half)
    c = summary.plan_constants()
    assert c["min_for_reduction"] == 0.05 and c["chain_speed_noninferiority"] == 0.98
