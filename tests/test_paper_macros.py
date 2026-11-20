"""scripts/paper_macros.py."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BS = chr(92)

def _mod():
    path = Path(__file__).resolve().parents[1] / "scripts" / "paper_macros.py"
    spec = importlib.util.spec_from_file_location("paper_macros_script", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def test_join_list():
    j = _mod().join_list
    assert j([]) == "none" and j(["A"]) == "A" and j(["A", "C"]) == "A and C" and j(["A", "C", "G"]) == "A, C and G"

def test_macro_names_and_tex():
    mod = _mod()
    m = mod.Macros()
    m.add("SimDt", "0.1", "config/sim.yaml", "dt")
    with pytest.raises(ValueError):
        m.add("Sim2", "1", "x", "y")
    with pytest.raises(ValueError):
        m.add("SimDt", "1", "x", "y")
    t = m.tex("abc1234")
    assert BS + "newcommand{" + BS + "SimDt}{0.1}" in t and "abc1234" in t

def test_noise_macros(tmp_path):
    mod = _mod()
    p = tmp_path / "noise.json"
    rows = [{"regime": "I", "noise_sd": v} for v in (0.551, 0.76)] + [{"regime": "II", "noise_sd": v} for v in (1.18, 1.479)]
    p.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    m = mod.Macros()
    mod.noise_macros(m, p, "noise.json")
    got = {i["name"]: i["value"] for i in m.items}
    assert got == {"NoiseSdIMin": "0.55", "NoiseSdIMax": "0.76", "NoiseSdIIMin": "1.18", "NoiseSdIIMax": "1.48"}

def test_chain_macros(tmp_path):
    mod = _mod()
    p = tmp_path / "chain.json"
    rows = [{"exp": "A", "ok_sigma": True, "ok_slope": True, "ok_speed": True, "speed_rel_diff": 0.01},
            {"exp": "E", "ok_sigma": False, "ok_slope": True, "ok_speed": True, "speed_rel_diff": 0.0},
            {"exp": "F", "ok_sigma": True, "ok_slope": True, "ok_speed": False, "speed_rel_diff": 0.0641}]
    p.write_text(json.dumps({"summary": rows}), encoding="utf-8")
    m = mod.Macros()
    mod.chain_macros(m, p, "chain.json", 5)
    got = {i["name"]: i["value"] for i in m.items}
    assert got["ChainAdequate"] == "A" and got["ChainSpeedOnly"] == "F" and got["ChainInadequate"] == "E"
    assert got["ChainSpeedF"] == "6.4" and got["ChainSeeds"] == "5"

def test_data_macros_dev_only(monkeypatch):
    mod = _mod()
    loaded = []

    def fake_load(e):
        loaded.append(e)
        k = np.arange(0, 30 * 100)
        n = 3 if e == "A" else 4
        return pd.concat([pd.DataFrame({"veh_id": v, "k": k, "t": k / 30.0, "a": np.sin(k / 10.0 + v),
                                        "follower_ok": True}) for v in range(n)], ignore_index=True)

    monkeypatch.setattr(mod.sp, "experiments", lambda role: ["A", "F"])
    monkeypatch.setattr(mod.ld, "load_experiment", fake_load)
    split_cfg = {"instruction_regime": {"A": "I", "F": "II"}, "segments": {"drop_first_s": 30}}
    m = mod.Macros()
    rows = mod.data_macros(m, split_cfg, 60.0)
    got = {i["name"]: i["value"] for i in m.items}
    assert loaded == ["A", "F"] and got["DevVehMin"] == "3" and got["DevVehMax"] == "4"
    assert rows[0]["usable_s"] == pytest.approx(2999 / 30.0 - 30.0)
    assert float(got["RealAccSdI"]) > 0 and float(got["RealAccSdII"]) > 0

def test_experiments_table_locks_test_rows():
    mod = _mod()
    split_cfg = {"experiments": ["A", "B"], "roles": {"dev": ["A"], "test": ["B"]}, "instruction_regime": {"A": "I", "B": "I"}}
    t = mod.experiments_table([{"exp": "A", "regime": "I", "vehicles": 20, "usable_s": 600.4}], split_cfg)
    assert "A & I & dev & 20 & 600 " in t and "B & I & test & locked & locked " in t and "tab:experiments" in t

def test_method_macros_from_repo_config():
    mod = _mod()
    m = mod.Macros()
    mod.method_macros(m)
    got = {i["name"]: i["value"] for i in m.items}
    assert got["FilterConv"] == "0.2" and got["MpcMoves"] == "20" and got["RlSeeds"] == "10"
    assert got["RlFixedPhi"] == "0, 2, 4, 6 and 8" and got["RlTuneCandidates"] == "6" and got["RlDynCount"] == "5"
    assert got["ActWindows"] == "10, 20 and 40" and got["PhiRanges"] == "(3, 7), (4, 7.5) and (5, 8)"
    assert all(i["name"].isalpha() for i in m.items)

def test_result_method_macros(tmp_path):
    mod = _mod()
    g, p = tmp_path / "g.json", tmp_path / "p.json"
    thr = {r: {"min_gap": v, "speed_ratio": 0.98887, "mean_gap": 13.6886, "n_vehicle_windows": 61}
           for r, v in (("I", 1.4946), ("II", 0.8008))}
    pol = {r: {"s0": 3.5108, "h": 1.347, "sigma_gap": 2.7606, "sigma_dv": 0.5474, "sigma_a": 0.2876, "sigma_u": 0.7323}
           for r in ("I", "II")}
    g.write_text(json.dumps({"thresholds": thr}), encoding="utf-8")
    p.write_text(json.dumps({"policy": pol}), encoding="utf-8")
    m = mod.Macros()
    mod.result_method_macros(m, g, "g.json", p, "p.json")
    got = {i["name"]: i["value"] for i in m.items}
    assert got["GuardMinGapI"] == "1.49" and got["FilterFloorI"] == "1.54" and got["FilterFloorII"] == "0.85"
    assert got["GuardSpeedRatioI"] == "0.989" and got["PolSigAII"] == "0.29" and got["GuardWindowsI"] == "61"

def test_config_macros_from_repo_config():
    mod = _mod()
    m = mod.Macros()
    mod.config_macros(m)
    got = {i["name"]: i["value"] for i in m.items}
    assert got["SimDt"] == "0.1" and got["MetricStart"] == "60" and got["DelayMax"] == "1.2"

def test_all_macro_builders_together_unique_names(tmp_path):
    mod = _mod()
    m = mod.Macros()
    mod.config_macros(m)
    mod.method_macros(m)
    mod.plan_macros(m, Path(__file__).resolve().parents[1] / "config" / "decision_constants.yaml")
    assert len({i["name"] for i in m.items}) == len(m.items) > 60

def test_plan_macros():
    mod = _mod()
    m = mod.Macros()
    mod.plan_macros(m, Path(__file__).resolve().parents[1] / "config" / "decision_constants.yaml")
    got = {i["name"]: i["value"] for i in m.items}
    assert got["MinForReduction"] == "5" and got["LnThreshold"] == "-0.0513" and got["ChainSpeedNoninf"] == "98"
    assert int(got["PlanVersion"]) >= 6


# ---- RL development results ----

NL = chr(10)
RL_SYN = {"fixed_phi_pi16": [0, 8], "training_seeds": [0, 1], "training_steps": 1000000,
          "dynamic_configs": [{"window_s": 40, "phi_lo_pi16": 4, "phi_hi_pi16": 7.5},
                              {"window_s": 20, "phi_lo_pi16": 3, "phi_hi_pi16": 7}],
          "selected_hp": {"learning_rate": 0.0003, "gamma": 0.98}}
DEV_SYN = ["A", "G"]
CFG_SYN = ["rl_fixed_0", "rl_fixed_8", "rl_dyn_W20_lo3_hi7", "rl_dyn_W40_lo4_hi7p5"]


def _rates(g1b, g2=0.01, g3=0.02, n=40):
    return {"n_runs": n, "collisions": 0, "fail_rate_g1b": g1b, "fail_rate_g2": g2, "fail_rate_g3": g3, "eligible": False}


def _rl_docs():
    g1b_all, g1b_g, g2_pass = [0.05, 0.0, 0.10, 0.15], [0.10, 0.0, 0.20, 0.30], [0.004, 0.5, 0.006, 0.0]
    docs = {}
    for tag, variant, run in (("pri", "primary", "eval_dev_rl_v1"), ("r5", "r5", "eval_dev_r5_rl_v1")):
        lf = {"hdv|" + e: 0.0 for e in DEV_SYN}
        conf = {}
        for i, c in enumerate(CFG_SYN):
            for e in DEV_SYN:
                lf[c + "|" + e] = -0.1 * (i + 1) - (0.2 if e == "G" else 0.0) + (0.001 * i if tag == "r5" else 0.0)
            f = 1.0 if tag == "pri" else 0.0
            conf[c] = {"all_dev": _rates(f * g1b_all[i]),
                       "per_experiment": {"A": _rates(0.0, n=20), "G": _rates(f * g1b_g[i], n=20)},
                       "by_g4": {"g4_pass": dict(_rates(0.0, g2=g2_pass[i], n=40), n_policies=4),
                                 "g4_fail": {"n_runs": 0, "collisions": 0, "fail_rate_g1b": None, "fail_rate_g2": None,
                                             "fail_rate_g3": None, "n_policies": 0}}}
        docs["eval_" + tag] = {"commits": ["abc"], "any_dirty": False, "summary_provenance": {"dirty": False},
                               "variant": variant, "variants": [variant], "run_name": run, "train_run": "rl_train_v1",
                               "experiment_log_for": lf, "seed_sd_dev_mean": {c: {"sd": 0.05, "n_seeds": 2} for c in CFG_SYN}}
        docs["elig_" + tag] = {"run_name": run, "variant": variant, "train_run": "rl_train_v1", "g4_run": "g4_rl_train_v1",
                               "eval_commits": ["abc"], "report_provenance": {"dirty": False}, "configs": conf}
    fails = []
    for kind, k in (("gap", 3), ("ratiogap", 2), ("all", 1)):
        for _ in range(k):
            fails.append({"config": "rl_fixed_8", "exp": "A", "seed": 0, "gap_limit": 100.0,
                          "max_last_gap": 150.0, "mean_speed_ratio": 0.9 if kind != "gap" else 0.99,
                          "reward_not_below_zero_av": kind != "all"})
    docs["g4"] = {"stage": "train", "n_policies": 16, "n_pass": 10, "pass_share_all": 10 / 16.0,
                  "pass_counts": {c: [k, 4] for c, k in zip(CFG_SYN, (4, 1, 3, 2))}, "not_passing": fails}
    docs["gate"] = {"selected": "rl_tune_lr0.0003_g0.98", "n_selected": 15, "selected_pass": 15, "gate_pass": True,
                    "pass_counts": {"rl_tune_lr0.0003_g0.98": [15, 15], "rl_tune_lr0.0001_g0.99": [10, 15]},
                    "pass_share_all": 25 / 30.0}
    return docs


def test_rl_result_macros_values_and_rounding():
    mod = _mod()
    docs = _rl_docs()
    assert mod.rl_problems(docs, RL_SYN, DEV_SYN) == []
    m = mod.Macros()
    mod.rl_result_macros(m, docs, RL_SYN, DEV_SYN)
    got = {i["name"]: i["value"] for i in m.items}
    want = {"RlGatePass": "15", "RlGateN": "15", "RlGateAllPass": "25", "RlGateAllN": "30", "RlGateAllPct": "83.3",
            "RlGFourPass": "10", "RlGFourN": "16", "RlGFourPct": "62.5", "RlGFourPerConfig": "4", "RlTopFixedPhi": "8",
            "RlGFourTopFixedPass": "1", "RlGFourExTopFixedPct": "75.0", "RlGFourFailGapOnly": "3",
            "RlGFourFailRatioGap": "2", "RlGFourFailAll": "1", "RlGFourFailOther": "0", "RlRunsPerConfig": "40",
            "RlCollisions": "0", "RlEligiblePri": "0", "RlEligibleRfive": "0", "RlGOneBMinPri": "5.0",
            "RlGOneBMaxPri": "15.0", "RlGOneBHighExp": "G", "RlGOneBHighMin": "10.0", "RlGOneBHighMax": "30.0",
            "RlGOneBLowExp": "A", "RlGOneBLowMax": "0.0", "RlGOneBMaxRfive": "0.0", "RlGTwoPassMaxPri": "0.6",
            "RlForDiffMax": "0.003"}
    assert got == want
    m2 = mod.Macros()
    mod.rl_method_macros(m2, RL_SYN, "torch==2.9.1" + NL + "stable-baselines3==2.7.1" + NL + "gymnasium==1.2.2" + NL, "lock")
    got2 = {i["name"]: i["value"] for i in m2.items}
    assert got2["RlTrainSteps"] == "1" + BS + ",000" + BS + ",000" and got2["RlLrSel"] == "0.0003" and got2["RlGammaSel"] == "0.98"
    assert (got2["TorchVersion"], got2["SbVersion"], got2["GymVersion"]) == ("2.9.1", "2.7.1", "1.2.2")


def test_rl_problems_each_reason_alone():
    mod = _mod()
    cases = [
        (lambda d: (d["eval_pri"].update(commits=["a", "b"]), d["elig_pri"].update(eval_commits=["a", "b"])), "tasks from 2 commits"),
        (lambda d: d["eval_pri"].update(any_dirty=True), "tasks from a dirty tree"),
        (lambda d: d["eval_r5"]["summary_provenance"].update(dirty=True), "summary written from a dirty tree"),
        (lambda d: d["eval_pri"].update(variant="r5"), "variant is not primary"),
        (lambda d: d["eval_r5"].update(run_name="x"), "run_name is not eval_dev_r5_rl_v1"),
        (lambda d: d["eval_pri"].update(train_run="x"), "train_run is not rl_train_v1"),
        (lambda d: d["eval_pri"]["experiment_log_for"].update({"rl_fixed_4|A": 0.0}), "summary.json: configurations differ"),
        (lambda d: d["eval_pri"]["experiment_log_for"].update({"rl_fixed_0|Z": 0.0}), "experiments differ"),
        (lambda d: d["elig_pri"].update(run_name="x"), "eligibility.json: run_name is not"),
        (lambda d: d["elig_r5"].update(g4_run="x"), "g4_run is not g4_rl_train_v1"),
        (lambda d: d["elig_pri"].update(eval_commits=["zzz"]), "eval_commits differ"),
        (lambda d: d["elig_pri"]["report_provenance"].update(dirty=True), "report written from a dirty tree"),
        (lambda d: d["elig_r5"]["configs"].pop("rl_fixed_0"), "eligibility.json: configurations differ"),
        (lambda d: d["g4"].update(stage="tune"), "stage is not train"),
        (lambda d: d["g4"].update(n_pass=11), "pass counts are inconsistent"),
        (lambda d: d["g4"].update(pass_share_all=0.5), "pass_share_all is inconsistent"),
        (lambda d: d["gate"].update(selected="rl_tune_lr0.0001_g0.99"), "selected candidate is not"),
        (lambda d: d["gate"].update(gate_pass=False), "the gate did not pass"),
    ]
    for change, text in cases:
        docs = json.loads(json.dumps(_rl_docs()))
        change(docs)
        probs = mod.rl_problems(docs, RL_SYN, DEV_SYN)
        assert len(probs) == 1 and text in probs[0], (text, probs)
    docs = _rl_docs()
    docs["g4"]["pass_counts"]["rl_fixed_0"] = [4, 5]
    assert any("number of policies" in p for p in mod.rl_problems(docs, RL_SYN, DEV_SYN))


def test_rl_tables_order_and_format():
    mod = _mod()
    docs = _rl_docs()
    t = mod.rl_for_table(docs, RL_SYN, DEV_SYN)
    rows = [ln for ln in t.split(NL) if ln.startswith("fixed") or ln.startswith("dyn.")]
    assert [r.split(" & ")[0] for r in rows] == ["fixed, $" + BS + "phi=0$", "fixed, $" + BS + "phi=8$",
                                                  "dyn., $W_a=20$, $(3, 7)$", "dyn., $W_a=40$, $(4, 7.5)$"]
    assert rows[0].split(" & ")[1:] == ["$-0.100$", "$-0.300$", "$-0.200$", "0.819", "0.050", "$-0.200$ " + BS + BS]
    assert BS + "label{tab:rl-dev-for}" in t and t.startswith(BS + "begin{table*}") and BS + "RlForDiffMax" in t
    g = mod.rl_guardrail_table(docs, RL_SYN, DEV_SYN)
    grow = [ln for ln in g.split(NL) if ln.startswith("dyn., $W_a=40$")][0]
    assert grow.split(" & ")[1:] == ["2", "15.0", "1.0", "2.0", "0.0", "1.0", "2.0 " + BS + BS]
    s = mod.rl_g4split_table(docs, RL_SYN, DEV_SYN)
    srow = [ln for ln in s.split(NL) if ln.startswith("fixed, $" + BS + "phi=8$")][0]
    assert srow.split(" & ")[1:] == ["4", "0.0", "50.0", "2.0", "0", "--", "--", "-- " + BS + BS]
    assert s.startswith(BS + "begin{table}[") and BS + "label{tab:rl-dev-g4split}" in s


def test_requirement_versions_num_and_pep503():
    mod = _mod()
    text = NL.join(["# lock ", "torch==2.9.1", "Stable_Baselines3==2.7.1  # pinned", "", "gymnasium==1.2.2", ""])
    assert mod.requirement_versions(text) == {"torch": "2.9.1", "stable-baselines3": "2.7.1", "gymnasium": "1.2.2"}
    assert mod.pep503("Farama.Notifications") == "farama-notifications"
    with pytest.raises(ValueError):
        mod.requirement_versions("torch>=2.0")
    assert mod.num(9999) == "9999" and mod.num(10100) == "10" + BS + ",100" and mod.num(0) == "0"
    assert mod.pct(0.1505) in ("15.0", "15.1") and mod.pct(0.0) == "0.0"


def test_rl_configs_match_repo_and_all_builders_unique():
    from dyn_svo.config import load_config
    from dyn_svo.rl import summary as RS
    mod = _mod()
    rl = load_config("rl")
    names = [n for n, _ in mod.rl_configs(rl)]
    assert sorted(names) == sorted(k for k in RS.config_meta(rl) if k != "hdv")
    assert names[:len(rl["fixed_phi_pi16"])] == [mod.rlr.fixed_name(p) for p in sorted(rl["fixed_phi_pi16"])]
    m = mod.Macros()
    mod.config_macros(m)
    mod.method_macros(m)
    mod.plan_macros(m, Path(__file__).resolve().parents[1] / "config" / "decision_constants.yaml")
    req = Path(__file__).resolve().parents[1] / "config" / "requirements_rl.txt"
    mod.rl_method_macros(m, rl, req.read_text(encoding="utf-8"), "config/requirements_rl.txt")
    mod.rl_result_macros(m, _rl_docs(), RL_SYN, DEV_SYN)
    assert len({i["name"] for i in m.items}) == len(m.items) > 90


def test_rl_for_table_fits_the_text_width():
    mod = _mod()
    docs = _rl_docs()
    t = mod.rl_for_table(docs, RL_SYN, DEV_SYN).split(NL)
    assert t[2] == BS + "scriptsize" and t[5] == BS + "setlength{" + BS + "tabcolsep}{2.5pt}"
    g = mod.rl_guardrail_table(docs, RL_SYN, DEV_SYN).split(NL)
    assert g[2] == BS + "footnotesize"
