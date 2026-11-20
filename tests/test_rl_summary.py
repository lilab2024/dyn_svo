"""RL development summary: run checks, families and ranks, selection with tie rules, seed spread,
seed share, nested LOEO d and phi_match, on synthetic task records."""
import math

import numpy as np
import pytest

from dyn_svo.config import load_config
from dyn_svo.experiments import summary as S
from dyn_svo.rl import evaluate as V
from dyn_svo.rl import summary as RS

DEV = ["A", "C", "E"]
THR = {"I": {"min_gap": 1.0, "speed_ratio": 0.9, "mean_gap": 30.0}}
BASE = {"rl_fixed_0": 0.0, "rl_fixed_2": -0.02, "rl_fixed_4": -0.04, "rl_fixed_6": -0.06, "rl_fixed_8": -0.2,
        "rl_dyn_W10_lo4_hi7p5": -0.05, "rl_dyn_W20_lo4_hi7p5": -0.08, "rl_dyn_W40_lo4_hi7p5": -0.08,
        "rl_dyn_W20_lo3_hi7": -0.0805, "rl_dyn_W20_lo5_hi8": -0.3}
HDV_SIGMA = [1.0, 2.0]


def _row(av, seed, factor, **kw):
    row = {"av": av, "seed": seed, "follower_sigma": [x * math.exp(factor) for x in HDV_SIGMA], "collision": False,
           "min_gap": 5.0, "speed_ratio": 1.0, "mean_gap": 10.0, "safety_share": 0.01, "clip_share": 0.0}
    row.update(kw)
    return row


def _records(variant="primary"):
    rl = load_config("rl")
    tasks = V.eval_tasks("train", DEV, "full", variant, rl=rl)
    recs = []
    for t in tasks:
        rows = []
        for av in (1, 2):
            for seed in (0, 1):
                if t["config"] == "hdv":
                    rows.append(_row(av, seed, 0.0))
                    continue
                c, s = t["config"], t["train_seed"]
                dyn = c.startswith("rl_dyn")
                f = BASE[c] + (0.006 * (s - 4.5) if dyn else 0.0)
                kw = {"train_seed": s, "phi_mean": 1.0 + 0.01 * s if dyn else 0.1 * len(c)}
                if c == "rl_fixed_8":
                    kw["mean_gap"] = 50.0                       # fails G3 everywhere
                if c == "rl_dyn_W20_lo5_hi8" and av == 1 and seed == 0 and s == 3:
                    kw["collision"] = True                     # one collision makes it ineligible
                rows.append(_row(av, seed, f, **kw))
        res = {"exp": t["exp"], "regime": "I", "variant": variant, "mode": "full", "rows": rows}
        if t["config"] != "hdv":
            res["train_commit"] = "ed63a09"
        recs.append({"task": t, "result": res, "provenance": {"commit": "c0ffee1", "dirty": False,
                                                                "config_md5": {"train_run": "rl_train_v1", "eval_mode": "full"}}})
    return tasks, recs


def test_config_meta_families_and_ranks():
    meta = RS.config_meta()
    assert len(meta) == 11 and meta["hdv"]["family"] == "hdv"
    assert meta["rl_fixed_6"] == {"family": "rl_fixed", "rank": [6.0]}
    assert meta["rl_dyn_W20_lo4_hi7p5"] == {"family": "rl_dynamic", "rank": [20.0, 3.5]}
    assert meta["rl_dyn_W20_lo5_hi8"]["rank"] == [20.0, 3.0]


def test_run_problems_reports_each_defect():
    tasks, recs = _records()
    assert len(tasks) == 303 and [t["config"] for t in tasks[:3]] == ["hdv"] * 3
    assert RS.run_problems(recs, tasks, "primary", 2) == []
    bad = [dict(r) for r in recs[1:]]
    bad[0] = dict(bad[0], provenance=dict(bad[0]["provenance"], commit="beef000", dirty=True))
    bad[5] = dict(bad[5], result=dict(bad[5]["result"], mode="criterion"))
    bad[7] = dict(bad[7], result=dict(bad[7]["result"], rows=[dict(x, train_seed=99) for x in bad[7]["result"]["rows"]]))
    probs = RS.run_problems(bad, tasks, "primary", 2)
    assert len(probs) == 5
    assert probs[0].startswith("missing tasks: 1") and probs[1].startswith("commits:") and probs[2].startswith("dirty")
    assert probs[3].startswith("mode:") and "another training seed" in probs[4]
    assert RS.run_problems(recs, tasks, "r5", 2)[0].startswith("variant:")
    assert "noise seeds" in RS.run_problems(recs, tasks, "primary", 3)[0]


def test_summary_selection_ties_and_extra_items():
    tasks, recs = _records()
    df = RS.load_rows(recs)
    out = RS.build_rl_summary(df, THR, DEV)
    assert out["final_selection"] == {"rl_dynamic": "rl_dyn_W20_lo4_hi7p5", "rl_fixed": "rl_fixed_6"}
    assert out["n_rows"] == 1212 and out["n_runs"]["rl_fixed_0|A"] == 40
    vals = S.experiment_values(S.add_log_for(df))
    sv = RS.seed_values(S.add_log_for(df))
    for c in BASE:
        for e in DEV:
            assert sv.loc[c, e].mean() == pytest.approx(vals[(c, e)], abs=1e-12)
            assert vals[(c, e)] == pytest.approx(BASE[c], abs=1e-12)
    sd = out["seed_sd"]["rl_dyn_W20_lo4_hi7p5|A"]
    assert sd["n_seeds"] == 10 and sd["sd"] == pytest.approx(0.006 * np.std(np.arange(10), ddof=1), rel=1e-9)
    assert out["seed_sd"]["rl_fixed_6|A"]["sd"] == pytest.approx(0.0, abs=1e-12)
    assert out["seed_sd_dev_mean"]["rl_dyn_W20_lo4_hi7p5"]["n_seeds"] == 10
    sh = out["seed_share"]
    assert sh["n_negative"] == 8 and sh["n_seeds"] == 10 and sh["share_negative"] == 0.8
    assert sh["per_experiment_negative"] == {"A": [8, 10], "C": [8, 10], "E": [8, 10]}
    cv = out["cv_d"]
    assert cv["n_folds"] == 3 and cv["n_negative"] == 3 and cv["mean_d"] == pytest.approx(-0.02, abs=1e-12)
    assert all(v["dynamic_eligible_on_held_out"] for v in cv["per_fold"].values())
    assert out["final_d"]["mean_d"] == pytest.approx(-0.02, abs=1e-12)
    assert out["phi_match"] == {"config": "rl_dyn_W20_lo4_hi7p5", "phi_match": 1.045,
                                "phi_match_pi16": pytest.approx(1.045 / (math.pi / 16)), "n_runs": 120}
    assert set(out["platform_shares"]) == set(BASE)


def test_summary_without_dynamic_selection():
    tasks, recs = _records()
    for r in recs:
        if r["task"]["config"].startswith("rl_dyn"):
            r["result"]["rows"] = [dict(x, collision=True) for x in r["result"]["rows"]]
    out = RS.build_rl_summary(RS.load_rows(recs), THR, DEV)
    assert out["final_selection"]["rl_dynamic"] is None and out["final_selection"]["rl_fixed"] == "rl_fixed_6"
    assert "not_computed" in out["phi_match"] and "not_computed" in out["seed_share"] and "not_computed" in out["final_d"]
    assert out["cv_d"]["mean_d"] is None and len(out["cv_d"]["not_computed"]) == 3


def test_unknown_configuration_is_refused():
    tasks, recs = _records()
    recs[5]["task"] = dict(recs[5]["task"], config="rl_match_1.0954")
    with pytest.raises(ValueError):
        RS.load_rows(recs)


def _g4(tasks, fail=()):
    out = []
    for t in tasks:
        if t["config"] == "hdv":
            continue
        ok = (t["config"], t["exp"], t["train_seed"]) not in fail
        out.append({"task": {}, "result": {"config": t["config"], "exp": t["exp"], "seed": t["train_seed"], "g4": {"pass": ok}},
                    "provenance": {"commit": "11701ca", "dirty": False, "config_md5": {"stage": "train", "train_run": "rl_train_v1"}}})
    return out


def test_eligibility_report_counts_and_g4_split():
    """ counts and rates per configuration, eligibility only on the sets, G4 groups without eligibility."""
    tasks, recs = _records()
    fail = {("rl_dyn_W20_lo5_hi8", e, 3) for e in DEV} | {("rl_fixed_8", "A", 0)}
    g4 = _g4(tasks, fail)
    assert RS.g4_problems(g4, tasks, "rl_train_v1") == []
    rep = RS.eligibility_report(RS.load_rows(recs), THR, DEV, g4)
    assert rep["fail_rate_max"] == 0.01 and set(rep["configs"]) == set(BASE)
    c = rep["configs"]["rl_dyn_W20_lo5_hi8"]
    assert c["all_dev"] == {"n_runs": 120, "collisions": 3, "fail_rate_g1b": 0.0, "fail_rate_g2": 0.0, "fail_rate_g3": 0.0, "eligible": False}
    assert c["per_experiment"]["A"]["collisions"] == 1 and c["per_experiment"]["A"]["n_runs"] == 40
    assert c["by_g4"]["g4_fail"] == {"n_runs": 12, "collisions": 3, "fail_rate_g1b": 0.0, "fail_rate_g2": 0.0, "fail_rate_g3": 0.0, "n_policies": 3}
    assert c["by_g4"]["g4_pass"]["collisions"] == 0 and c["by_g4"]["g4_pass"]["n_policies"] == 27
    assert "eligible" not in c["by_g4"]["g4_pass"] and "eligible" not in c["by_g4"]["g4_fail"]
    f8 = rep["configs"]["rl_fixed_8"]
    assert f8["all_dev"]["fail_rate_g3"] == 1.0 and f8["all_dev"]["eligible"] is False
    assert f8["by_g4"]["g4_fail"]["n_policies"] == 1 and f8["by_g4"]["g4_fail"]["n_runs"] == 4
    ok = rep["configs"]["rl_fixed_6"]
    assert ok["all_dev"]["eligible"] is True and ok["by_g4"]["g4_fail"] == {"n_runs": 0, "collisions": 0, "fail_rate_g1b": None,
                                                                            "fail_rate_g2": None, "fail_rate_g3": None, "n_policies": 0}


def test_g4_problems_reports_each_defect():
    tasks, recs = _records()
    g4 = _g4(tasks)
    bad = g4[1:]
    bad[0] = dict(bad[0], provenance=dict(bad[0]["provenance"], commit="beef000", dirty=True))
    bad[1] = dict(bad[1], provenance=dict(bad[1]["provenance"], config_md5={"stage": "tune", "train_run": "rl_train_v1"}))
    probs = RS.g4_problems(bad, tasks, "rl_train_v1")
    assert len(probs) == 4 and probs[0].startswith("G4 run lacks 1") and probs[1].startswith("G4 run commits")
    assert probs[2].startswith("G4 run has dirty") and probs[3].startswith("G4 run stage")
    assert RS.g4_problems(g4, tasks, "rl_train_v9")[0].startswith("G4 run train_run")
