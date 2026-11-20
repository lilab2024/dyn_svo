"""MPC development macros of Section 6 on synthetic summaries."""
import copy
import importlib.util
import math
from pathlib import Path

from dyn_svo.config import load_config

ROOT = Path(__file__).resolve().parents[1]
DEV = ["A", "C", "E", "F", "G"]
DYN = "dyn_W40_lo4_hi7p5"
PHI = load_config("test_freeze")["variants"]["r5"]["matched"]["phi"]


def _pm():
    spec = importlib.util.spec_from_file_location("paper_macros_dev", ROOT / "scripts" / "paper_macros.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pm = _pm()


def _row(family, e, chosen, log_for, g1b=0.0):
    return {"family": family, "held_out": e, "chosen": chosen, "log_for": log_for, "eligible_on_held_out": g1b == 0.0,
            "collisions": 0.0, "fail_rate_g1b": g1b, "fail_rate_g2": 0.0, "fail_rate_g3": 0.0}


def _summary(variant, dyn, fix, dyn_g1b=None, with_variant=True):
    cv = []
    for i, e in enumerate(DEV):
        cv.append(_row("mpc_dynamic", e, dyn[0] if dyn[1][i] is not None else None,
                       dyn[1][i] if dyn[1][i] is not None else float("nan"), (dyn_g1b or {}).get(e, 0.0)))
        cv.append(_row("mpc_fixed", e, fix[0], fix[1][i]))
        cv.append(_row("fs", e, None, float("nan")))
    d = {"commits": ["c" * 40], "any_dirty": False, "cross_validation": cv,
         "final_selection": {"mpc_dynamic": dyn[0], "mpc_fixed": fix[0]}, "experiment_log_for": {}}
    if with_variant:
        d["variant"] = variant
    return d


def _docs():
    none5 = [None] * 5
    r5v5_dyn = [-0.13, -0.06, -0.04, 0.02, -0.01]
    docs = {"primary_v2b": _summary("primary", ("dyn_K2_tau2", none5), ("fixed_0", [-0.04] * 5), with_variant=False),
            "r5_v1": _summary("r5", ("dyn_K8_tau2", [-0.19, -0.16, -0.06, -0.01, -0.02]), ("fixed_7", [-0.16, -0.14, -0.05, 0.0, -0.01])),
            "primary_v5": _summary("primary", (DYN, [None, None, None, None, -0.001]), ("fixed_0", [-0.04] * 5), dyn_g1b={"G": 0.4762}),
            "r5_v5": _summary("r5", (DYN, r5v5_dyn), ("fixed_7", [-0.16, -0.14, -0.05, 0.005, -0.005]))}
    docs["primary_v5"]["final_selection"]["mpc_dynamic"] = None
    docs["r5_v5"]["experiment_log_for"] = {DYN + "|" + e: x for e, x in zip(DEV, r5v5_dyn)}
    matched = [-0.07, -0.07, -0.01, 0.03, 0.02]
    pe = {e: {"log_for_dynamic": x, "log_for_matched": y, "d_match": x - y} for e, x, y in zip(DEV, r5v5_dyn, matched)}
    ds = [pe[e]["d_match"] for e in DEV]
    docs["match"] = {"source_run": "eval_dev_r5_v5", "variant": "r5", "dynamic": DYN, "phi_match": PHI,
                     "n_episodes_for_phi": 1010, "per_experiment": pe, "mean_d_match": sum(ds) / 5,
                     "n_negative": sum(x < 0 for x in ds), "provenance": {"dirty": False}}
    return docs


def test_paper_sources_list_the_dev_mpc_files():
    src = load_config("paper_sources")["dev_mpc"]
    assert set(src) == set(pm.DEV_MPC_KEYS)
    assert src["primary_v2b"] == {"path": "eval/eval_dev_v2b/summary.json", "md5": "40151858f951551e93bc5bca3493adac"}
    assert src["r5_v1"] == {"path": "eval/eval_dev_r5_v1/summary.json", "md5": "e10ed8062582a28595f1809b4a05965b"}
    frozen = load_config("test_freeze")["dev_sources"]
    for k in ("primary_v5", "r5_v5", "match"):
        assert frozen[src[k]["path"]] == src[k]["md5"]


def test_dev_mpc_macros_values_and_names():
    docs = _docs()
    assert pm.dev_mpc_problems(docs, DEV, PHI) == []
    m = pm.Macros()
    pm.dev_mpc_macros(m, docs, DEV, {k: k for k in pm.DEV_MPC_KEYS})
    v = {it["name"]: it["value"] for it in m.items}
    assert v["DevFolds"] == "5" and v["DevPriOldDynFolds"] == "0"
    assert v["DevPriNewDynFolds"] == "1" and v["DevPriNewDynExp"] == "G" and v["DevPriNewGOneB"] == "47.6"
    d_old = [-0.03, -0.02, -0.01, -0.01, -0.01]
    assert v["DevRfiveOldLower"] == "5" and v["DevRfiveOldMeanD"] == "%.3f" % (sum(d_old) / 5)
    assert v["DevRfiveOldChange"] == pm.pct(1 - math.exp(sum(d_old) / 5))
    d_new = [0.03, 0.08, 0.01, 0.015, -0.005]
    assert v["DevRfiveNewHigher"] == "4" and v["DevRfiveNewMeanD"] == "%.3f" % (sum(d_new) / 5)
    assert v["DevRfiveNewChange"] == pm.pct(math.exp(sum(d_new) / 5) - 1)
    assert v["DevMatchNeg"] == "4" and v["DevMatchPhi"] == "5.58" and v["DevMatchEpisodes"] == "1010"
    assert v["DevMatchRed"] == pm.pct(1 - math.exp(docs["match"]["mean_d_match"]))
    pm.config_macros(m)                                  # Macros.add refuses a name used by another builder
    pm.method_macros(m)
    pm.plan_macros(m, ROOT / "config" / "decision_constants.yaml")
    assert all(i["name"].isalpha() for i in m.items) and len({i["name"] for i in m.items}) == len(m.items)


def test_dev_mpc_problems_each_reason_alone():
    good = _docs()

    def check(mutate, text, alone=True):
        bad = copy.deepcopy(good)
        mutate(bad)
        probs = pm.dev_mpc_problems(bad, DEV, PHI)
        assert any(text in p for p in probs) and (len(probs) == 1 or not alone), (text, probs)

    check(lambda d: d.pop("r5_v1"), "r5_v1: file missing")
    check(lambda d: d["r5_v5"].update(any_dirty=True), "r5_v5: commits or dirty flag")
    check(lambda d: d["r5_v5"].update(commits=["a", "b"]), "r5_v5: commits or dirty flag")
    check(lambda d: d["primary_v5"].update(variant="r5"), "primary_v5: variant r5")
    check(lambda d: d["r5_v1"]["cross_validation"].pop(0), "r5_v1: mpc_dynamic folds differ")
    check(lambda d: d["r5_v1"]["cross_validation"][3].update(chosen=None), "r5_v1: mpc_dynamic has no selection in fold C")
    check(lambda d: d["match"].update(source_run="eval_dev_r5_v1"), "match: source_run or variant")
    check(lambda d: d["match"]["provenance"].update(dirty=True), "match: written from a dirty tree")
    check(lambda d: d["match"].update(dynamic="dyn_W10_lo3_hi7"), "match: dynamic configuration differs", alone=False)
    check(lambda d: d["match"]["per_experiment"].pop("G"), "match: experiments differ")
    check(lambda d: d["match"]["per_experiment"]["E"].update(log_for_dynamic=-0.0398, log_for_matched=-0.0098),
          "match: dynamic log FOR differs from eval_dev_r5_v5 in E")
    check(lambda d: d["match"]["per_experiment"]["F"].update(log_for_matched=0.0301), "match: d_match inconsistent in F")
    check(lambda d: d["match"].update(mean_d_match=0.0), "match: mean_d_match inconsistent")
    check(lambda d: d["match"].update(n_negative=5), "match: n_negative inconsistent")
    check(lambda d: d["match"].update(phi_match=1.1), "match: phi_match differs")
