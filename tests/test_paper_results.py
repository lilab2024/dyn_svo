"""Test-result macros, tables and figure on synthetic summaries built with the real summary code."""
import copy
import hashlib
import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dyn_svo.config import load_config
from dyn_svo.experiments import summary

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pm = _load("paper_macros")
pf = _load("paper_figures")
SRC = load_config("paper_sources")
THR = {"I": {"min_gap": 1.0, "speed_ratio": 0.9, "mean_gap": 30.0}}
CONST = {"min_for_reduction": 0.05, "chain_speed_noninferiority": 0.98}
DYN = "dyn_W40_lo4_hi7p5"


def _rows(variant):
    spec = {"hdv": ((1.0, 1.0, 1.0), 1.0, False, 30.0),
            "fixed_0": ((0.97,) * 3, 1.2, False, 10.0), "fixed_7": ((0.85, 0.88, 0.9), 1.6, False, 10.0),
            DYN: ((0.9, 0.92, 0.95), 2.0, False, 10.0), "fixed_match": ((0.93,) * 3, 1.8, False, 10.0),
            "fs": ((0.6,) * 3, 2.5, False, 40.0), "pi": ((0.75,) * 3, 2.2, True, 10.0)}
    keep = {"primary": ("hdv", "fixed_0", "fs", "pi"), "r5": ("hdv", "fixed_7", DYN, "fixed_match", "fs", "pi")}[variant]
    out = []
    for c in keep:
        sig, gstd, collide_h, mean_gap = spec[c]
        for e in ("B", "D", "H"):
            for av in (1, 2):
                for seed in (0, 1):
                    out.append({"config": c, "family": "x", "rank": (0,), "exp": e, "regime": "I", "av": av, "seed": seed,
                                "follower_sigma": list(sig), "collision": collide_h and e == "H" and seed == 0,
                                "min_gap": 5.0, "speed_ratio": 1.0, "mean_gap": mean_gap, "av_gap_std": gstd,
                                "chain_mean_speed": 5.0, "safety_share": 0.0, "clip_share": 0.0,
                                "commit": SRC["test_commit"], "dirty": False, "variant": variant})
    return pd.DataFrame(out)


FZ = load_config("test_freeze")


def _docs():
    return {v: summary.build_test_summary(_rows(v), THR, FZ, v, CONST) for v in ("primary", "r5")}


def test_paper_sources_match_the_recorded_runs():
    assert SRC["test_commit"].startswith("43eacc1")
    assert SRC["test_summaries"]["primary"] == {"path": "eval/eval_test_v5/summary.json", "md5": "7864b5220cdf86c68f2ea755972023b3"}
    assert SRC["test_summaries"]["r5"] == {"path": "eval/eval_test_r5_v5/summary.json", "md5": "6113a105b67e0005aa5aa3180ce44a78"}


def test_problems_accept_good_and_refuse_bad_summaries():
    docs = _docs()
    assert pm.test_problems(docs, SRC) == []
    bad = copy.deepcopy(docs); bad["r5"]["any_dirty"] = True
    assert any("dirty" in p for p in pm.test_problems(bad, SRC))
    bad = copy.deepcopy(docs); bad["primary"]["h1"] = {"supported": True}
    assert any("must not be examined" in p for p in pm.test_problems(bad, SRC))
    bad = copy.deepcopy(docs); del bad["r5"]["experiment_log_for"]["fixed_match|D"]
    assert any("missing fixed_match|D" in p for p in pm.test_problems(bad, SRC))
    bad = copy.deepcopy(docs); bad["r5"]["h2"] = None
    assert any("decision rules missing" in p for p in pm.test_problems(bad, SRC))


def test_result_macros_values():
    m = pm.Macros()
    pm.test_result_macros(m, _docs(), {"primary": "p", "r5": "r"})
    v = {it["name"]: it["value"] for it in m.items}
    assert v["TestCommit"] == "43eacc1" and v["TestFixedZeroForB"] == "0.970"
    assert v["TestFixedZeroRedMin"] == "3.0" and v["TestFixedZeroRedMax"] == "3.0"
    d1 = math.log(np.mean([0.9, 0.92, 0.95])) - math.log(np.mean([0.85, 0.88, 0.9]))
    assert v["TestHOneMeanD"] == "%.3f" % d1
    assert v["TestMatchRed"] == pm.pct(1 - math.exp(math.log(np.mean([0.9, 0.92, 0.95])) - math.log(0.93)))
    assert v["TestHTwoFirstB"] == "0.900" and v["TestHTwoLastD"] == "0.950" and v["TestHTwoHPositions"] == "3"
    assert v["TestHTwoHAbove"] == "0" and float(v["TestHTwoSlopeB"]) > 0
    assert all(name.isalpha() for name in v)


def test_tables_content_and_format():
    docs = _docs()
    t = pm.test_results_table(docs, FZ["variants"]["r5"]["matched"]["phi"])
    bs = chr(92)
    assert t.startswith(bs + "begin{table*}") and "tab:test-results" in t
    for label in ("fixed, $" + bs + "phi=0$", "fixed, $" + bs + "phi=7$", "dyn., $40$, $(4, 7.5)$",
                  "matched, $" + bs + "phi=5.58$", "FollowerStopper", "PI"):
        assert label in t
    assert t.index("Primary analysis") < t.index("fixed, $" + bs + "phi=0$") < t.index("Sensitivity analysis R5")
    assert "G3" in t and "G1a" in t and bs + "checkmark" in t        # FS fails G3; PI collides in H
    d = pm.test_decisions_table(docs["r5"])
    assert "tab:test-decisions" in d and "H1 supported" in d and "tabcolsep}{3pt}" in d
    assert d.count(bs + bs) >= 17 and "$[" in d and "H2 supported" in d


@pytest.mark.filterwarnings("ignore::DeprecationWarning")      # pyparsing inside matplotlib, not our code
def test_figure_is_reproducible(tmp_path):
    r5 = _docs()["r5"]
    a, b = tmp_path / "a.pdf", tmp_path / "b.pdf"
    data = pf.position_ratio_figure(r5, a)
    pf.position_ratio_figure(r5, b)
    assert a.read_bytes()[:4] == b"%PDF" and a.read_bytes() == b.read_bytes()
    assert data["B"] == pytest.approx([0.9, 0.92, 0.95])
