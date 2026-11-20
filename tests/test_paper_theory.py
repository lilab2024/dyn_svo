"""Theory macros, certificate table, gain figure, R5 selection macros and test rows on synthetic inputs."""
import copy
import importlib.util
import math
from pathlib import Path

import pandas as pd
import pytest

from dyn_svo.config import load_config

ROOT = Path(__file__).resolve().parents[1]
U = math.pi / 16
BS = chr(92)


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pm = _load("paper_macros")
pf = _load("paper_figures")


def _cert(g):
    return {"gamma": g, "frozen_reference": g * 0.9, "verified": True, "inflation": 1e-6}


def _v1():
    out = {}
    for r, lam_h in (("I", 12.0), ("II", 1.8)):
        rows = []
        for i in range(80):
            phi = i * math.pi / 160
            rows.append({"phi": phi, "phi_over_pi16": phi / U, "lambda_av": 8.0 - 10.0 * phi / U,
                         "gain_at": {"20.0": 0.9 - 0.05 * phi / U}, "band_peak": [1.0 + 0.001 * i, 27.0],
                         "long_peak": [1.1 + 0.001 * i, 150.0]})
        iv = [dict({k: _cert(1.2) for k in ("continuous_dv", "discrete_dv", "continuous_e_s", "discrete_e_s")},
                   phi=[a * U, b * U], rate_gridded_dv={"gamma": 1.1}) for a, b in ((3, 7), (4, 7.5))]
        fx = [dict({k: _cert(1.0) for k in ("continuous_dv", "discrete_dv", "continuous_e_s", "discrete_e_s")},
                   phi=[j * U, j * U]) for j in range(8)]
        out[r] = {"speeds": {"median": 3.0, "p10": 2.0, "p90": 4.5, "n_rows": 9},
                  "human": {k: {"lambda_h": lam_h + d} for k, d in (("median", 0.0), ("p10", -1.0), ("p90", 2.0))},
                  "idm_pooled": {"v0": 30.0, "T": 1.8, "s0": 1.7, "a_max": 0.4, "b": 0.1},
                  "signs": {"positive": True, "k_s_lower": 0.0174, "k_v_lower": 0.1394, "q_lower": 0.0254},
                  "gain_table": rows, "certificates": {"intervals": iv, "fixed": fx}}
    settings = {"band_s": [20.0, 64.0], "long_s": [64.0, 600.0], "sign_range": [0.0, 7.5 * U], "sign_grid": 2000,
                "n_grid": 400, "intervals": [[3 * U, 7 * U], [4 * U, 7.5 * U]], "fixed": [j * U for j in range(8)],
                "rate_grid": 60, "n_reference": 41}
    for r in out:
        for c in out[r]["certificates"]["intervals"]:
            c["n_corners"] = 3208
    out["II"]["certificates"]["intervals"][1]["discrete_dv"]["inflation"] = 1e-4
    return {"theory": out, "provenance": {"dirty": False}, "settings": settings,
            "solver_packages": {"cvxpy": "1.9.3", "clarabel": "0.11.1"}}


def _case(cross, lam=12.0):
    return {"human": {"median": {"lambda_h": lam}}, "head_to_tail": {"19": {"crossings": cross}, "20": {"crossings": cross}}}


def _v2(v1):
    out = {}
    for r in ("I", "II"):
        a = v1["theory"][r]
        out[r] = {"speeds": a["speeds"], "all_dev": dict(_case([6.7 * U]), params=dict(a["idm_pooled"])),
                  "loeo": {"loeo_A": _case([6.4 * U], 9.6), "loeo_C": _case([6.9 * U], 15.4)},
                  "b_grid": [dict(_case([6.7 * U]), b=0.1), dict(_case([2.2 * U]), b=0.3), dict(_case([]), b=0.5)],
                  "critical_b_median": {"b": 0.336 if r == "I" else None},
                  "gain_curves": {"periods_s": [10.0, 100.0, 600.0],
                                  "curves": [{"phi_over_pi16": float(j), "gain": [0.9, 1.0 + 0.01 * j, 1.05]} for j in range(3)]}}
    return {"sensitivity": out, "settings": {"b_grid": [0.1, 0.3, 0.5]}, "provenance": {"dirty": False}}


def test_paper_sources_list_theory_v1():
    src = load_config("paper_sources")
    assert src["theory"]["v1"] == {"path": "theory/theory_v1/theory.json", "md5": "68865cd1c8b5a47bc1d0ab797223fcba"}


def test_r5_selection_macros_from_the_frozen_config():
    m = pm.Macros()
    pm.r5_selection_macros(m, load_config("test_freeze"))
    v = {it["name"]: it["value"] for it in m.items}
    assert v == {"TestRfiveFixedPhi": "7", "TestRfiveDynWindow": "40", "TestRfiveDynLo": "4", "TestRfiveDynHi": "7.5",
                 "TestRfiveMatchPhi": "5.58"}


def test_theory_problems_accept_good_and_refuse_bad():
    v1 = _v1(); v2 = _v2(v1)
    assert pm.theory_problems(v1, v2) == []
    b = copy.deepcopy(v1); b["provenance"]["dirty"] = True
    assert any("dirty" in p for p in pm.theory_problems(b, v2))
    b = copy.deepcopy(v1); b["theory"]["II"]["certificates"]["fixed"][3]["discrete_e_s"]["verified"] = False
    assert any("not verified" in p for p in pm.theory_problems(b, v2))
    b = copy.deepcopy(v1); b["theory"]["I"]["signs"]["positive"] = False
    assert any("signs" in p for p in pm.theory_problems(b, v2))
    c = copy.deepcopy(v2); c["sensitivity"]["I"]["all_dev"]["params"]["b"] = 0.2
    assert any("pooled IDM" in p for p in pm.theory_problems(v1, c))
    c = copy.deepcopy(v2); c["sensitivity"]["II"]["all_dev"]["head_to_tail"]["19"]["crossings"] = []
    assert any("sign changes" in p for p in pm.theory_problems(v1, c))


def test_theory_macros_values():
    v1 = _v1(); v2 = _v2(v1)
    m = pm.Macros()
    pm.theory_macros(m, v1, v2, "r1", "r2", [0.1, 5.0])
    v = {it["name"]: it["value"] for it in m.items}
    assert all(k.isalpha() for k in v) and len(v) == len(m.items)
    assert v["ThBLowerBound"] == "0.1" and v["ThIdmBI"] == "0.100" and v["ThSpeedMedI"] == "3.00"
    assert v["ThLamHMedI"] == "12.00" and v["ThLamHLowII"] == "0.80" and v["ThKsLowI"] == "0.0174"
    assert v["ThLamAvSignLoI"] == "0.7" and v["ThLamAvSignHiI"] == "0.8"
    assert v["ThHttLoI"] == "6.70" and v["ThHttHiI"] == "6.70"
    assert v["ThLoeoLamHMinI"] == "9.60" and v["ThLoeoLamHMaxII"] == "15.40"
    assert v["ThBCritI"] == "0.34" and v["ThBCritII"] == "none"
    assert v["ThBGridHttLoI"] == "2.20" and v["ThBGridHttHiI"] == "6.70" and v["ThBGridNoCrossI"] == "0.5"
    assert v["ThBGrid"] == "0.1, 0.3 and 0.5"
    assert v["ThGainTwentyZero"] == "0.900" and v["ThGainTwentySeven"] == "0.550"
    assert v["ThBandPeakPhiII"] == "7.5" and v["ThBandPeakPeriodII"] == "27" and v["ThLongPeakI"] == "1.175"
    assert (v["ThBandLo"], v["ThBandHi"], v["ThLongHi"], v["ThSignHi"]) == ("20", "64", "600", "7.5")
    assert (v["ThIntALo"], v["ThIntAHi"], v["ThIntBLo"], v["ThIntBHi"]) == ("3", "7", "4", "7.5")
    assert (v["ThFixedHi"], v["ThFixedCount"], v["ThSignGrid"], v["ThCertGrid"]) == ("7", "8", "2000", "400")
    assert (v["ThRateGrid"], v["ThRefPoints"], v["ThCorners"]) == ("60", "41", "3208")
    assert v["ThInflMin"] == "10^{-6}" and v["ThInflMax"] == "10^{-4}"
    assert v["ThCvxpy"] == "1.9.3" and v["ThClarabel"] == "0.11.1"
    assert pm._power10(2.5e-5) == "2.5e-05"


def test_certificate_table():
    t = pm.theory_certificates_table(_v1())
    assert t.startswith(BS + "begin{table*}") and "tab:theory-certificates" in t
    for row in ("I, $[3, 7]$", "I, $[4, 7.5]$", "II, $[3, 7]$", "II, $[4, 7.5]$"):
        assert row in t
    assert "All " + BS + "ThFixedCount{} fixed values" in t and "string stability" in t and BS + "PhiRate" in t
    assert t.count(BS + BS) == 6 and "1.200 & 1.200 & 1.080 & 1.100" in t
    assert BS + "caption{Certified $L_2$ gains of the closed AV loop.}" in t
    bad = _v1()
    bad["settings"]["fixed"] = bad["settings"]["fixed"][:-1]
    with pytest.raises(SystemExit):
        pm.theory_certificates_table(bad)


def test_short_captions_and_notes_d106():
    split_cfg = load_config("data_split")
    rows = [{"exp": e, "regime": split_cfg["instruction_regime"][e], "vehicles": 20, "usable_s": 300.0}
            for e in split_cfg["experiments"]]
    for t in (pm.experiments_table(rows, split_cfg), pm.theory_certificates_table(_v1())):
        cap = t.split(BS + "caption{", 1)[1].split("}" + chr(10), 1)[0]
        assert cap.count(". ") == 0 and cap.endswith(".") and len(cap) < 100
        note = [ln for ln in t.split(chr(10)) if ln.startswith("{" + BS + "footnotesize" + BS + "emph{Note:} ")]
        assert len(note) == 1 and note[0].endswith(BS + "par}")
        lines = t.split(chr(10))
        assert lines.index(BS + "end{tabular}") < lines.index(note[0]) < len(lines) - 2
    assert pm._table("table", "l", "C.", "tab:x", ["h " + BS + BS], [["a"]]).count("Note:") == 0


def test_experiments_table_with_test_rows_and_test_rows_use_the_opened_set(monkeypatch):
    split_cfg = load_config("data_split")
    rows = [{"exp": e, "regime": split_cfg["instruction_regime"][e], "vehicles": 20, "usable_s": 300.0}
            for e in split_cfg["experiments"]]
    t = pm.experiments_table(rows, split_cfg)
    assert "locked" not in t and "Test rows are filled" not in t and BS + "DropFirst" in t and "B & I & test & 20 & 300 " in t
    seen = []

    def fake(e, allow_test=False, **kw):
        seen.append((e, allow_test))
        return pd.DataFrame({"k": [0, 30 * 400], "veh_id": [1, 2], "t": [0.0, 1.0]})
    monkeypatch.setattr(pm.ld, "load_experiment", fake)
    out = pm.test_data_rows(split_cfg)
    assert seen == [("B", True), ("D", True), ("H", True)]
    assert out[0]["vehicles"] == 2 and out[0]["usable_s"] == pytest.approx(400.0 - split_cfg["segments"]["drop_first_s"])


@pytest.mark.filterwarnings("ignore::DeprecationWarning")      # pyparsing inside matplotlib, not our code
def test_gain_figure_is_reproducible(tmp_path):
    v2 = _v2(_v1())
    a, b = tmp_path / "a.pdf", tmp_path / "b.pdf"
    data = pf.gain_figure(v2, "II", a)
    pf.gain_figure(v2, "II", b)
    assert a.read_bytes()[:4] == b"%PDF" and a.read_bytes() == b.read_bytes()
    assert data["2"] == [0.9, 1.05]
