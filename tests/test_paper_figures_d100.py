"""Figures of (test results, chain validation, data waves) and controller labels, on synthetic inputs."""
import copy
import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dyn_svo.analysis import spectrum as spm
from dyn_svo.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pf = _load("paper_figures")


def _docs():
    def one(names, fail):
        lf, gr = {}, {}
        for i, n in enumerate(names):
            for j, e in enumerate(("B", "D", "H")):
                k = n + "|" + e
                lf[k] = -0.01 * (i + 1) - 0.02 * j
                gr[k] = {"passes": k not in fail}
        return {"experiment_log_for": lf, "guardrails": gr}
    return {"primary": one(["fixed_0", "fs", "pi"], {"fs|H"}),
            "r5": one(["dyn_W40_lo4_hi7p5", "fixed_7", "fixed_match", "fs", "pi"], {"pi|H"})}


def ring_frames(n_veh=20, seconds=400, period=40.0, ring=260.0, hz=30, seed=0):
    """Synthetic ring trajectories at the recording rate (30 Hz) with a travelling speed wave of the given period."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, seconds, 1.0 / hz)
    rows = []
    for i in range(n_veh):
        v = 5.0 + 1.5 * np.sin(2 * np.pi * t / period - 0.6 * i) + 0.05 * rng.standard_normal(len(t))
        x = ring * i / n_veh + np.cumsum(v) / hz
        rows.append(pd.DataFrame({"exp": "A", "veh_id": i, "k": np.arange(len(t)), "t": t, "x": x, "v": v,
                                  "in_window": t >= 30.0}))
    return pd.concat(rows, ignore_index=True)


def test_paper_sources_list_chain_validation():
    src = load_config("paper_sources")
    assert src["validation"]["chainval"] == {"path": "validation/chainval_v2/summary.json",
                                             "md5": "d04532bbb54622b94fdad121f1401472"}


def test_controller_labels():
    assert pf.controller_label("fs") == "FollowerStopper" and pf.controller_label("pi") == "PI"
    assert pf.controller_label("fixed_match").startswith("MPC, matched fixed")
    assert pf.controller_label("dyn_W40_lo4_hi7p5").startswith("MPC, dynamic") and "phi_t" in pf.controller_label("dyn_W40_lo4_hi7p5")
    assert pf.controller_label("fixed_0").startswith("MPC, fixed") and pf.controller_label("fixed_0").endswith("phi = 0$")
    assert "phi = 7" in pf.controller_label("fixed_7") and "pi/16" in pf.controller_label("fixed_7")


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_test_results_figures_values_and_reproducible(tmp_path):
    a, b, c = tmp_path / "a.pdf", tmp_path / "b.pdf", tmp_path / "c.pdf"
    d = pf.test_results_figure(_docs(), "primary", a)
    pf.test_results_figure(_docs(), "primary", b)
    r = pf.test_results_figure(_docs(), "r5", c)
    assert a.read_bytes() == b.read_bytes() and a.read_bytes()[:4] == b"%PDF" and c.read_bytes()[:4] == b"%PDF"
    assert d["fs|H"] == [round(math.exp(-0.02 - 0.04), 6), False]
    assert d["fixed_0|B"][1] is True and r["pi|H"][1] is False
    assert len(d) == 9 and len(r) == 15


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_chain_validation_figure_means_logs_by_position(tmp_path):
    cv = {"summary": [{"exp": "A", "ok_sigma": True, "ok_slope": True, "ok_speed": True},
                      {"exp": "E", "ok_sigma": False, "ok_slope": True, "ok_speed": True}],
          "episodes": [{"exp": "A", "log_ratio_by_position": [0.0, -0.2, -0.4]},
                       {"exp": "A", "log_ratio_by_position": [-0.2, -0.2]},
                       {"exp": "E", "log_ratio_by_position": [0.1, float("nan"), 0.3]}]}
    a, b = tmp_path / "a.pdf", tmp_path / "b.pdf"
    d = pf.chain_validation_figure(cv, a)
    pf.chain_validation_figure(copy.deepcopy(cv), b)
    assert a.read_bytes() == b.read_bytes()
    assert d["A"] == [round(math.exp(-0.1), 6), round(math.exp(-0.2), 6), round(math.exp(-0.4), 6)]
    assert d["E"] == [round(math.exp(0.1), 6), round(math.exp(0.3), 6)]


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_data_figures_reproducible_and_find_the_wave(tmp_path):
    df = ring_frames()
    spectra = {"A": spm.experiment_spectrum(df)[:2], "C": spm.experiment_spectrum(ring_frames(period=50.0, seed=1))[:2]}
    win = df[df["in_window"]][["exp", "veh_id", "k", "t", "x", "v"]]
    a, b, c, e = tmp_path / "a.pdf", tmp_path / "b.pdf", tmp_path / "c.pdf", tmp_path / "e.pdf"
    d = pf.data_spectrum_figure(spectra, [20.0, 64.0], a)
    pf.data_spectrum_figure(spectra, [20.0, 64.0], b)
    t1 = pf.data_vehicle_speed_figure(win, 260.0, c)
    pf.data_vehicle_speed_figure(win, 260.0, e)
    assert a.read_bytes() == b.read_bytes() and a.read_bytes()[:4] == b"%PDF"
    assert c.read_bytes() == e.read_bytes() and c.read_bytes()[:4] == b"%PDF"
    assert t1["experiment"] == "A" and t1["t_span_s"] == 300.0 and t1["vehicles"] == 20 and t1["missing_cells"] == 0
    assert t1["rate_hz"] == pytest.approx(10.0)
    lo, hi = t1["speed_clip_p2_p98_m_s"]
    assert 3.3 < lo < 3.7 and 6.3 < hi < 6.7   # 5 +- 1.5 sin: p2 and p98 near 3.5 and 6.5
    assert d["A"] == pytest.approx(40.0, rel=0.05) and d["C"] == pytest.approx(50.0, rel=0.05)


def test_vehicle_speed_matrix_order_and_values():
    df = ring_frames(n_veh=5, seconds=60)
    win = df[df["in_window"]]
    order, tt, m = pf.vehicle_speed_matrix(win, 260.0, t_span=10.0)
    start = win[win["k"] == win["k"].min()].set_index("veh_id")["x"] % 260.0
    assert order == list(start.sort_values(ascending=False).index)
    assert tt[0] == 0.0 and tt[-1] == pytest.approx(10.0) and m.shape == (5, 101)
    for r, vid in enumerate(order):
        g = win[(win["veh_id"] == vid) & (win["k"] % 3 == 0) & (win["t"] <= win["t"].min() + 10.0)]
        assert np.array_equal(m[r], g.sort_values("t")["v"].to_numpy())
    with pytest.raises(ValueError):
        pf.vehicle_speed_matrix(win[~((win["veh_id"] == 2) & (win["k"] == win["k"].min()))], 260.0, t_span=10.0)


def test_period_axis_ticks(tmp_path):
    plt = pf._plt()
    fig, ax = plt.subplots()
    ax.plot([10, 600], [1, 1])
    pf._period_axis(ax, 10.0, 600.0)
    assert list(ax.get_xticks()) == [10, 20, 50, 100, 200, 500]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["10", "20", "50", "100", "200", "500"]
    pf._period_axis(ax, 10.0, 120.0)
    assert list(ax.get_xticks()) == [10, 20, 50, 100]
    plt.close(fig)
