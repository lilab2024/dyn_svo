import pandas as pd
import pytest

from dyn_svo.calibration import report

B = {"v0": [5.0, 30.0], "T": [0.1, 3.0], "s0": [0.1, 5.0], "a_max": [0.1, 4.0], "b": [0.1, 5.0]}


def _row(fold, veh, variant, train, held, b=1.0, v0=15.0, delay=0):
    return {"fold": fold, "regime": "I", "veh": veh, "variant": variant, "rmse_m": train,
            "held_rmse_m": held, "v0": v0, "T": 1.5, "s0": 2.0, "a_max": 1.0, "b": b,
            "delay_steps": delay, "collided_share": 0.0, "seconds": 60.0, "commit": "c" * 40,
            "dirty": False, "data_check": "full_md5 (parent)"}


def _frame(nodelay_held, delay_held, b_values=None):
    rows = []
    b_values = b_values or [1.0] * len(nodelay_held)
    for i, (hn, hd, b) in enumerate(zip(nodelay_held, delay_held, b_values)):
        rows.append(_row("loeo_A", str(i + 1), "nodelay", 1.0, hn, b=b))
        rows.append(_row("loeo_A", str(i + 1), "delay", 1.0, hd, b=b, delay=3))
    rows.append(_row("all_dev", "1", "nodelay", 1.0, None))
    rows.append(_row("all_dev", "pooled", "nodelay", 1.2, None))
    return pd.DataFrame(rows)


def test_delay_rule_picks_lower_mean():
    out = report.delay_rule(_frame([1.2, 1.4], [1.0, 1.1]))
    assert out["chosen"] == "delay" and out["n_pairs"] == 2
    assert out["mean_held_rmse_m"]["nodelay"] == pytest.approx(1.3)
    assert out["share_pairs_delay_better"] == 1.0


def test_delay_rule_tie_prefers_nodelay():
    assert report.delay_rule(_frame([1.0, 2.0], [2.0, 1.0]))["chosen"] == "nodelay"


def test_bound_shares():
    df = _frame([1.0, 1.0, 1.0], [1.0, 1.0, 1.0], b_values=[0.1, 0.1004, 2.0])
    s = report.bound_shares(report.per_vehicle(df[df["variant"] == "nodelay"]), B)
    assert s["b"]["lower"] == pytest.approx(2 / 4)   # 3 LOEO fits + 1 all_dev fit; 2 at 0.1
    assert s["v0"]["upper"] == 0.0


def test_boundary_rule_needs_both_conditions():
    df = _frame([2.0, 2.0, 2.0], [2.0, 2.0, 2.0], b_values=[0.1, 0.1, 0.1])
    assert report.boundary_rule(df, "nodelay", B)["triggered"]
    df = _frame([1.2, 1.2, 1.2], [1.2, 1.2, 1.2], b_values=[0.1, 0.1, 0.1])
    assert not report.boundary_rule(df, "nodelay", B)["triggered"]
    df = _frame([2.0, 2.0, 2.0], [2.0, 2.0, 2.0], b_values=[1.0, 1.0, 1.0])
    assert not report.boundary_rule(df, "nodelay", B)["triggered"]


def test_build_report_keys():
    rep = report.build_report(_frame([1.2, 1.4], [1.0, 1.1]), B)
    assert rep["delay_rule"]["chosen"] == "delay"
    assert rep["provenance_of_results"]["any_dirty"] is False
    assert len(rep["pooled"]) == 1 and rep["delay_steps_counts"] == {3: 2}
