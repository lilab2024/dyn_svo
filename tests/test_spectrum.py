import numpy as np
import pandas as pd
import pytest

from dyn_svo.analysis import spectrum as spc
from dyn_svo.config import load_config
from dyn_svo.data.manifest import REPO_ROOT


def _frame(period_s, seconds=600, n_veh=4, trend=0.0, seed=0):
    rng = np.random.default_rng(seed)
    k = np.arange(int(seconds * 30))
    t = k / 30.0
    rows = []
    for veh in range(1, n_veh + 1):
        v = 4.0 + trend * t + (np.sin(2 * np.pi * t / period_s + veh) if period_s else 0.0)
        v = v + 0.05 * rng.standard_normal(len(t))
        rows.append(pd.DataFrame({"veh_id": veh, "k": k, "v": v, "in_window": t >= 30}))
    return pd.concat(rows, ignore_index=True)


def test_dominant_period_found():
    f, psd, n_veh, n = spc.experiment_spectrum(_frame(40.0))
    period, ratio, edge = spc.dominant_period(f, psd)
    assert period == pytest.approx(40.0, rel=0.05) and ratio > 5 and not edge and n_veh == 4


def test_trend_only_gives_edge_peak():
    f, psd, _, _ = spc.experiment_spectrum(_frame(None, trend=0.01))
    assert spc.dominant_period(f, psd)[2]


def test_detrend_window_rule():
    assert spc.detrend_window([33.1, 43.6, 64.0, 36.9, 19.6]) == 70
    assert spc.detrend_window([12.0]) == 30
    assert spc.detrend_window([150.0]) == 120
    assert spc.detrend_window([60.0]) == 60
    with pytest.raises(ValueError):
        spc.detrend_window([])


def test_metrics_config_and_plan_agree():
    c = load_config("metrics")
    w = c["detrend_window_s"]
    assert w == 70 and c["detrend_sensitivity_s"] == [w // 2, 2 * w] and c["warmup_s"] == 30
    import yaml
    front = yaml.safe_load((REPO_ROOT / "config" / "decision_constants.yaml").read_text(encoding="utf-8"))
    assert front["detrend_window_s"] == w and front["warmup_s"] == c["warmup_s"]
