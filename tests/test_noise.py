import numpy as np
import pandas as pd
import pytest

from dyn_svo.calibration import noise
from dyn_svo.sim import idm

P = {"v0": 15.0, "T": 1.2, "s0": 2.0, "a_max": 1.0, "b": 1.5}


def _pairs(veh, extra=0.0, n=900, seed=0):
    rng = np.random.default_rng(seed)
    k = np.arange(n)
    v = 4.0 + np.sin(k / 50.0)
    v_l = 4.0 + np.sin(k / 50.0 + 0.3)
    gap = 12.0 + np.cos(k / 40.0)
    a = idm.idm_accel(v, gap, v - v_l, **P) + extra * rng.standard_normal(n)
    return pd.DataFrame({"veh_id": veh, "k": k, "v": v, "v_l": v_l, "gap": gap, "a": a})


def test_residuals_zero_for_exact_idm():
    r = noise.accel_residuals(_pairs(3), {3: P})
    assert len(r) == 300 and np.abs(r).max() < 1e-12


def test_residual_sd_recovers_noise():
    r = noise.accel_residuals(_pairs(3, extra=0.3, n=30000), {3: P})
    assert np.std(r) == pytest.approx(0.3, rel=0.05)


def test_noise_table_per_fold_and_regime():
    pairs_of = {"A": _pairs(3, extra=0.2, n=9000), "C": _pairs(3, extra=0.2, n=9000, seed=1),
                "F": _pairs(3, extra=0.5, n=9000, seed=2)}
    regimes_of = {"A": "I", "C": "I", "F": "II"}
    rows = []
    for fold in ("loeo_A", "all_dev"):
        for reg in ("I", "II"):
            for veh in ("3", "pooled"):
                rows.append(dict(fold=fold, regime=reg, variant="nodelay", veh=veh, **P))
    table = pd.DataFrame(rows)
    folds = [("loeo_A", ["C", "F"], "A"), ("all_dev", ["A", "C", "F"], None)]
    out = noise.noise_table(folds, regimes_of, pairs_of, table)
    assert set(zip(out["fold"], out["regime"])) == {("loeo_A", "I"), ("loeo_A", "II"),
                                                    ("all_dev", "I"), ("all_dev", "II")}
    assert noise.noise_sd(out, "loeo_A", "I") == pytest.approx(0.2, rel=0.1)
    assert noise.noise_sd(out, "all_dev", "II") == pytest.approx(0.5, rel=0.1)
    assert out.set_index(["fold", "regime"]).loc[("loeo_A", "I"), "n"] == 3000
    with pytest.raises(KeyError):
        noise.noise_sd(out, "loeo_G", "I")
