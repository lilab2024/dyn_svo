import numpy as np
import pandas as pd
import pytest

from dyn_svo.data import loader as ld
from dyn_svo.data import manifest as mf
from dyn_svo.data import split as sp

L = ld.RING_LENGTH_M


def _ring(exp, ids, seconds=70.0, fuel_nan=()):
    """Synthetic ring (x = front bumper): ids[0] leads ids[1]...; ids[0] follows ids[-1] (seam)."""
    n, lengths = len(ids), [4.5 + 0.2 * i for i in range(len(ids))]
    k = np.arange(int(seconds * ld.HZ))
    t = k / ld.HZ
    rows = []
    fronts = []
    for i, vid in enumerate(ids):
        x = -i * L / n + 3.0 * t + 0.5 * np.sin(0.2 * t + i)
        fronts.append(x)
    for i, vid in enumerate(ids):
        lead = (i - 1) % n
        gap = fronts[lead] - fronts[i] - lengths[lead]   # front-bumper convention
        fuel = np.full(len(k), np.nan) if vid in fuel_nan else np.full(len(k), 2.0)
        rows.append(pd.DataFrame({
            "exp_id": "exp" + exp, "veh_id": vid, "length": lengths[i], "height": 1.5,
            "make": "X", "year": 2014, "model": "M", "timestamp": t, "cam_distance": fronts[i],
            "cam_velocity": 3.0, "cam_acceleration": 0.0, "leader_id": ids[lead],
            "cam_leader_gap": gap, "follower_id": ids[(i + 1) % n], "cam_follower_gap": 0.0,
            "obd_velocity": 3.0, "obd_fuel_rate": fuel, "cam_upper_front_x": 0,
            "cam_upper_front_y": 0, "cam_lower_rear_x": 0, "cam_lower_rear_y": 0}))
    return pd.concat(rows, ignore_index=True)


def _setup(tmp_path, frames):
    folder = tmp_path / "shared_data" / "arizona_ring"
    folder.mkdir(parents=True)
    files = {}
    for exp, frame in frames.items():
        p = folder / ("exp" + exp + ".csv")
        frame.to_csv(p, index=False)
        files[p.name] = {"bytes": p.stat().st_size, "md5": mf.md5_of(p)}
    return {"data_dir": "shared_data/arizona_ring", "files": files}


def test_correct_gaps():
    out = ld.correct_gaps([5.0, -250.0, 0.2])
    assert np.allclose(out, [5.0, 10.0, 0.2])


def test_load_and_conservation(tmp_path):
    m = _setup(tmp_path, {"A": _ring("A", [1, 2, 3, 4, 5])})
    df = ld.load_experiment("A", manifest=m, root=tmp_path)
    assert df["gap"].min() > 0
    assert ld.check_conservation(df) < 1e-6
    seam = df.groupby("veh_id")["gap"].median()
    assert seam.max() < 60
    assert set(df["regime"]) == {"I"}
    assert df["t"].iloc[1] == pytest.approx(1 / 30)


def test_flags_r1_r2(tmp_path):
    m = _setup(tmp_path, {"F": _ring("F", [18, 19, 20, 21])})
    df = ld.load_experiment("F", manifest=m, root=tmp_path)
    assert not df.loc[df["t"] < 30, "in_window"].any()
    assert df.loc[df["t"] >= 30, "in_window"].all()
    v20 = df[df["veh_id"] == 20]
    assert v20.loc[v20["t"] < 59, "follower_ok"].all()
    assert not v20.loc[v20["t"] >= 59, "follower_ok"].any()
    assert df.loc[df["veh_id"] != 20, "follower_ok"].all()
    assert set(df["regime"]) == {"II"}


def test_follower_pairs_and_position_check(tmp_path):
    m = _setup(tmp_path, {"F": _ring("F", [18, 19, 20, 21])})
    df = ld.load_experiment("F", manifest=m, root=tmp_path)
    pairs = ld.follower_pairs(df)
    assert pairs["in_window"].all() and pairs["follower_ok"].all()
    assert {"x_l", "v_l", "a_l", "len_l"} <= set(pairs.columns)
    assert ld.position_gap_check(df)["max_abs_diff_m"] < 1e-6
    w = ld.with_leader(df)
    assert np.abs(w["x_l"] - w["len_l"] - w["x"] - w["gap"]).max() < 1e-9


def test_conservation_violation_detected(tmp_path):
    frame = _ring("A", [1, 2, 3, 4, 5])
    frame.loc[frame["veh_id"] == 3, "cam_leader_gap"] += 2.0
    m = _setup(tmp_path, {"A": frame})
    with pytest.raises(ld.DataQualityError):
        ld.load_experiment("A", manifest=m, root=tmp_path)


def test_fuel_missing_must_match_config(tmp_path):
    m = _setup(tmp_path, {"A": _ring("A", [1, 2, 3, 4, 5], fuel_nan=(3,))})
    with pytest.raises(ld.DataQualityError):
        ld.load_experiment("A", manifest=m, root=tmp_path)


def test_fuel_ok_flag(tmp_path):
    split = sp.load_split()
    split["fuel"]["missing_full"]["A"] = [3]
    frame = _ring("A", [1, 2, 3, 4, 5], fuel_nan=(3,))
    m = _setup(tmp_path / "x", {"A": frame})
    df = ld.load_experiment("A", split=split, manifest=m, root=tmp_path / "x")
    assert not df.loc[df["veh_id"] == 3, "fuel_ok"].any()
    assert df.loc[df["veh_id"] != 3, "fuel_ok"].all()


def test_test_set_locked_before_reading(tmp_path):
    with pytest.raises(sp.TestSetLocked):
        ld.load_experiment("B", manifest={"data_dir": "none", "files": {}}, root=tmp_path)


def test_regime_map_in_config():
    s = sp.load_split()
    assert s["instruction_regime"] == {"A": "I", "B": "I", "C": "I", "D": "I", "E": "I",
                                       "F": "II", "G": "II", "H": "II"}
