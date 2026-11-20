"""Load one ARED experiment into a clean, flagged frame."""
import numpy as np
import pandas as pd

from dyn_svo.data import manifest as mf
from dyn_svo.data import split as sp

HZ = 30
RING_LENGTH_M = 260.0      # by construction: 260 m / 3840 px (Wu et al. 2019, Sec. 3.3)
CONSERVATION_TOL_M = 0.5   # |sum(gap + length) - L| per frame
MIN_GAP_TOL_M = -0.5       # corrected gaps below this are an error

RAW_COLS = ["exp_id", "veh_id", "length", "timestamp", "cam_distance", "cam_velocity",
            "cam_acceleration", "leader_id", "cam_leader_gap", "obd_velocity", "obd_fuel_rate"]
RENAME = {"cam_distance": "x", "cam_velocity": "v", "cam_acceleration": "a",
          "cam_leader_gap": "gap_raw", "obd_velocity": "v_obd", "obd_fuel_rate": "fuel"}


class DataQualityError(RuntimeError):
    """Raised when a loaded experiment violates a structural check."""


def correct_gaps(gap_raw, ring_length=RING_LENGTH_M):
    """Add one ring length to gaps computed across the unwrapped-distance seam."""
    gap_raw = np.asarray(gap_raw, dtype=float)
    return np.where(gap_raw < -ring_length / 2.0, gap_raw + ring_length, gap_raw)


def check_conservation(df, ring_length=RING_LENGTH_M, tol=CONSERVATION_TOL_M):
    """Per frame, sum(gap + length) must equal the ring length; return max abs error."""
    per_frame = (df["gap"] + df["length"]).groupby(df["k"]).sum()
    err = float((per_frame - ring_length).abs().max())
    if err > tol:
        raise DataQualityError("Ring conservation error " + str(round(err, 3)) + " m > tol")
    return err


def load_experiment(exp, allow_test=False, data_check="size", split=None, manifest=None,
                    root=None):
    """Return one experiment with corrected gaps and usage flags.

    Columns: exp, veh_id, regime, k (frame), t (s), x, v, a, length, leader_id, gap (m),
    v_obd, fuel (L/h), fuel_ok, in_window (R1), follower_ok (R2).
    """
    split = split if split is not None else sp.load_split()
    sp.check_access([exp], allow_test=allow_test, split=split)
    manifest = manifest if manifest is not None else mf.load_manifest()
    mf.verify_data(data_check, manifest=manifest, root=root)

    raw = pd.read_csv(mf.data_file(exp, manifest, root), usecols=RAW_COLS)
    df = raw.rename(columns=RENAME).drop(columns=["exp_id"])
    df.insert(0, "exp", exp)
    df.insert(2, "regime", split["instruction_regime"][exp])

    df["k"] = np.rint(df.pop("timestamp").to_numpy() * HZ).astype(int)
    df = df.sort_values(["veh_id", "k"]).reset_index(drop=True)
    if df.duplicated(["veh_id", "k"]).any():
        raise DataQualityError("Duplicate (veh_id, frame) rows")
    steps = df.groupby("veh_id")["k"].diff().dropna()
    if not (steps == 1).all():
        raise DataQualityError("Frames are not consecutive for every vehicle")
    n_veh = df["veh_id"].nunique()
    if not (df.groupby("k").size() == n_veh).all():
        raise DataQualityError("Not every vehicle is present in every frame")
    df["t"] = df["k"] / HZ

    df["gap"] = correct_gaps(df.pop("gap_raw"), RING_LENGTH_M)
    if float(df["gap"].min()) < MIN_GAP_TOL_M:
        raise DataQualityError("Negative corrected gap " + str(round(float(df["gap"].min()), 3)))
    check_conservation(df)

    seg = split["segments"]
    df["in_window"] = df["t"] >= seg["drop_first_s"]
    t_int = seg["intervention_s"].get(exp)
    excluded = df["veh_id"].isin(seg["exclude_follower_after_intervention"])
    df["follower_ok"] = True if t_int is None else ~(excluded & (df["t"] >= t_int))

    all_nan = df.groupby("veh_id")["fuel"].apply(lambda s: bool(s.isna().all()))
    found = sorted(int(v) for v in all_nan[all_nan].index)
    expected = sorted(split["fuel"]["missing_full"].get(exp, []))
    if found != expected:
        raise DataQualityError("Fully missing fuel " + str(found) + " != config " + str(expected))
    df["fuel_ok"] = ~df["veh_id"].isin(found)

    cols = ["exp", "veh_id", "regime", "k", "t", "x", "v", "a", "length", "leader_id", "gap",
            "v_obd", "fuel", "fuel_ok", "in_window", "follower_ok"]
    return df[cols]


def with_leader(df):
    """Attach leader x, v, a, length to every row (same frame)."""
    lead = df[["veh_id", "k", "x", "v", "a", "length"]].rename(
        columns={"veh_id": "leader_id", "x": "x_l", "v": "v_l", "a": "a_l", "length": "len_l"})
    out = df.merge(lead, on=["leader_id", "k"], how="left", validate="many_to_one")
    if out["x_l"].isna().any():
        raise DataQualityError("Leader missing for some rows")
    # Seam pair: the leader's unwrapped position is one lap behind; shift it by whole laps so
    # that x_l - len_l - x equals the corrected gap for every row.
    laps = np.rint((out["x"] + out["gap"] + out["len_l"] - out["x_l"]) / RING_LENGTH_M)
    out["x_l"] = out["x_l"] + laps * RING_LENGTH_M
    return out


def follower_pairs(df):
    """Follower-leader rows usable for human-model calibration (R1 and R2 applied)."""
    out = with_leader(df)
    return out[out["in_window"] & out["follower_ok"]].reset_index(drop=True)


def position_gap_check(df, ring_length=RING_LENGTH_M):
    """Compare recorded gap with the front-bumper gap from raw positions; report only."""
    lead = df[["veh_id", "k", "x", "length"]].rename(
        columns={"veh_id": "leader_id", "x": "x_l", "length": "len_l"})
    w = df.merge(lead, on=["leader_id", "k"], how="left", validate="many_to_one")
    gp = np.mod(w["x_l"] - w["x"] - w["len_l"], ring_length)
    diff = w["gap"] - gp
    return {"median_abs_diff_m": float(diff.abs().median()),
            "p99_abs_diff_m": float(diff.abs().quantile(0.99)),
            "max_abs_diff_m": float(diff.abs().max())}
