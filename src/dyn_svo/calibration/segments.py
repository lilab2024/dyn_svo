"""Cut follower-leader pairs into fixed calibration segments at the simulation step."""
import numpy as np

from dyn_svo.config import load_config

_SIM = load_config("sim")
STEP_FRAMES = int(round(_SIM["dt"] * _SIM["data_hz"]))


def vehicle_segments(pairs, veh_id, segment_s, step_frames=STEP_FRAMES, data_hz=None):
    """Non-overlapping segments for one follower.

    pairs: rows from loader.follower_pairs (one experiment); only consecutive frames are used.
    Returns dict of arrays: x_l, v_l, x, v, gap (S, N); len_l (S,); k_start (S,).
    N = segment frames / step_frames; frames are taken every step_frames (no interpolation).
    """
    data_hz = data_hz or _SIM["data_hz"]
    seg_frames = int(round(segment_s * data_hz))
    if seg_frames % step_frames:
        raise ValueError("segment length must be a multiple of the step")
    d = pairs[pairs["veh_id"] == veh_id].sort_values("k")
    k = d["k"].to_numpy()
    breaks = np.flatnonzero(np.diff(k) != 1) + 1
    runs = np.split(np.arange(len(k)), breaks)
    cols = {c: d[c].to_numpy(dtype=float) for c in ("x_l", "v_l", "x", "v", "gap", "len_l")}
    out = {c: [] for c in ("x_l", "v_l", "x", "v", "gap", "len_l", "k_start")}
    for run in runs:
        for i in range(len(run) // seg_frames):
            idx = run[i * seg_frames:(i + 1) * seg_frames:step_frames]
            for c in ("x_l", "v_l", "x", "v", "gap"):
                out[c].append(cols[c][idx])
            out["len_l"].append(cols["len_l"][idx[0]])
            out["k_start"].append(int(k[idx[0]]))
    n_step = seg_frames // step_frames
    res = {}
    for c in ("x_l", "v_l", "x", "v", "gap"):
        res[c] = np.array(out[c]).reshape(-1, n_step)
    res["len_l"] = np.array(out["len_l"], dtype=float)
    res["k_start"] = np.array(out["k_start"], dtype=int)
    return res


def stack(segment_dicts):
    """Concatenate segment dicts (e.g. one vehicle across experiments)."""
    keys = ("x_l", "v_l", "x", "v", "gap", "len_l", "k_start")
    return {c: np.concatenate([s[c] for s in segment_dicts], axis=0) for c in keys}


def n_segments(seg):
    return int(seg["x_l"].shape[0])
