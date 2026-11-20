"""Per-vehicle human-model parameters from a calibration run."""
import pandas as pd

from dyn_svo.data import manifest as mf
from dyn_svo.data import split as sp
from dyn_svo.sim import idm

DEFAULT_RUN = "idm_v1"
DEFAULT_VARIANT = "nodelay"


def load_table(run_name=DEFAULT_RUN, root=None):
    root = mf.data_root() if root is None else root
    path = root / "dyn_svo" / "results" / "calibration" / run_name / "params.csv"
    return pd.read_csv(path, dtype={"veh": str})


def fold_for_experiment(exp, split=None):
    """Dev experiment k -> 'loeo_k' (k never used in calibration); test experiment -> 'all_dev'."""
    split = split if split is not None else sp.load_split()
    if exp in split["roles"]["dev"]:
        return "loeo_" + exp
    if exp in split["roles"]["test"]:
        return "all_dev"
    raise KeyError("Unknown experiment: " + str(exp))


def vehicle_params(table, fold, regime, veh_ids, variant=DEFAULT_VARIANT):
    """Return {veh_id: (params dict, source)}; source is 'vehicle' or 'pooled' (fallback)."""
    t = table[(table["fold"] == fold) & (table["regime"] == regime) & (table["variant"] == variant)]
    pooled = t[t["veh"] == "pooled"]
    if len(pooled) != 1:
        raise KeyError("Need exactly one pooled row for " + fold + "/" + regime + "/" + variant)
    pooled_p = {k: float(pooled.iloc[0][k]) for k in idm.PARAM_NAMES}
    out = {}
    for v in veh_ids:
        row = t[t["veh"] == str(int(v))]
        if len(row) > 1:
            raise KeyError("Duplicate parameters for vehicle " + str(v))
        if len(row) == 1:
            out[int(v)] = ({k: float(row.iloc[0][k]) for k in idm.PARAM_NAMES}, "vehicle")
        else:
            out[int(v)] = (dict(pooled_p), "pooled")
    return out
