"""Evaluated configurations and their AV drivers (-)."""
from dyn_svo.config import load_config
from dyn_svo.control import baselines as bl
from dyn_svo.control import cooperation as co
from dyn_svo.control import mpc as M
from dyn_svo.control.platform import AVPlatform

HDV = {"name": "hdv", "family": "hdv", "rank": [0]}


def av_configs():
    """20 AV configurations: 9 fixed phi, 9 dynamic (v5: W_a x phi range), FollowerStopper, PI."""
    out = [{"name": "fixed_%d" % i, "family": "mpc_fixed", "phi": phi, "rank": [i]}
           for i, phi in enumerate(M.fixed_phi_grid())]
    for w, lo, hi in co.leader_activity_grid():
        name = "dyn_W%s_lo%s_hi%s" % (co.window_key(w), co.window_key(lo).replace(".", "p"),
                                      co.window_key(hi).replace(".", "p"))
        out.append({"name": name, "family": "mpc_dynamic", "window_s": w, "phi_lo_pi16": lo,
                    "phi_hi_pi16": hi, "rank": [w, hi - lo]})       # ties: shorter window, narrower range
    out.append({"name": "fs", "family": "fs", "rank": [0]})
    out.append({"name": "pi", "family": "pi", "rank": [0]})
    return out


def all_configs():
    return [HDV] + av_configs()


def make_driver(cfg, regime, ctx):
    """AV-slot driver for one episode; None = the slot's own human model (all-HDV)."""
    fam = cfg["family"]
    if fam == "hdv":
        return None
    if fam == "fs":
        ctrl = bl.FollowerStopper()
    elif fam == "pi":
        ctrl = bl.PISaturation()
    else:
        if fam in ("mpc_fixed", "matched"):          # matched = fixed phi at a dynamic arm mean
            src = M.FixedPhi(cfg["phi"])
        else:                                   # v5 phi_t with the thresholds of the episode's fold
            thr = ctx["phi_thr_e"][regime][co.window_key(cfg["window_s"])]
            src = co.LeaderActivityWeight(cfg["window_s"], cfg["phi_lo_pi16"] * co.PI16,
                                          cfg["phi_hi_pi16"] * co.PI16, thr["a_lo"], thr["a_hi"])
        ctrl = M.MPC(ctx["policy"][regime], src)
    return AVPlatform(ctrl, s_min=ctx["thresholds"][regime]["min_gap"])
