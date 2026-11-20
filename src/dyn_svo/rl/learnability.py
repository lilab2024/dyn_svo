"""Learnability check of RL policies: does a policy follow its leader?

On every AV position of the held-out experiment, without noise and with deterministic actions: the ratio of
mean AV speed to mean leader speed from t = 60 s, the last gap, and the mean reward per step from t = 60 s
against an AV that never accelerates. No FOR and no other guardrail statistic.
"""
import numpy as np

from dyn_svo.calibration import params as cp
from dyn_svo.metrics import oscillation as osc
from dyn_svo.rl import env as E
from dyn_svo.rl import evaluate as V
from dyn_svo.rl.policy import make_rl_driver
from dyn_svo.sim import chain


class ConstantAction:
    """Policy stand-in that always returns the action of zero acceleration."""

    def __init__(self, action_map):
        self.u = np.array([0.0 if action_map == "centred" else (0.0 - E.A_LO) / (E.A_HI - E.A_LO) * 2.0 - 1.0])

    def action(self, obs):
        return self.u


def follow_check(w, action_map, ctx, exp, fold, phi_cfg):
    df = ctx["data"][exp]
    regime = str(df["regime"].iloc[0])
    pol, s_min = ctx["policy"][regime], float(ctx["thresholds"][regime]["min_gap"])
    slot = {"phi_thr": ctx["phi_thr"][fold][regime]}
    n0 = int(round(osc.WARMUP_S / osc.DT))
    rows = []
    for av in sorted(int(v) for v in df["veh_id"].unique()):
        ep = chain.build_episode(df, av, t_start_s=ctx["t_start"])
        pvp = cp.vehicle_params(ctx["table"], fold, regime, ep["order"])
        plist = [pvp[v][0] for v in ep["order"]]
        row = {"av": av}
        for name, weights in (("policy", w), ("zero", ConstantAction(action_map))):
            rec = V.RewardRecording(make_rl_driver(weights, pol, E.make_phi_source(phi_cfg, slot), s_min, action_map))
            sim = chain.simulate(ep, plist, av_driver=rec, n_rear=8, noise_sd=0.0)
            row[name + "_mean_reward"] = float(np.mean(V.step_rewards(ep, sim, rec, pol)[n0:]))
            if name == "policy":
                row["speed_ratio"] = float(np.mean(sim["v"][n0:, 0]) / np.mean(ep["v_l"][n0:]))
                row["last_gap"] = float(sim["gap"][-1, 0])
        rows.append(row)
    return rows, float(ctx["thresholds"][regime]["mean_gap"])


def g4(rows, mean_gap_p99):
    """Mean speed ratio >= 0.95, every last gap <= 2 x human p99 mean gap, mean reward >= zero-acceleration AV."""
    sr = float(np.mean([r["speed_ratio"] for r in rows]))
    lg = max(r["last_gap"] for r in rows)
    pr = float(np.mean([r["policy_mean_reward"] for r in rows]))
    zr = float(np.mean([r["zero_mean_reward"] for r in rows]))
    return {"mean_speed_ratio": sr, "max_last_gap": lg, "gap_limit": 2.0 * mean_gap_p99, "policy_mean_reward": pr,
            "zero_mean_reward": zr, "pass": bool(sr >= 0.95 and lg <= 2.0 * mean_gap_p99 and pr >= zr)}


def gate_summary(results, selected, n_selected=15):
    """ the gate passes only if all n_selected policies of the selected candidate pass;
    the other candidates are described by their pass counts."""
    by = {}
    for r in results:
        by.setdefault(r["config"], []).append(bool(r["g4"]["pass"]))
    sel = by.get(selected, [])
    return {"selected": selected, "n_selected": len(sel), "selected_pass": sum(sel),
            "gate_pass": len(sel) == n_selected and all(sel),
            "pass_counts": {c: [sum(v), len(v)] for c, v in sorted(by.items())},
            "pass_share_all": sum(sum(v) for v in by.values()) / max(1, sum(len(v) for v in by.values()))}


def train_summary(results):
    """ G4 of the out-of-fold policies of the main training, descriptive only (no gate): pass
    counts per configuration and per (configuration, experiment), overall share, and every policy that fails."""
    by, by_exp, fails = {}, {}, []
    for r in results:
        ok = bool(r["g4"]["pass"])
        by.setdefault(r["config"], []).append(ok)
        by_exp.setdefault(r["config"] + "|" + r["exp"], []).append(ok)
        if not ok:
            fails.append({"config": r["config"], "exp": r["exp"], "seed": r["seed"],
                          "mean_speed_ratio": r["g4"]["mean_speed_ratio"], "max_last_gap": r["g4"]["max_last_gap"],
                          "gap_limit": r["g4"]["gap_limit"], "reward_not_below_zero_av": r["g4"]["policy_mean_reward"] >= r["g4"]["zero_mean_reward"]})
    n = sum(len(v) for v in by.values())
    k = sum(sum(v) for v in by.values())
    return {"stage": "train", "n_policies": n, "n_pass": k, "pass_share_all": k / max(1, n),
            "pass_counts": {c: [sum(v), len(v)] for c, v in sorted(by.items())},
            "pass_counts_by_experiment": {c: [sum(v), len(v)] for c, v in sorted(by_exp.items())},
            "not_passing": sorted(fails, key=lambda f: (f["config"], f["exp"], f["seed"]))}
