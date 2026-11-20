"""Diagnostic debug runs of RL-MLP learnability. One job = (variant, seed): train on one training set
with fixed phi = pi/4, then check on its held-out experiment whether the policy follows the leader (no noise,
deterministic actions; no FOR). Variants: V0 = as is; V1 = centred action mapping; V2 = V1 + reward
normalization; V3 = V2 + observation normalization.

  python scripts/rl_diagnose.py --run-name debug_rl_diag_v1 --variant V1 --seed 0
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dyn_svo import provenance as pv
from dyn_svo.calibration import params as cp
from dyn_svo.control import cooperation as co
from dyn_svo.data import manifest as mf
from dyn_svo.data import split as sp
from dyn_svo.experiments import runner as erun
from dyn_svo.metrics import oscillation as osc
from dyn_svo.rl import env as E
from dyn_svo.rl import evaluate as V
from dyn_svo.rl import runner as R
from dyn_svo.rl import train as T
from dyn_svo.rl.policy import make_rl_driver
from dyn_svo.sim import chain

VARIANTS = {"V0": ("linear", "none"), "V1": ("centred", "none"), "V2": ("centred", "reward"), "V3": ("centred", "reward_obs")}
LR, GAMMA, PHI = 3e-4, 0.99, 4 * co.PI16


class ConstantAction:
    """Policy stand-in that always returns the action of zero acceleration (baseline)."""

    def __init__(self, action_map):
        self.u = np.array([0.0 if action_map == "centred" else (0.0 - E.A_LO) / (E.A_HI - E.A_LO) * 2.0 - 1.0])

    def action(self, obs):
        return self.u


def follow_check(w, action_map, ctx, exp, fold):
    """Per AV position of the held-out experiment, without noise: speed ratio, last gap, mean reward from t = 60 s,
    and the mean reward of the zero-acceleration baseline."""
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
            rec = V.RewardRecording(make_rl_driver(weights, pol, E.make_phi_source({"phi": PHI}, slot), s_min, action_map))
            sim = chain.simulate(ep, plist, av_driver=rec, n_rear=8, noise_sd=0.0)
            r = V.step_rewards(ep, sim, rec, pol)[n0:]
            row[name + "_mean_reward"] = float(np.mean(r))
            if name == "policy":
                row["speed_ratio"] = float(np.mean(sim["v"][n0:, 0]) / np.mean(ep["v_l"][n0:]))
                row["last_gap"] = float(sim["gap"][-1, 0])
        rows.append(row)
    return rows, float(ctx["thresholds"][regime]["mean_gap"])


def g4(rows, mean_gap_p99):
    """ mean speed ratio >= 0.95, every last gap <= 2 x human p99 mean gap, mean reward not below zero baseline."""
    sr = float(np.mean([r["speed_ratio"] for r in rows]))
    lg = max(r["last_gap"] for r in rows)
    pr = float(np.mean([r["policy_mean_reward"] for r in rows]))
    zr = float(np.mean([r["zero_mean_reward"] for r in rows]))
    return {"mean_speed_ratio": sr, "max_last_gap": lg, "gap_limit": 2.0 * mean_gap_p99, "policy_mean_reward": pr,
            "zero_mean_reward": zr, "pass": bool(sr >= 0.95 and lg <= 2.0 * mean_gap_p99 and pr >= zr)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--held-out", default="A")
    ap.add_argument("--steps", type=int, default=300000)
    args = ap.parse_args(argv)
    if not args.run_name.startswith("debug_"):
        raise SystemExit("Diagnostic runs are debug runs; the name must start with debug_.")
    out = mf.data_root() / "dyn_svo" / "results" / "rl" / args.run_name / ("%s_s%d" % (args.variant, args.seed))
    if out.exists():
        raise SystemExit("Output exists, refusing to overwrite: " + str(out))
    prov = pv.provenance(R.CONFIGS, mf.verify_data("size"))
    prov["rl_packages"] = R.rl_versions()
    pv.require_clean(prov)
    dev = sp.experiments("dev")
    ctx = erun.build_context(dev)
    fold = "loeo_" + args.held_out
    exps = [e for e in dev if e != args.held_out]
    slots = E.build_slots({e: ctx["data"][e] for e in exps}, fold, ctx["table"], ctx["noise"], ctx["thresholds"],
                          ctx["policy"], ctx["phi_thr"], ctx["t_start"])
    action_map, normalize = VARIANTS[args.variant]
    t0 = time.time()
    w, rec, _ = T.train_policy(slots, {"phi": PHI}, args.seed, args.steps, LR, GAMMA, action_map=action_map, normalize=normalize)
    rows, p99 = follow_check(w, action_map, ctx, args.held_out, fold)
    out.mkdir(parents=True)
    w.to_npz(out / "policy.npz")
    log = rec.pop("episode_log")
    with open(out / "episodes.csv", "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(log[0]) if log else ["exp"])
        wr.writeheader()
        wr.writerows(log)
    res = {"variant": args.variant, "action_map": action_map, "normalize": normalize, "seed": args.seed, "train_set": fold,
           "held_out": args.held_out, "steps": args.steps, "learning_rate": LR, "gamma": GAMMA, "phi": PHI,
           "train_record": rec, "n_episodes": len(log), "follow_rows": rows, "g4": g4(rows, p99),
           "wall_s": time.time() - t0, "provenance": prov}
    (out / "result.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(args.variant, "seed", args.seed, "| G4", res["g4"], flush=True)


if __name__ == "__main__":
    main()
