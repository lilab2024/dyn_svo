"""Evaluation of RL policies: one task = (configuration, held-out experiment k, training seed).

The policy trained on loeo_k runs through sim.chain.simulate on every AV position of k x noise seeds, with the
loeo_k human parameters, exactly as the MPC evaluation (experiments.runner). Mode "criterion" (tuning) stores
only the mean reward per step from t = 60 s; mode "full" stores the MPC evaluation row plus that reward.
Mode "full" also runs the all-HDV baseline (tasks hdv__<exp>, first in the list) through experiments.runner at
the same commit, so both sides of FOR come from one run.
"""
import json
from pathlib import Path

import numpy as np

from dyn_svo.calibration import noise as nz
from dyn_svo.calibration import params as cp
from dyn_svo.config import load_config
from dyn_svo.experiments import configs as cf
from dyn_svo.experiments import runner as erun
from dyn_svo.metrics import oscillation as osc
from dyn_svo.rl import env as E
from dyn_svo.rl import runner as R
from dyn_svo.rl.policy import PolicyWeights, make_rl_driver
from dyn_svo.sim import chain

MODES = ("criterion", "full")
HDV = "hdv"


class RewardRecording(erun.Recording):
    """erun.Recording plus U and the actual acceleration of every step (for the reward)."""

    def reset(self, info):
        super().reset(info)
        self.Us, self.acts = [], []

    def act(self, obs):
        a = self.p.act(obs)
        self.phis.append(self.p.ctrl.last_phi)
        self.Us.append(self.p.ctrl.last_U)
        self.acts.append(a)
        return a


def step_rewards(ep, sim, rec, pol):
    """Reward of every step k = 0..n-2 as in the training environment: -J of the state after the step."""
    n = len(ep["t"])
    out = np.empty(n - 1)
    for k in range(n - 1):
        o = {"v": float(sim["v"][k + 1, 0]), "gap": float(sim["gap"][k + 1, 0]), "v_leader": float(ep["v_l"][k + 1])}
        out[k] = -E.step_cost(o, rec.acts[k], rec.Us[k], rec.phis[k], pol)
    return out


def eval_tasks(stage, dev, mode, variant="primary", rl=None, phi_match=None):
    """One evaluation task per out-of-fold training task (training sets loeo_k only); mode full adds one all-HDV
    task per dev experiment at the start of the list."""
    if mode not in MODES:
        raise ValueError("mode must be one of " + ", ".join(MODES))
    out = []
    if mode == "full":
        out += [{"task_id": HDV + "__" + e, "config": HDV, "exp": e, "variant": variant, "mode": mode} for e in dev]
    for t in R.enumerate_tasks(stage, dev, rl, phi_match=phi_match):
        if t["train_set"] == "all_dev":
            continue
        k = t["train_set"][len("loeo_"):]
        out.append({"task_id": "%s__%s__s%d" % (t["config"], k, t["seed"]), "config": t["config"], "exp": k,
                    "train_seed": t["seed"], "train_task_id": t["task_id"], "phi_cfg": t["phi_cfg"],
                    "variant": variant, "mode": mode})
    return out


def load_policy(train_dir, train_task_id):
    """Weights of a finished training task; the npz md5 must equal the training record."""
    train_dir = Path(train_dir)
    rec = json.loads((train_dir / "tasks" / (train_task_id + ".json")).read_text(encoding="utf-8"))
    path = train_dir / rec["result"]["policy_file"]
    if R.md5_file(path) != rec["result"]["policy_md5"]:
        raise RuntimeError("policy md5 differs from the training record: " + path.name)
    return PolicyWeights.from_npz(path), rec


def run_hdv_task(task, ctx, cfg=None):
    """All-HDV baseline of one experiment, computed exactly as in the MPC evaluation (experiments.runner)."""
    if task["mode"] != "full":
        raise RuntimeError("all-HDV tasks belong to mode full only")
    etask = {"task_id": task["task_id"], "config": cf.HDV, "exp": task["exp"], "variant": task["variant"]}
    return dict(erun.run_task(etask, ctx, cfg), mode=task["mode"])


def run_task(task, ctx, cfg=None):
    if task["config"] == HDV:
        return run_hdv_task(task, ctx, cfg)
    ecfg = cfg or load_config("eval")
    e = task["exp"]
    monotone = bool(erun.variant_settings(task["variant"]).get("leader_monotone", False))
    df = ctx["data"][e]
    regime, fold = str(df["regime"].iloc[0]), ctx["folds"][e]
    if fold != "loeo_" + e:
        raise RuntimeError("experiment " + e + " is not evaluated with its own fold")
    w, trec = load_policy(ctx["train_dir"], task["train_task_id"])
    sd = nz.noise_sd(ctx["noise"], fold, regime)
    pol, s_min = ctx["policy"][regime], float(ctx["thresholds"][regime]["min_gap"])
    slot = {"phi_thr": ctx["phi_thr"][fold][regime]}
    amap = load_config("rl")["action_map"]                        # (plan v7)
    n0 = int(round(osc.WARMUP_S / osc.DT))
    rows = []
    for av in sorted(int(v) for v in df["veh_id"].unique()):
        ep = chain.build_episode(df, av, t_start_s=ctx["t_start"], leader_monotone=monotone)
        pv = cp.vehicle_params(ctx["table"], fold, regime, ep["order"])
        plist = [pv[v][0] for v in ep["order"]]
        for seed in range(int(ecfg["seeds"])):
            rec = RewardRecording(make_rl_driver(w, pol, E.make_phi_source(task["phi_cfg"], slot), s_min, amap))
            sim = chain.simulate(ep, plist, av_driver=rec, n_rear=int(ecfg["n_rear"]), noise_sd=sd,
                                 rng=chain.episode_rng(seed, e, av))
            reward = float(np.mean(step_rewards(ep, sim, rec, pol)[n0:]))
            if task["mode"] == "criterion":
                row = {"av": av, "seed": seed, "mean_reward": reward}
            else:
                row = erun.episode_row(ep, sim, rec, n0)
                row.update(seed=seed, mean_reward=reward)
            row["train_seed"] = task["train_seed"]
            rows.append(row)
    return {"exp": e, "regime": regime, "fold": fold, "noise_sd": sd, "variant": task["variant"], "mode": task["mode"],
            "train_task_id": task["train_task_id"], "train_commit": trec["provenance"]["commit"],
            "policy_md5": trec["result"]["policy_md5"], "action_map": amap, "rows": rows}


def tuning_table(results):
    """Criterion per candidate from criterion-mode results: per fold the mean over all runs
    (AV positions x noise seeds x training seeds), then the mean over folds."""
    per = {}
    for r in results:
        cfg = r["task"]["config"]
        for row in r["result"]["rows"]:
            per.setdefault(cfg, {}).setdefault(r["result"]["exp"], []).append(row["mean_reward"])
    table = []
    for cfg, folds in per.items():
        fm = {e: float(np.mean(v)) for e, v in sorted(folds.items())}
        lr = float(cfg.split("_lr")[1].split("_g")[0])
        g = float(cfg.split("_g")[-1])
        table.append({"config": cfg, "learning_rate": lr, "gamma": g, "per_fold": fm,
                      "n_runs": {e: len(v) for e, v in sorted(folds.items())}, "criterion": float(np.mean(list(fm.values())))})
    return sorted(table, key=lambda t: (t["learning_rate"], t["gamma"]))


def select_hp(table):
    """Largest criterion; a tie (equal to 6 significant digits) goes to the smaller learning rate, then to the
    gamma closest to 0.99."""
    key = lambda t: (-float("%.6g" % t["criterion"]), t["learning_rate"], abs(t["gamma"] - 0.99))
    best = sorted(table, key=key)[0]
    return {"learning_rate": best["learning_rate"], "gamma": best["gamma"], "config": best["config"]}
