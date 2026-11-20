"""Episode evaluation tasks: one task = (configuration, experiment), all AV positions x seeds
. Raw statistics are stored; FOR, eligibility and selection are computed by summary.py."""
import numpy as np

from dyn_svo.calibration import noise as nz
from dyn_svo.calibration import params as cp
from dyn_svo.config import load_config
from dyn_svo.control import cooperation as co
from dyn_svo.control import mpc as M
from dyn_svo.control import platform as pf
from dyn_svo.data import loader as ld
from dyn_svo.data import split as sp
from dyn_svo.experiments import configs as cf
from dyn_svo.metrics import evaluation as ev
from dyn_svo.metrics import oscillation as osc
from dyn_svo.sim import chain

CONFIGS = ("sim", "data_split", "data_manifest", "metrics", "calibration", "av", "baselines",
           "cooperation", "mpc", "eval", "variants")


def variant_names():
    return list(load_config("variants"))


def variant_settings(name):
    """Settings of an analysis variant; primary = no change."""
    cfg = load_config("variants")
    if name not in cfg:
        raise ValueError("unknown variant " + repr(name) + "; known: " + ", ".join(cfg))
    return dict(cfg[name] or {})


def enumerate_tasks(exps, variant="primary"):
    return [{"task_id": c["name"] + "__" + e, "config": c, "exp": e, "variant": variant}
            for c in cf.all_configs() for e in exps]


def build_context(exps, load=None, split=None, allow_test=False):
    """Data and fixed inputs shared by all tasks (loaded once per shard). allow_test only from
    scripts/evaluate.py --stage test."""
    load = load or (lambda e: ld.load_experiment(e, allow_test=allow_test))
    split = split if split is not None else sp.load_split()
    table = cp.load_table()
    ctx = {"data": {e: load(e) for e in exps}, "table": table, "noise": nz.load_noise_table(),
           "thresholds": pf.load_thresholds(), "policy": M.load_policy(), "phi_thr": co.load_phi_thresholds(),
           "t_start": float(split["segments"]["drop_first_s"]), "folds": {}}
    for e in exps:
        ctx["folds"][e] = cp.fold_for_experiment(e, split)
    return ctx


class Recording:
    """Wraps an AV driver and records the controller's phi at every step."""

    human = False

    def __init__(self, platform):
        self.p = platform
        self.phis = []

    def reset(self, info):
        self.p.reset(info)
        self.phis = []

    def act(self, obs):
        a = self.p.act(obs)
        self.phis.append(getattr(self.p.ctrl, "last_phi", None))
        return a


def episode_row(ep, sim, rec, n0):
    st = ev.av_window_stats(ep, sim)
    row = {"av": int(ep["av_id"]), "follower_sigma": [float(x) for x in osc.sigma_band(sim["v"][:, 1:])],
           "av_sigma_v": float(osc.sigma_band(sim["v"][:, 0])),
           "av_gap_std": float(np.std(sim["gap"][n0:, 0])), **st,
           "chain_mean_speed": float(np.mean(sim["v"][n0:, :])),
           "n_collided": int(np.sum(sim["collided"]))}
    if rec is not None:
        row.update(rec.p.stats())
        phis = [p for p in rec.phis[n0:] if p is not None]
        row.update(phi_mean=float(np.mean(phis)) if phis else None, phi_std=float(np.std(phis)) if phis else None)
    return row


def run_task(task, ctx, cfg=None):
    """Simulate every AV position x seed of one experiment with one configuration."""
    ecfg = cfg or load_config("eval")
    e, c = task["exp"], task["config"]
    variant = task.get("variant", "primary")
    monotone = bool(variant_settings(variant).get("leader_monotone", False))
    df = ctx["data"][e]
    regime, fold = str(df["regime"].iloc[0]), ctx["folds"][e]
    sd = nz.noise_sd(ctx["noise"], fold, regime)
    pooled = cp.vehicle_params(ctx["table"], fold, regime, [-1])[-1][0]
    ctx_e = dict(ctx, pooled={regime: pooled})
    if "phi_thr" in ctx:                        # v5 phi_t: thresholds of this fold only
        ctx_e["phi_thr_e"] = {regime: ctx["phi_thr"][fold][regime]}
    n0 = int(round(osc.WARMUP_S / osc.DT))
    rows, n_pooled = [], 0
    for av in sorted(int(v) for v in df["veh_id"].unique()):
        ep = chain.build_episode(df, av, t_start_s=ctx["t_start"], leader_monotone=monotone)
        pv = cp.vehicle_params(ctx["table"], fold, regime, ep["order"])
        n_pooled += sum(src == "pooled" for _, src in pv.values())
        plist = [pv[v][0] for v in ep["order"]]
        for seed in range(int(ecfg["seeds"])):
            drv = cf.make_driver(c, regime, ctx_e)
            rec = None if drv is None else Recording(drv)
            sim = chain.simulate(ep, plist, av_driver=rec, n_rear=int(ecfg["n_rear"]), noise_sd=sd,
                                 rng=chain.episode_rng(seed, e, av))
            row = episode_row(ep, sim, rec, n0)
            row["seed"] = seed
            rows.append(row)
    return {"exp": e, "regime": regime, "fold": fold, "noise_sd": sd, "n_pooled_slots": n_pooled,
            "variant": variant, "rows": rows}
