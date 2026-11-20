"""Open-chain replay simulator.

The ring is cut at the AV's leader L: L replays its recorded trajectory; the AV slot p and the
N-2 vehicles behind it are simulated. Conventions: x = front bumper,
gap = x_leader - length_leader - x. Synchronous update, ballistic step.
"""
import numpy as np

from dyn_svo.config import load_config
from dyn_svo.sim import idm

_SIM = load_config("sim")
STEP_FRAMES = int(round(_SIM["dt"] * _SIM["data_hz"]))
DATA_HZ = int(_SIM["data_hz"])


def chain_order(df, av_id):
    """Return (leader_id, [av_id, follower, follower, ...]) covering N-1 vehicles."""
    lead = df.groupby("veh_id")["leader_id"].first().to_dict()
    follower = {l: f for f, l in lead.items()}
    leader = lead[av_id]
    order = [av_id]
    while follower[order[-1]] != leader:
        order.append(follower[order[-1]])
        if len(order) > len(lead):
            raise RuntimeError("Ring order is not closed")
    if len(order) != len(lead) - 1:
        raise RuntimeError("Chain does not cover N-1 vehicles")
    return int(leader), [int(v) for v in order]


MAX_ROLLBACK_M = 5.0   # larger backward steps mean wrapped positions, not measurement error


def monotone_leader(x, v):
    """R5: non-receding leader. Position = running maximum; speed >= 0 and 0 while held."""
    x = np.asarray(x, dtype=float)
    v = np.asarray(v, dtype=float)
    xm = np.maximum.accumulate(x)
    if np.max(xm - x) > MAX_ROLLBACK_M:
        raise ValueError("leader rollback above MAX_ROLLBACK_M: positions are not unwrapped")
    held = x < xm
    return xm, np.where(held, 0.0, np.maximum(v, 0.0))


def build_episode(df, av_id, t_start_s, t_end_s=None, leader_monotone=False):
    """Replay leader, initial chain state and real chain trajectories at the simulation step.

    df: one experiment from loader.load_experiment. Frames from round(t_start_s * 30) to the
    end (or t_end_s), every STEP_FRAMES frames. leader_monotone: R5 variant; False leaves the
    replayed leader exactly as recorded (primary analysis).
    """
    leader, order = chain_order(df, av_id)
    k_all = np.sort(df["k"].unique())
    k0 = int(round(t_start_s * DATA_HZ))
    k1 = int(k_all[-1]) if t_end_s is None else int(round(t_end_s * DATA_HZ))
    ks = np.arange(k0, k1 + 1, STEP_FRAMES)
    piv = {c: df.pivot(index="k", columns="veh_id", values=c).loc[ks] for c in ("x", "v", "gap")}
    length = df.groupby("veh_id")["length"].first()
    lengths = np.array([length[v] for v in order], dtype=float)
    len_l = float(length[leader])
    gap0 = piv["gap"].iloc[0][order].to_numpy(dtype=float)
    x0 = np.empty(len(order))
    front, front_len = float(piv["x"][leader].iloc[0]), len_l
    for i in range(len(order)):
        x0[i] = front - front_len - gap0[i]
        front, front_len = x0[i], lengths[i]
    x_l = piv["x"][leader].to_numpy(dtype=float)
    v_l = piv["v"][leader].to_numpy(dtype=float)
    if leader_monotone:
        x_l, v_l = monotone_leader(x_l, v_l)
    return {"exp": str(df["exp"].iloc[0]), "regime": str(df["regime"].iloc[0]), "av_id": int(av_id),
            "leader_id": leader, "order": order, "lengths": lengths, "len_l": len_l,
            "k": ks, "t": ks / DATA_HZ,
            "x_l": x_l, "v_l": v_l,
            "x0": x0, "v0": piv["v"].iloc[0][order].to_numpy(dtype=float),
            "real_v": piv["v"][order].to_numpy(dtype=float),
            "real_gap": piv["gap"][order].to_numpy(dtype=float)}


def episode_rng(seed, exp, av_id):
    """Noise stream for (seed, experiment, AV position): identical in every compared arm."""
    return np.random.default_rng([int(seed), ord(str(exp)[0]), int(av_id)])


class HumanDriver:
    """Default AV-slot driver: the vehicle's own IDM (all-HDV run). Receives human noise."""

    human = True

    def __init__(self, params):
        self.p = dict(params)

    def reset(self, info):
        pass

    def act(self, obs):
        return float(idm.idm_accel(obs["v"], obs["gap"], obs["v"] - obs["v_leader"], **self.p))


def simulate(ep, params_list, av_driver=None, n_rear=5, dt=None, noise_sd=0.0, rng=None):
    """Run the chain. params_list[i] is the IDM parameter dict of ep['order'][i].

    av_driver: None -> slot 0 uses its IDM like everyone else; otherwise an object with
    reset(info) and act(obs) -> acceleration command (floored at the physical limit).
    noise_sd > 0: i.i.d. Gaussian acceleration noise on every human vehicle. One draw per
    vehicle and step is always taken, so the stream is identical whatever drives slot 0
    (common random numbers); a non-human slot-0 driver ignores its draw.
    Returns x, v, a, gap (T, N), the noise draws and collision information; the run continues
    after a collision.
    """
    dt = idm.DT if dt is None else dt
    n_veh, n_step = len(ep["order"]), len(ep["t"])
    P = {k: np.array([p[k] for p in params_list], dtype=float) for k in idm.PARAM_NAMES}
    lead_len = np.concatenate([[ep["len_l"]], ep["lengths"][:-1]])
    x = np.empty((n_step, n_veh))
    v = np.empty((n_step, n_veh))
    a = np.empty((n_step, n_veh))
    gap = np.empty((n_step, n_veh))
    eps = np.zeros((n_step, n_veh))
    if noise_sd > 0.0:
        if rng is None:
            raise ValueError("noise_sd > 0 needs an rng (use episode_rng)")
        eps = rng.standard_normal((n_step, n_veh)) * float(noise_sd)
    slot0_human = av_driver is None or getattr(av_driver, "human", False)
    x[0], v[0] = ep["x0"], ep["v0"]
    if av_driver is not None:
        av_driver.reset({k: ep[k] for k in ("exp", "regime", "av_id", "leader_id", "order")})
    for n in range(n_step):
        lead_x = np.concatenate([[ep["x_l"][n]], x[n, :-1]])
        lead_v = np.concatenate([[ep["v_l"][n]], v[n, :-1]])
        gap[n] = lead_x - lead_len - x[n]
        acc = idm.idm_accel(v[n], gap[n], v[n] - lead_v, P["v0"], P["T"], P["s0"], P["a_max"], P["b"])
        acc = np.maximum(acc + eps[n], idm.ACCEL_FLOOR)
        if av_driver is not None:
            obs = {"t": float(ep["t"][n]), "v": float(v[n, 0]), "gap": float(gap[n, 0]),
                   "v_leader": float(lead_v[0]), "rear_v": v[n, 1:1 + n_rear].copy(),
                   "rear_gap": gap[n, 1:1 + n_rear].copy()}
            cmd = float(av_driver.act(obs)) + (float(eps[n, 0]) if slot0_human else 0.0)
            acc[0] = max(cmd, idm.ACCEL_FLOOR)
        a[n] = acc
        if n + 1 < n_step:
            x[n + 1], v[n + 1] = idm.ballistic_step(x[n], v[n], acc, dt)
    hit = gap <= 0.0
    first = np.where(hit.any(axis=0), hit.argmax(axis=0), -1)
    return {"x": x, "v": v, "a": a, "gap": gap, "noise": eps, "collided": hit.any(axis=0),
            "first_collision_step": first}
