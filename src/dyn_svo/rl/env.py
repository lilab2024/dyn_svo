"""Gymnasium training environment for RL-MLP.

One episode = one window of one training experiment with one AV position. Windows are drawn uniformly over
all (experiment, AV position, start) combinations that fit the drawn length. Humans: IDM of the
fold plus acceleration noise of the fold (fold-pure); AV: the shared platform.
Reward of a step = -J of the state after the step, with the actual AV acceleration.
"""
import math

import gymnasium as gym
import numpy as np

from dyn_svo.calibration import noise as nz
from dyn_svo.calibration import params as cp
from dyn_svo.config import load_config
from dyn_svo.control import cooperation as co
from dyn_svo.control.estimator import MeanSpeedEstimator
from dyn_svo.control.mpc import FixedPhi
from dyn_svo.control.platform import AVPlatform
from dyn_svo.rl.stepper import ChainStepper, slice_episode
from dyn_svo.sim import chain, idm

_AV = load_config("av")
A_LO, A_HI = float(_AV["accel_min"]), float(_AV["accel_max"])
OBS_DIM = 10


def build_slots(data, fold, table, noise_table, thresholds, policy, phi_thr, t_start):
    """Every (training experiment, AV position) with its full episode from t_start and the fold's settings."""
    slots = []
    for e in sorted(data):
        df = data[e]
        regime = str(df["regime"].iloc[0])
        sd = nz.noise_sd(noise_table, fold, regime)
        for av in sorted(int(v) for v in df["veh_id"].unique()):
            ep = chain.build_episode(df, av, t_start_s=t_start)
            pv = cp.vehicle_params(table, fold, regime, ep["order"])
            slots.append({"exp": e, "av": av, "regime": regime, "ep": ep, "params": [pv[v][0] for v in ep["order"]],
                          "noise_sd": sd, "policy": policy[regime], "s_min": float(thresholds[regime]["min_gap"]),
                          "phi_thr": phi_thr[fold][regime]})
    return slots


def window_weights(slots, n_points):
    """Number of start positions of a window with n_points steps in every slot (0 if it does not fit)."""
    return np.array([max(0, len(s["ep"]["t"]) - n_points + 1) for s in slots], dtype=float)


def make_phi_source(phi_cfg, slot):
    if "phi" in phi_cfg:
        return FixedPhi(phi_cfg["phi"])
    thr = slot["phi_thr"][co.window_key(phi_cfg["window_s"])]
    return co.LeaderActivityWeight(phi_cfg["window_s"], phi_cfg["phi_lo_pi16"] * co.PI16,
                                   phi_cfg["phi_hi_pi16"] * co.PI16, thr["a_lo"], thr["a_hi"])


def action_to_accel(action, mapping="linear"):
    """linear: [-1, 1] -> [A_LO, A_HI], action 0 = -0.75 m/s^2; centred (diagnostic V1):
    [-1, 0] -> [A_LO, 0] and [0, 1] -> [0, A_HI], action 0 = no acceleration."""
    u = float(np.clip(np.asarray(action, dtype=float).reshape(-1)[0], -1.0, 1.0))
    if mapping == "linear":
        return A_LO + (u + 1.0) * 0.5 * (A_HI - A_LO)
    if mapping == "centred":
        return -u * A_LO if u < 0.0 else u * A_HI
    raise ValueError("unknown action mapping " + str(mapping))


def obs_vector(o, U, phi, a_cmd_prev, a_act, pol):
    v, s, vl = o["v"], o["gap"], o["v_leader"]
    return np.array([v / 10.0, s / 20.0, vl / 10.0, U / 10.0,
                     (s - pol["s0"] - pol["h"] * v) / pol["sigma_gap"], (v - vl) / pol["sigma_dv"],
                     (v - U) / pol["sigma_u"], a_cmd_prev / pol["sigma_a"], a_act / pol["sigma_a"],
                     phi / (math.pi / 2)], dtype=np.float32)


def step_cost(o_next, a_act, U, phi, pol):
    """J of / for one step: state after the step, actual acceleration, U and phi at decision time."""
    v, s, vl = o_next["v"], o_next["gap"], o_next["v_leader"]
    ego = ((s - pol["s0"] - pol["h"] * v) / pol["sigma_gap"]) ** 2 + ((v - vl) / pol["sigma_dv"]) ** 2
    soc = (a_act / pol["sigma_a"]) ** 2 + ((v - U) / pol["sigma_u"]) ** 2
    return math.cos(phi) * ego + math.sin(phi) * soc


class _ActionController:
    """Controller stub inside the platform: returns the acceleration chosen by the agent."""

    def __init__(self):
        self.a = 0.0

    def reset(self, info):
        pass

    def act(self, obs):
        return {"a": self.a}


class ChainEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, slots, phi_cfg, seed, window_s=None, noise_offset=None, n_rear=None, dt=None, action_map="linear"):
        super().__init__()
        self.action_map = action_map
        self.episode_log = []
        rl = load_config("rl")
        self.slots, self.phi_cfg = slots, dict(phi_cfg)
        self.dt = idm.DT if dt is None else float(dt)
        w = rl["train_window_s"] if window_s is None else window_s
        self.n_min, self.n_max = (int(round(float(x) / self.dt)) + 1 for x in w)
        self.noise_offset = int(rl["noise_seed_offset"] if noise_offset is None else noise_offset)
        self.n_rear = int(load_config("eval")["n_rear"] if n_rear is None else n_rear)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self._seed(seed)

    def _seed(self, seed):
        self.train_seed = int(seed)
        self.rng = np.random.default_rng([self.noise_offset, self.train_seed])

    def draw_window(self):
        for _ in range(1000):
            n_points = int(self.rng.integers(self.n_min, self.n_max + 1))
            w = window_weights(self.slots, n_points)
            if w.sum() > 0:
                k = int(self.rng.choice(len(self.slots), p=w / w.sum()))
                i0 = int(self.rng.integers(0, int(w[k])))
                return k, i0, i0 + n_points - 1
        raise RuntimeError("no slot can hold a training window")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._seed(seed)
        k, i0, i1 = self.draw_window()
        slot = self.slots[k]
        self.slot, self.window = slot, (slot["exp"], slot["av"], i0, i1)
        ep = slice_episode(slot["ep"], i0, i1)
        child = np.random.default_rng(int(self.rng.integers(0, 2 ** 63 - 1)))
        self.sim = ChainStepper(ep, slot["params"], n_rear=self.n_rear, noise_sd=slot["noise_sd"], rng=child, dt=self.dt)
        self.ctrl = _ActionController()
        self.platform = AVPlatform(self.ctrl, s_min=slot["s_min"], dt=self.dt)
        self.platform.reset({})
        self.est = MeanSpeedEstimator(dt=self.dt)
        self.phi_src = make_phi_source(self.phi_cfg, slot)
        for vl in slot["ep"]["v_l"][:i0]:           # causal history from t = 30 s up to the window start
            self.est.update(vl)
            self.phi_src.update({"v_leader": float(vl)})
        self._decide_state(self.sim.obs)
        self._ep = {"ret": 0.0, "n": 0, "v": 0.0, "vl": 0.0, "e_gap": 0.0, "at_bound": 0, "filter": 0}
        return self._obs(), {"window": self.window}

    def _decide_state(self, o):
        self.U = self.est.update(o["v_leader"])
        self.phi = self.phi_src.update(o)

    def _obs(self):
        return obs_vector(self.sim.obs, self.U, self.phi, self.platform.a_cmd_prev, self.platform.a_act, self.slot["policy"])

    def step(self, action):
        u = float(np.asarray(action, dtype=float).reshape(-1)[0])
        self.ctrl.a = action_to_accel(action, self.action_map)
        n_safe = self.platform.n_safety
        a_act = self.platform.act(self.sim.obs)
        o_next = self.sim.step(a_act)
        cost = step_cost(o_next, a_act, self.U, self.phi, self.slot["policy"])
        info = {"phi": self.phi, "U": self.U, "a_act": a_act, "cost": cost,
                "collided": bool(np.any(self.sim.gap <= 0.0)), "filter_active": self.platform.n_safety > n_safe}
        self._decide_state(o_next)
        pol, ep = self.slot["policy"], self._ep
        ep["ret"] -= cost
        ep["n"] += 1
        ep["v"] += o_next["v"]
        ep["vl"] += o_next["v_leader"]
        ep["e_gap"] += (o_next["gap"] - pol["s0"] - pol["h"] * o_next["v"]) / pol["sigma_gap"]
        ep["at_bound"] += abs(u) >= 1.0
        ep["filter"] += info["filter_active"]
        if self.sim.done:                                # diagnostics per window
            n = ep["n"]
            self.episode_log.append({"exp": self.window[0], "av": self.window[1], "steps": n, "return": ep["ret"],
                                     "mean_reward": ep["ret"] / n, "speed_ratio": ep["v"] / max(ep["vl"], 1e-9),
                                     "mean_gap_err": ep["e_gap"] / n,
                                     "last_gap_err": (o_next["gap"] - pol["s0"] - pol["h"] * o_next["v"]) / pol["sigma_gap"],
                                     "frac_at_bound": ep["at_bound"] / n, "frac_filter": ep["filter"] / n})
        return self._obs(), -cost, False, self.sim.done, info
