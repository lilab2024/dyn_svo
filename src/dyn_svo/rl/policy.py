"""RL policies as controllers of the shared platform.

PolicyWeights holds the MLP of a trained Stable-Baselines3 PPO policy as numpy arrays; its deterministic action
(mean of the Gaussian, clipped to [-1, 1]) equals model.predict(obs, deterministic=True) (unit test).
RLController builds the same 10-value observation as the training environment, using the platform it sits
in for the previous command and the actual acceleration, so a policy sees the same inputs in training and in
evaluation through sim.chain.simulate (unit test: identical trajectories without noise).
"""
import json

import numpy as np

from dyn_svo.control.estimator import MeanSpeedEstimator
from dyn_svo.control.platform import AVPlatform
from dyn_svo.rl import env as E


class PolicyWeights:
    def __init__(self, layers, out_w, out_b, activation="tanh", obs_mean=None, obs_var=None, clip_obs=10.0, eps=1e-8):
        if activation != "tanh":
            raise ValueError("only tanh is supported ")
        self.layers = [(np.asarray(w, dtype=np.float64), np.asarray(b, dtype=np.float64)) for w, b in layers]
        self.out_w, self.out_b = np.asarray(out_w, dtype=np.float64), np.asarray(out_b, dtype=np.float64)
        self.obs_mean = None if obs_mean is None else np.asarray(obs_mean, dtype=np.float64)
        self.obs_var = None if obs_var is None else np.asarray(obs_var, dtype=np.float64)
        self.clip_obs, self.eps = float(clip_obs), float(eps)

    def with_obs_norm(self, venv):
        """Freeze the observation statistics of an SB3 VecNormalize (diagnostic V3)."""
        rms = venv.obs_rms
        return PolicyWeights([(w, b) for w, b in self.layers], self.out_w, self.out_b, obs_mean=rms.mean.copy(),
                             obs_var=rms.var.copy(), clip_obs=venv.clip_obs, eps=venv.epsilon)

    @classmethod
    def from_sb3(cls, model):
        import torch
        lin = [m for m in model.policy.mlp_extractor.policy_net if isinstance(m, torch.nn.Linear)]
        act = model.policy.action_net
        get = lambda t: t.detach().cpu().numpy().astype(np.float64)
        return cls([(get(m.weight), get(m.bias)) for m in lin], get(act.weight), get(act.bias))

    def action(self, obs):
        h = np.asarray(obs, dtype=np.float32).astype(np.float64)
        if self.obs_mean is not None:
            h = np.clip((h - self.obs_mean) / np.sqrt(self.obs_var + self.eps), -self.clip_obs, self.clip_obs)
        for w, b in self.layers:
            h = np.tanh(w @ h + b)
        return np.clip(self.out_w @ h + self.out_b, -1.0, 1.0)

    def to_npz(self, path):
        arrays = {"out_w": self.out_w, "out_b": self.out_b}
        if self.obs_mean is not None:
            arrays.update(obs_mean=self.obs_mean, obs_var=self.obs_var, clip_obs=np.array(self.clip_obs), eps=np.array(self.eps))
        for i, (w, b) in enumerate(self.layers):
            arrays["w%d" % i], arrays["b%d" % i] = w, b
        np.savez(path, **arrays)

    @classmethod
    def from_npz(cls, path):
        with np.load(path) as z:
            n = sum(1 for k in z.files if k.startswith("w") and k[1:].isdigit())
            norm = {}
            if "obs_mean" in z.files:
                norm = {"obs_mean": z["obs_mean"], "obs_var": z["obs_var"], "clip_obs": float(z["clip_obs"]), "eps": float(z["eps"])}
            return cls([(z["w%d" % i], z["b%d" % i]) for i in range(n)], z["out_w"], z["out_b"], **norm)


class RLController:
    """Policy inside the shared platform; phi from a FixedPhi or LeaderActivityWeight source."""

    def __init__(self, weights, policy, phi_source, estimator=None, action_map="linear"):
        self.w, self.pol, self.phi_source = weights, dict(policy), phi_source
        self.action_map = action_map
        self.est = estimator or MeanSpeedEstimator()
        self.platform = None
        self.last_phi = None
        self.last_U = None

    def reset(self, info):
        self.est.reset()
        self.phi_source.reset()

    def act(self, obs):
        if self.platform is None:
            raise RuntimeError("RLController needs its platform (use make_rl_driver)")
        self.last_U = self.est.update(obs["v_leader"])
        self.last_phi = self.phi_source.update(obs)
        x = E.obs_vector(obs, self.last_U, self.last_phi, self.platform.a_cmd_prev, self.platform.a_act, self.pol)
        return {"a": E.action_to_accel(self.w.action(x), self.action_map)}


def make_rl_driver(weights, policy, phi_source, s_min, action_map="linear"):
    ctrl = RLController(weights, policy, phi_source, action_map=action_map)
    platform = AVPlatform(ctrl, s_min=s_min)
    ctrl.platform = platform
    return platform
