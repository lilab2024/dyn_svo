"""PPO training of one RL policy."""
import time

import numpy as np

from dyn_svo.config import load_config
from dyn_svo.rl import env as E
from dyn_svo.rl.policy import PolicyWeights


def make_model(env, seed, learning_rate, gamma, rl=None):
    import torch
    from stable_baselines3 import PPO
    rl = rl or load_config("rl")
    torch.set_num_threads(1)
    if rl["activation"] != "tanh":
        raise ValueError(" fixes tanh")
    kwargs = {"net_arch": {"pi": list(rl["net_arch"]), "vf": list(rl["net_arch"])}, "activation_fn": torch.nn.Tanh}
    return PPO("MlpPolicy", env, learning_rate=float(learning_rate), gamma=float(gamma), seed=int(seed),
               device="cpu", verbose=0, policy_kwargs=kwargs)


NORMALIZE = ("none", "reward", "reward_obs")


def train_policy(slots, phi_cfg, seed, steps, learning_rate, gamma, window_s=None, action_map="linear", normalize="none"):
    """Train one policy; returns (PolicyWeights, record, model). All PPO settings other than learning rate and gamma
    are Stable-Baselines3 defaults. action_map and normalize are diagnostic options;
    the defaults are the behaviour. normalize: VecNormalize of the reward (scale only), for reward_obs,
    of the observations; the observation statistics are frozen into the returned weights."""
    if normalize not in NORMALIZE:
        raise ValueError("normalize must be one of " + ", ".join(NORMALIZE))
    env = E.ChainEnv(slots, phi_cfg, seed=seed, window_s=window_s, action_map=action_map)
    venv = None
    train_env = env
    if normalize != "none":
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        venv = VecNormalize(DummyVecEnv([lambda: env]), norm_obs=(normalize == "reward_obs"), norm_reward=True,
                            gamma=float(gamma), clip_reward=float("inf"))      # scale only, never clip the reward
        venv.seed(int(seed))
        train_env = venv
    model = make_model(train_env, seed, learning_rate, gamma)
    t0 = time.time()
    model.learn(total_timesteps=int(steps))
    sec = time.time() - t0
    w = PolicyWeights.from_sb3(model)
    if normalize == "reward_obs":
        w = w.with_obs_norm(venv)
    rec = {"seed": int(seed), "steps": int(steps), "learning_rate": float(learning_rate), "gamma": float(gamma),
           "phi_cfg": dict(phi_cfg), "seconds": sec, "us_per_step": 1e6 * sec / max(int(steps), 1),
           "n_steps": int(model.n_steps), "batch_size": int(model.batch_size), "n_epochs": int(model.n_epochs),
           "action_map": action_map, "normalize": normalize, "episode_log": list(env.episode_log)}
    return w, rec, model
