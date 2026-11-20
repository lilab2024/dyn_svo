"""RL policy weights and evaluation driver: numpy policy equals SB3, training and evaluation see the same inputs."""
import numpy as np
import pytest

from dyn_svo.control import mpc as M
from dyn_svo.rl import env as E
from dyn_svo.rl import train as T
from dyn_svo.rl.policy import PolicyWeights, make_rl_driver
from dyn_svo.sim import chain

import test_rl_env as TE


def _random_weights(seed=0):
    r = np.random.default_rng(seed)
    return PolicyWeights([(r.normal(0, 0.5, (64, E.OBS_DIM)), r.normal(0, 0.1, 64)),
                          (r.normal(0, 0.3, (64, 64)), r.normal(0, 0.1, 64))], r.normal(0, 0.5, (1, 64)), r.normal(0, 0.1, 1))


def test_numpy_policy_equals_sb3_predict(tmp_path):
    slots = TE._slots()
    env = E.ChainEnv(slots, TE.DYN, seed=0, window_s=(5.0, 20.0))
    model = T.make_model(env, seed=0, learning_rate=3e-4, gamma=0.99)
    w = PolicyWeights.from_sb3(model)
    obs, _ = env.reset()
    for _ in range(20):
        a_sb3, _ = model.predict(obs, deterministic=True)
        assert np.allclose(w.action(obs), a_sb3, atol=1e-6)
        obs = env.step(a_sb3)[0]
    w.to_npz(tmp_path / "p.npz")
    w2 = PolicyWeights.from_npz(tmp_path / "p.npz")
    assert np.array_equal(w2.action(obs), w.action(obs))


@pytest.mark.parametrize("phi_cfg", [{"phi": 0.6}, TE.DYN])
def test_training_env_and_evaluation_driver_give_identical_trajectories(phi_cfg):
    slots = TE._slots()
    for s in slots:
        s["noise_sd"] = 0.0
    k = 0
    n = len(slots[k]["ep"]["t"])
    env = E.ChainEnv(slots, phi_cfg, seed=0, window_s=(5.0, 20.0))
    env.draw_window = lambda: (k, 0, n - 1)
    w = _random_weights(1)
    obs, _ = env.reset()
    xs = [env.sim.x.copy()]
    trunc = False
    while not trunc:
        obs, r, term, trunc, info = env.step(w.action(obs))
        xs.append(env.sim.x.copy())
    slot = slots[k]
    src = E.make_phi_source(phi_cfg, slot)
    drv = make_rl_driver(w, slot["policy"], src, slot["s_min"])
    ref = chain.simulate(slot["ep"], slot["params"], av_driver=drv, n_rear=8, noise_sd=0.0)
    assert np.array_equal(np.array(xs), ref["x"])


def test_short_training_run_records_defaults():
    w, rec, model = T.train_policy(TE._slots(), {"phi": 0.785}, seed=3, steps=256, learning_rate=1e-4, gamma=0.98,
                                   window_s=(5.0, 20.0))
    assert rec["n_steps"] == 2048 and rec["batch_size"] == 64 and rec["n_epochs"] == 10
    assert len(w.layers) == 2 and w.layers[0][0].shape == (64, E.OBS_DIM) and w.out_w.shape == (1, 64)
    w2, rec2, _ = T.train_policy(TE._slots(), {"phi": 0.785}, seed=3, steps=256, learning_rate=1e-4, gamma=0.98,
                                 window_s=(5.0, 20.0))
    assert all(np.array_equal(a[0], b[0]) for a, b in zip(w.layers, w2.layers))
