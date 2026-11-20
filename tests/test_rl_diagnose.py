"""Diagnostic options of RL training: centred action mapping, normalization frozen into the weights,
episode diagnostics, and identical behaviour of training environment and evaluation driver."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from dyn_svo.rl import env as E
from dyn_svo.rl import train as T
from dyn_svo.rl.policy import PolicyWeights, make_rl_driver
from dyn_svo.sim import chain

import test_rl_env as TE
import test_rl_policy as TP


def test_action_mappings():
    assert E.action_to_accel([0.0], "centred") == 0.0 and E.action_to_accel([0.0]) == pytest.approx(-0.75)
    for m in ("linear", "centred"):
        assert E.action_to_accel([-1.0], m) == E.A_LO and E.action_to_accel([1.0], m) == E.A_HI
    assert E.action_to_accel([-0.5], "centred") == pytest.approx(0.5 * E.A_LO)
    with pytest.raises(ValueError):
        E.action_to_accel([0.0], "other")


def test_centred_mapping_same_in_training_and_evaluation():
    slots = TE._slots()
    for s in slots:
        s["noise_sd"] = 0.0
    n = len(slots[0]["ep"]["t"])
    env = E.ChainEnv(slots, TE.DYN, seed=0, window_s=(5.0, 20.0), action_map="centred")
    env.draw_window = lambda: (0, 0, n - 1)
    w = TP._random_weights(4)
    obs, _ = env.reset()
    xs, trunc = [env.sim.x.copy()], False
    while not trunc:
        obs, r, term, trunc, info = env.step(w.action(obs))
        xs.append(env.sim.x.copy())
    s = slots[0]
    drv = make_rl_driver(w, s["policy"], E.make_phi_source(TE.DYN, s), s["s_min"], action_map="centred")
    ref = chain.simulate(s["ep"], s["params"], av_driver=drv, n_rear=8, noise_sd=0.0)
    assert np.array_equal(np.array(xs), ref["x"])
    log = env.episode_log[-1]
    assert log["steps"] == n - 1 and 0.0 <= log["frac_at_bound"] <= 1.0 and "speed_ratio" in log


def test_observation_normalization_frozen_into_weights(tmp_path):
    w, rec, model = T.train_policy(TE._slots(), {"phi": 0.785}, seed=1, steps=256, learning_rate=3e-4, gamma=0.99,
                                   window_s=(5.0, 20.0), action_map="centred", normalize="reward_obs")
    venv = model.get_env()
    assert w.obs_mean is not None and rec["normalize"] == "reward_obs" and len(rec["episode_log"]) > 0
    env = E.ChainEnv(TE._slots(), {"phi": 0.785}, seed=9, window_s=(5.0, 20.0), action_map="centred")
    obs, _ = env.reset()
    for _ in range(10):
        a_sb3, _ = model.predict(venv.normalize_obs(obs), deterministic=True)
        assert np.allclose(w.action(obs), a_sb3, atol=1e-6)
        obs = env.step(a_sb3)[0]
    w.to_npz(tmp_path / "p.npz")
    w2 = PolicyWeights.from_npz(tmp_path / "p.npz")
    assert np.array_equal(w2.action(obs), w.action(obs)) and np.array_equal(w2.obs_var, w.obs_var)
    with pytest.raises(ValueError):
        T.train_policy(TE._slots(), {"phi": 0.785}, seed=1, steps=8, learning_rate=3e-4, gamma=0.99, normalize="x")


def test_diagnose_script_rules():
    path = Path(__file__).resolve().parents[1] / "scripts" / "rl_diagnose.py"
    spec = importlib.util.spec_from_file_location("rl_diagnose", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.VARIANTS["V0"] == ("linear", "none") and mod.VARIANTS["V3"] == ("centred", "reward_obs")
    assert mod.ConstantAction("centred").action(None)[0] == 0.0
    assert E.action_to_accel(mod.ConstantAction("linear").action(None), "linear") == pytest.approx(0.0, abs=1e-12)
    rows = [{"speed_ratio": 0.97, "last_gap": 20.0, "policy_mean_reward": -3.0, "zero_mean_reward": -5.0}]
    assert mod.g4(rows, 13.7)["pass"] and not mod.g4([dict(rows[0], last_gap=40.0)], 13.7)["pass"]
    with pytest.raises(SystemExit):
        mod.main(["--run-name", "rl_diag", "--variant", "V1", "--seed", "0"])
