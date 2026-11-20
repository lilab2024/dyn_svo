"""RL training environment: equality with sim.chain, windows, prefill, reward, Gymnasium API."""
import math

import numpy as np
import pandas as pd
import pytest
from gymnasium.utils.env_checker import check_env

from dyn_svo.control import cooperation as co
from dyn_svo.control import mpc as M
from dyn_svo.control.platform import AVPlatform
from dyn_svo.rl import env as E
from dyn_svo.rl.stepper import ChainStepper, slice_episode
from dyn_svo.sim import chain

P = {"v0": 15.0, "T": 1.2, "s0": 2.0, "a_max": 1.0, "b": 1.5}
POL = {"s0": 3.51, "h": 1.35, "sigma_gap": 2.76, "sigma_dv": 0.55, "sigma_a": 0.29, "sigma_u": 0.73}
THR = {"I": {"min_gap": 1.0, "speed_ratio": 0.9, "mean_gap": 30.0}}
PHI_THR = {"loeo_A": {"I": {"10": {"a_lo": 0.05, "a_hi": 0.4}, "20": {"a_lo": 0.05, "a_hi": 0.4},
                            "40": {"a_lo": 0.05, "a_hi": 0.4}}}}
DYN = {"window_s": 10.0, "phi_lo_pi16": 4.0, "phi_hi_pi16": 7.5}


def _ring(ids, seconds=90.0, L=260.0, exp="A"):
    n = len(ids)
    k = np.arange(int(seconds * 30))
    t = k / 30.0
    rows = []
    for i, vid in enumerate(ids):
        j = (i - 1) % n
        fr = lambda m: L - m * L / n + 5.0 * t + 1.5 * np.sin(0.25 * t + m)
        gap = np.mod(fr(j) - fr(i) - 4.5, L)
        rows.append(pd.DataFrame({"exp": exp, "veh_id": vid, "regime": "I", "k": k, "t": t, "x": fr(i),
                                  "v": 5.0 + 0.375 * np.cos(0.25 * t + i), "a": 0.0, "length": 4.5,
                                  "leader_id": ids[j], "gap": gap}))
    return pd.concat(rows, ignore_index=True)


def _slots(seconds=(90.0, 60.0)):
    table = pd.DataFrame([dict(fold="loeo_A", regime="I", variant="nodelay", veh=v, **P) for v in ("1", "2", "pooled")])
    noise = pd.DataFrame([{"fold": "loeo_A", "regime": "I", "noise_sd": 0.3}])
    data = {"C": _ring([1, 2, 3, 4], seconds[0], exp="C"), "E": _ring([5, 6, 7], seconds[1], exp="E")}
    return E.build_slots(data, "loeo_A", table, noise, THR, {"I": POL}, PHI_THR, 30.0)


def test_slice_equals_episode_built_at_window_start():
    df = _ring([1, 2, 3, 4])
    full = chain.build_episode(df, 2, t_start_s=30.0)
    cut = slice_episode(full, 40, 250)
    ref = chain.build_episode(df, 2, t_start_s=(900 + 3 * 40) / 30.0, t_end_s=(900 + 3 * 250) / 30.0)
    for key in ("t", "x_l", "v_l", "v0", "real_gap"):
        assert np.array_equal(cut[key], ref[key])
    assert np.allclose(cut["x0"], ref["x0"], rtol=0, atol=1e-9)


@pytest.mark.parametrize("phi_src", ["fixed", "dynamic"])
def test_stepper_reproduces_chain_simulate_exactly(phi_src):
    df = _ring([1, 2, 3, 4])
    ep = chain.build_episode(df, 3, t_start_s=30.0)
    plist = [dict(P, T=1.0 + 0.1 * i) for i in range(len(ep["order"]))]

    def driver():
        src = M.FixedPhi(0.7) if phi_src == "fixed" else co.LeaderActivityWeight(10.0, 0.6, 1.4, 0.05, 0.4)
        return AVPlatform(M.MPC(POL, src), s_min=1.0)

    ref = chain.simulate(ep, plist, av_driver=driver(), n_rear=8, noise_sd=0.3, rng=chain.episode_rng(4, "A", 3))
    d = driver()
    d.reset({})
    st = ChainStepper(ep, plist, n_rear=8, noise_sd=0.3, rng=chain.episode_rng(4, "A", 3))
    xs, vs, gaps = [st.x.copy()], [st.v.copy()], [st.gap.copy()]
    while not st.done:
        st.step(d.act(st.obs))
        xs.append(st.x.copy()); vs.append(st.v.copy()); gaps.append(st.gap.copy())
    assert np.array_equal(np.array(xs), ref["x"]) and np.array_equal(np.array(vs), ref["v"])
    assert np.array_equal(np.array(gaps), ref["gap"])


def test_window_weights_and_drawn_windows_fit():
    slots = _slots()
    n_long = len(slots[0]["ep"]["t"])
    w = E.window_weights(slots, n_long)
    assert w[0] == 1 and all(w[k] == 0 for k, s in enumerate(slots) if s["exp"] == "E")
    env = E.ChainEnv(slots, {"phi": 0.5}, seed=1, window_s=(5.0, 20.0))
    for _ in range(50):
        k, i0, i1 = env.draw_window()
        assert 0 <= i0 < i1 < len(slots[k]["ep"]["t"]) and env.n_min <= i1 - i0 + 1 <= env.n_max


def test_prefill_matches_evaluation_controller():
    slots = _slots()
    env = E.ChainEnv(slots, DYN, seed=3, window_s=(5.0, 20.0))
    env.reset()
    exp, av, i0, i1 = env.window
    slot = [s for s in slots if s["exp"] == exp and s["av"] == av][0]
    thr = PHI_THR["loeo_A"]["I"]["10"]
    mpc = M.MPC(POL, co.LeaderActivityWeight(10.0, 4 * co.PI16, 7.5 * co.PI16, thr["a_lo"], thr["a_hi"]))
    mpc.reset({})
    for vl in slot["ep"]["v_l"][:i0 + 1]:
        mpc.act({"v": 5.0, "gap": 10.0, "v_leader": float(vl)})
    assert env.phi == mpc.last_phi and env.U == mpc.last_U


def test_reward_is_minus_cost_of_state_after_step():
    env = E.ChainEnv(_slots(), DYN, seed=5, window_s=(5.0, 20.0))
    env.reset()
    U, phi = env.U, env.phi
    obs, r, term, trunc, info = env.step(np.array([0.2], dtype=np.float32))
    s, v, vl = env.sim.gap[0], env.sim.v[0], env.sim.lead_v[0]
    ego = ((s - POL["s0"] - POL["h"] * v) / POL["sigma_gap"]) ** 2 + ((v - vl) / POL["sigma_dv"]) ** 2
    soc = (info["a_act"] / POL["sigma_a"]) ** 2 + ((v - U) / POL["sigma_u"]) ** 2
    assert r == pytest.approx(-(math.cos(phi) * ego + math.sin(phi) * soc), rel=1e-12)
    assert not term and obs.shape == (E.OBS_DIM,)
    assert E.action_to_accel([-1.0]) == E.A_LO and E.action_to_accel([1.0]) == E.A_HI


def test_episode_truncates_at_window_end_and_fixed_phi_is_constant():
    env = E.ChainEnv(_slots(), {"phi": 0.5}, seed=7, window_s=(5.0, 6.0))
    env.reset()
    n, phis = 0, set()
    while True:
        obs, r, term, trunc, info = env.step(np.zeros(1, dtype=np.float32))
        n += 1
        phis.add(info["phi"])
        if trunc:
            break
    exp, av, i0, i1 = env.window
    assert n == i1 - i0 and phis == {0.5}


def test_seeded_reset_is_reproducible():
    def run(seed):
        env = E.ChainEnv(_slots(), DYN, seed=seed, window_s=(5.0, 20.0))
        obs, _ = env.reset()
        out = [obs]
        for a in (0.1, -0.3, 0.5):
            out.append(env.step(np.array([a], dtype=np.float32))[0])
        return env.window, np.array(out)
    w1, o1 = run(11)
    w2, o2 = run(11)
    w3, o3 = run(12)
    assert w1 == w2 and np.array_equal(o1, o2)
    assert w1 != w3 or not np.array_equal(o1, o3)


def test_gymnasium_env_checker():
    check_env(E.ChainEnv(_slots(), DYN, seed=0, window_s=(5.0, 20.0)), skip_render_check=True)
