import numpy as np
import pytest

from dyn_svo.metrics import oscillation as osc
from dyn_svo.sim import chain, idm

P = {"v0": 15.0, "T": 1.2, "s0": 2.0, "a_max": 1.0, "b": 1.5}


def _ep(n=4, steps=600):
    t = np.arange(steps) * idm.DT
    v_l = 5.0 + np.sin(2 * np.pi * t / 25.0)
    x_l = 300.0 + np.concatenate([[0.0], np.cumsum(0.5 * (v_l[1:] + v_l[:-1]) * idm.DT)])
    s_e = float(idm.equilibrium_gap(5.0, **P))
    return {"exp": "A", "regime": "I", "av_id": 7, "leader_id": 0, "order": list(range(1, n + 1)),
            "lengths": np.full(n, 4.5), "len_l": 4.5, "t": t, "x_l": x_l, "v_l": v_l,
            "x0": x_l[0] - np.arange(1, n + 1) * (4.5 + s_e), "v0": np.full(n, 5.0),
            "real_v": np.tile(v_l[:, None], (1, n))}


class ConstantDriver:
    def reset(self, info):
        pass

    def act(self, obs):
        return 0.0


def test_zero_noise_equals_deterministic():
    ep = _ep()
    a = chain.simulate(ep, [P] * 4)
    b = chain.simulate(ep, [P] * 4, noise_sd=0.0)
    assert np.array_equal(a["v"], b["v"]) and not b["noise"].any()


def test_noise_needs_rng_and_is_reproducible():
    ep = _ep()
    with pytest.raises(ValueError):
        chain.simulate(ep, [P] * 4, noise_sd=0.5)
    r1 = chain.simulate(ep, [P] * 4, noise_sd=0.5, rng=chain.episode_rng(0, "A", 7))
    r2 = chain.simulate(ep, [P] * 4, noise_sd=0.5, rng=chain.episode_rng(0, "A", 7))
    r3 = chain.simulate(ep, [P] * 4, noise_sd=0.5, rng=chain.episode_rng(1, "A", 7))
    assert np.array_equal(r1["v"], r2["v"]) and not np.array_equal(r1["v"], r3["v"])
    assert r1["noise"].std() == pytest.approx(0.5, rel=0.1)


def test_common_random_numbers_across_slot0_drivers():
    ep = _ep()
    base = chain.simulate(ep, [P] * 4, noise_sd=0.5, rng=chain.episode_rng(0, "A", 7))
    human = chain.simulate(ep, [P] * 4, av_driver=chain.HumanDriver(P), noise_sd=0.5,
                           rng=chain.episode_rng(0, "A", 7))
    const = chain.simulate(ep, [P] * 4, av_driver=ConstantDriver(), noise_sd=0.5,
                           rng=chain.episode_rng(0, "A", 7))
    assert np.allclose(base["v"], human["v"])                 # human slot gets the same noise
    assert np.array_equal(base["noise"], const["noise"])     # same draws for every arm
    assert np.all(const["a"][:, 0] == 0.0)                    # non-human slot ignores its draw


def test_combine_seed_rows():
    ep = _ep()
    rows = []
    for s in range(3):
        sim = chain.simulate(ep, [P] * 4, noise_sd=0.3, rng=chain.episode_rng(s, "A", 7))
        rows.append(osc.episode_validation(ep, sim))
    c = osc.combine_seed_rows(rows)
    assert c["n_seeds"] == 3
    assert c["mean_log_ratio"] == pytest.approx(np.mean([r["mean_log_ratio"] for r in rows]))
    assert np.allclose(c["log_ratio_by_position"],
                       np.mean([r["log_ratio_by_position"] for r in rows], axis=0))
