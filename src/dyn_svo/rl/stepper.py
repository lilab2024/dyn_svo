"""Step-wise open-chain replay for RL training.

ChainStepper reproduces sim.chain.simulate step by step for a non-human AV driver: same noise draws, same
IDM, same ballistic update, same observation dict (equality is a unit test). Evaluation of RL policies still
uses sim.chain.simulate through the normal runner. slice_episode cuts a training window out of an episode
built from t = 30 s, with the initial chain state taken from the recorded data at the window start.
"""
import numpy as np

from dyn_svo.sim import idm


def slice_episode(ep, i0, i1):
    """Steps i0..i1 (inclusive) of an episode from chain.build_episode, as if built from step i0."""
    if not 0 <= i0 < i1 < len(ep["t"]):
        raise ValueError("need 0 <= i0 < i1 < episode length")
    out = dict(ep)
    for key in ("k", "t", "x_l", "v_l", "real_v", "real_gap"):
        out[key] = np.asarray(ep[key])[i0:i1 + 1]
    gap0 = np.asarray(ep["real_gap"])[i0]
    x0 = np.empty(len(ep["order"]))
    front, front_len = float(np.asarray(ep["x_l"])[i0]), float(ep["len_l"])
    for i in range(len(ep["order"])):
        x0[i] = front - front_len - gap0[i]
        front, front_len = x0[i], float(ep["lengths"][i])
    out["x0"] = x0
    out["v0"] = np.asarray(ep["real_v"])[i0].astype(float)
    return out


class ChainStepper:
    """Advance the chain one step at a time; slot 0 is the AV and receives no human noise."""

    def __init__(self, ep, params_list, n_rear=5, noise_sd=0.0, rng=None, dt=None):
        self.ep = ep
        self.dt = idm.DT if dt is None else dt
        self.n_veh, self.n_step = len(ep["order"]), len(ep["t"])
        self.P = {k: np.array([p[k] for p in params_list], dtype=float) for k in idm.PARAM_NAMES}
        self.lead_len = np.concatenate([[ep["len_l"]], ep["lengths"][:-1]])
        self.n_rear = int(n_rear)
        self.eps = np.zeros((self.n_step, self.n_veh))
        if noise_sd > 0.0:
            if rng is None:
                raise ValueError("noise_sd > 0 needs an rng")
            self.eps = rng.standard_normal((self.n_step, self.n_veh)) * float(noise_sd)
        self.x = np.asarray(ep["x0"], dtype=float).copy()
        self.v = np.asarray(ep["v0"], dtype=float).copy()
        self.n = 0
        self._observe()

    def _observe(self):
        n = self.n
        self.lead_x = np.concatenate([[self.ep["x_l"][n]], self.x[:-1]])
        self.lead_v = np.concatenate([[self.ep["v_l"][n]], self.v[:-1]])
        self.gap = self.lead_x - self.lead_len - self.x
        self.obs = {"t": float(self.ep["t"][n]), "v": float(self.v[0]), "gap": float(self.gap[0]),
                    "v_leader": float(self.lead_v[0]), "rear_v": self.v[1:1 + self.n_rear].copy(),
                    "rear_gap": self.gap[1:1 + self.n_rear].copy()}
        return self.obs

    @property
    def done(self):
        return self.n >= self.n_step - 1

    def step(self, a_av):
        """Apply the AV's actual acceleration for the current step and return the observation of the next one."""
        if self.done:
            raise RuntimeError("episode finished")
        n, P = self.n, self.P
        acc = idm.idm_accel(self.v, self.gap, self.v - self.lead_v, P["v0"], P["T"], P["s0"], P["a_max"], P["b"])
        acc = np.maximum(acc + self.eps[n], idm.ACCEL_FLOOR)
        acc[0] = max(float(a_av), idm.ACCEL_FLOOR)
        self.acc = acc
        self.x, self.v = idm.ballistic_step(self.x, self.v, acc, self.dt)
        self.n += 1
        return self._observe()
