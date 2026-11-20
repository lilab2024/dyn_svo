"""State-dependent cooperation weight phi_t = arctan(Gamma_t).

g_j: peak gain of the linearised IDM (leader speed -> follower speed) of rear vehicle j over the
wave band, linearised at its current gap and speed with zero relative speed, using the AV's model
of humans (pooled IDM parameters). Gamma_t = sum_k prod_{j<=k} g_j over the K rear vehicles,
low-pass filtered; phi is rate-limited.
"""
import math

import numpy as np

from dyn_svo.config import load_config
from dyn_svo.sim import idm

_C = load_config("cooperation")


def idm_partials(s, v, dv, p, delta=idm.DELTA):
    """dF/ds, dF/dv, dF/d(dv) of the IDM acceleration, dv = v - v_leader (arrays broadcast)."""
    s = np.maximum(np.asarray(s, dtype=float), idm.MIN_GAP_EPS)
    v = np.asarray(v, dtype=float)
    dv = np.asarray(dv, dtype=float)
    sab = 2.0 * math.sqrt(p["a_max"] * p["b"])
    raw = v * p["T"] + v * dv / sab
    inner = raw > 0.0
    s_star = p["s0"] + np.where(inner, raw, 0.0)
    ds_dv = np.where(inner, p["T"] + dv / sab, 0.0)
    ds_ddv = np.where(inner, v / sab, 0.0)
    f_s = 2.0 * p["a_max"] * s_star ** 2 / s ** 3
    f_v = -p["a_max"] * (delta * v ** (delta - 1) / p["v0"] ** delta + 2.0 * s_star * ds_dv / s ** 2)
    f_dv = -p["a_max"] * 2.0 * s_star * ds_ddv / s ** 2
    return f_s, f_v, f_dv


def transfer_gain(f_s, f_v, f_dv, omega):
    """|G(i omega)|, G = (F_s - lam F_dv) / (lam^2 - lam (F_v + F_dv) + F_s), leader -> follower speed."""
    lam = 1j * np.asarray(omega, dtype=float)
    return np.abs((f_s - lam * f_dv) / (lam * lam - lam * (f_v + f_dv) + f_s))


def band_omegas(cfg=None):
    c = cfg or _C
    lo, hi = c["wave_band_s"]
    return 2.0 * np.pi / np.geomspace(lo, hi, int(c["n_periods"]))


def peak_gains(gaps, speeds, p, omegas):
    """g_j for each rear vehicle: max over the band of |G| at (gap_j, v_j, dv = 0)."""
    f_s, f_v, f_dv = idm_partials(np.asarray(gaps)[:, None], np.asarray(speeds)[:, None], 0.0, p)
    return transfer_gain(f_s, f_v, f_dv, np.asarray(omegas)[None, :]).max(axis=1)


def gamma_from_gains(g):
    """Gamma = sum_k prod_{j<=k} g_j."""
    return float(np.sum(np.cumprod(np.asarray(g, dtype=float))))


class CooperationWeight:
    """Causal phi_t from rear-vehicle observations (rear_gap, rear_v of the K vehicles behind)."""

    def __init__(self, params, K, tau_f, cfg=None, dt=None):
        self.c = cfg or _C
        self.p = dict(params)
        self.K = int(K)
        self.tau_f = float(tau_f)
        self.dt = idm.DT if dt is None else float(dt)
        self.omegas = band_omegas(self.c)
        self.reset()

    def reset(self):
        self.gamma_f = None
        self.phi = None
        self.last_gamma = None

    def update(self, obs):
        gaps, speeds = obs["rear_gap"][:self.K], obs["rear_v"][:self.K]
        gamma = gamma_from_gains(peak_gains(gaps, speeds, self.p, self.omegas)) if len(gaps) else 0.0
        self.last_gamma = gamma
        if self.gamma_f is None:
            self.gamma_f = gamma
        else:
            self.gamma_f += (gamma - self.gamma_f) * min(1.0, self.dt / self.tau_f)
        target = math.atan(self.gamma_f)
        if self.phi is None:
            self.phi = target
        else:
            step = self.c["phi_rate_max"] * self.dt
            self.phi += min(max(target - self.phi, -step), step)
        return self.phi


# ---------- v5 phi_t from leader speed activity; the classes above are ----------
PI16 = math.pi / 16


def window_key(x):
    """Stable text key for a window or a phi bound: 10.0 -> '10', 7.5 -> '7.5'."""
    return "%g" % float(x)


class RollingSD:
    """Causal population SD (ddof 0) of the last n samples; the shorter history is used at the start."""

    def __init__(self, n):
        self.n = max(1, int(n))
        self.reset()

    def reset(self):
        self.buf = np.zeros(self.n)
        self.i = 0
        self.count = 0
        self.s = 0.0
        self.s2 = 0.0

    def push(self, x):
        x = float(x)
        if self.count == self.n:
            old = self.buf[self.i]
            self.s -= old
            self.s2 -= old * old
        else:
            self.count += 1
        self.buf[self.i] = x
        self.s += x
        self.s2 += x * x
        self.i = (self.i + 1) % self.n
        m = self.s / self.count
        return math.sqrt(max(self.s2 / self.count - m * m, 0.0))


def leader_activity(v, window_s, dt=None):
    """Activity series a_t of a speed series sampled at the simulation step (same as the controller)."""
    dt = idm.DT if dt is None else float(dt)
    r = RollingSD(int(round(float(window_s) / dt)))
    return np.array([r.push(x) for x in np.asarray(v, dtype=float)])


def phi_from_activity(a, a_lo, a_hi, phi_lo, phi_hi):
    z = (a - a_lo) / (a_hi - a_lo)
    return phi_lo + (phi_hi - phi_lo) * min(max(z, 0.0), 1.0)


class LeaderActivityWeight:
    """v5 phi_t: phi = phi_lo + (phi_hi - phi_lo) clip((a - a_lo) / (a_hi - a_lo), 0, 1), with a the
    causal SD of the leader speed over the last window_s seconds; first step = target, then rate-limited.
    phi_lo, phi_hi in radians; a_lo, a_hi from the episode's fold (scripts/phi_thresholds.py)."""

    def __init__(self, window_s, phi_lo, phi_hi, a_lo, a_hi, cfg=None, dt=None):
        self.c = cfg or _C
        self.dt = idm.DT if dt is None else float(dt)
        self.window_s = float(window_s)
        self.phi_lo, self.phi_hi = float(phi_lo), float(phi_hi)
        self.a_lo, self.a_hi = float(a_lo), float(a_hi)
        if not (0.0 <= self.phi_lo <= self.phi_hi <= math.pi / 2 + 1e-12) or not self.a_hi > self.a_lo:
            raise ValueError("need 0 <= phi_lo <= phi_hi <= pi/2 and a_hi > a_lo")
        self.sd = RollingSD(int(round(self.window_s / self.dt)))
        self.reset()

    def reset(self):
        self.sd.reset()
        self.phi = None
        self.last_activity = None

    def update(self, obs):
        a = self.sd.push(obs["v_leader"])
        self.last_activity = a
        target = phi_from_activity(a, self.a_lo, self.a_hi, self.phi_lo, self.phi_hi)
        if self.phi is None:
            self.phi = target
        else:
            step = self.c["phi_rate_max"] * self.dt
            self.phi += min(max(target - self.phi, -step), step)
        return self.phi


def leader_activity_grid(cfg=None):
    """[(window_s, phi_lo_pi16, phi_hi_pi16)] of the v5 dynamic arm, windows outer, ranges inner."""
    c = (cfg or _C)["leader_activity"]
    return [(float(w), float(lo), float(hi)) for w in c["window_s"] for lo, hi in c["phi_ranges_pi16"]]


def threshold_folds(dev):
    """Training experiments per fold: loeo_k = dev without k, all_dev = every dev experiment."""
    out = {"loeo_" + k: [e for e in dev if e != k] for k in dev}
    out["all_dev"] = list(dev)
    return out


def thresholds_from_samples(samples, folds, percentiles):
    """samples[(exp, regime, window_key)] = activity values; returns {fold: {regime: {window: {a_lo, a_hi, n}}}}
    from the fold's training experiments only."""
    lo_p, hi_p = percentiles
    out = {}
    for fold, train in folds.items():
        for (exp, regime, w), _ in sorted(samples.items()):
            if exp not in train:
                continue
            node = out.setdefault(fold, {}).setdefault(regime, {})
            if w in node:
                continue
            vals = np.concatenate([s for (e, r, ww), s in samples.items() if e in train and r == regime and ww == w])
            node[w] = {"a_lo": float(np.percentile(vals, lo_p)), "a_hi": float(np.percentile(vals, hi_p)),
                       "n": int(len(vals)), "train": sorted({e for (e, r, ww) in samples if e in train and r == regime})}
    return out


def load_phi_thresholds(run_name=None):
    import json
    from dyn_svo.data import manifest as mf
    run_name = run_name or _C["leader_activity"]["thresholds_run"]
    p = mf.data_root() / "dyn_svo" / "results" / "mpc" / run_name / "thresholds.json"
    with open(p, encoding="utf-8") as f:
        return json.load(f)["thresholds"]
