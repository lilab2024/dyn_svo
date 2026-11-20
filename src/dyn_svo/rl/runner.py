"""RL training tasks: enumeration and one PPO policy per task, written for resumable runs."""
import hashlib
import os
import sys
from importlib.metadata import version
from pathlib import Path

from dyn_svo.config import load_config
from dyn_svo.control import cooperation as co
from dyn_svo.rl import env as E
from dyn_svo.rl import train as T

CONFIGS = ("sim", "data_split", "data_manifest", "calibration", "av", "cooperation", "mpc", "eval", "rl")
STAGES = ("tune", "train", "match")
RL_PACKAGES = ("torch", "stable-baselines3", "gymnasium")


def training_sets(dev):
    """The 5 leave-one-out folds and all dev."""
    return [("loeo_" + k, [e for e in dev if e != k]) for k in dev] + [("all_dev", list(dev))]


def _p(x):
    return ("%g" % x).replace(".", "p")


def fixed_name(pi16):
    return "rl_fixed_" + _p(pi16)


def dyn_name(c):
    return "rl_dyn_W%d_lo%s_hi%s" % (int(c["window_s"]), _p(c["phi_lo_pi16"]), _p(c["phi_hi_pi16"]))


def _task(stage, config, phi_cfg, set_name, exps, seed, lr, gamma):
    return {"task_id": "%s__%s__%s__s%d" % (stage, config, set_name, seed), "stage": stage, "config": config,
            "phi_cfg": phi_cfg, "train_set": set_name, "train_exps": list(exps), "seed": int(seed),
            "learning_rate": float(lr), "gamma": float(gamma)}


def enumerate_tasks(stage, dev, rl=None, phi_match=None):
    """Seed-major order, so shards by index get a balanced mix of configurations and sets."""
    rl = rl or load_config("rl")
    if stage not in STAGES:
        raise ValueError("stage must be one of " + ", ".join(STAGES))
    seeds, sets = list(rl["training_seeds"]), training_sets(dev)
    jobs = []
    if stage == "tune":
        t = rl["tuning"]
        phi = {"phi": float(t["phi_pi16"]) * co.PI16}
        for s in seeds[:int(t["seeds"])]:
            for lr in t["learning_rate"]:
                for g in t["gamma"]:
                    for name, exps in sets[:-1]:
                        jobs.append(_task(stage, "rl_tune_lr%g_g%g" % (lr, g), phi, name, exps, s, lr, g))
        return jobs
    hp = rl.get("selected_hp")
    if not hp:
        raise SystemExit("config/rl.yaml selected_hp is not set (fixed from rl_tune_v1 before " + stage + ")")
    if stage == "train":
        configs = [(fixed_name(p), {"phi": float(p) * co.PI16}) for p in rl["fixed_phi_pi16"]]
        configs += [(dyn_name(c), {k: float(c[k]) for k in ("window_s", "phi_lo_pi16", "phi_hi_pi16")})
                    for c in rl["dynamic_configs"]]
    else:
        if phi_match is None:
            raise SystemExit("stage match needs --phi-match ")
        configs = [("rl_match_%.4f" % phi_match, {"phi": round(float(phi_match), 4)})]
    for s in seeds:
        for name, phi in configs:
            for set_name, exps in sets:
                jobs.append(_task(stage, name, phi, set_name, exps, s, hp["learning_rate"], hp["gamma"]))
    return jobs


def peak_memory_mb():
    """Peak working set (Windows) or peak resident size (Linux) of this process, in MB; None if unavailable."""
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class PMC(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
            k32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
            c = PMC()
            c.cb = ctypes.sizeof(PMC)
            ok = psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
            return round(c.PeakWorkingSetSize / 2 ** 20, 1) if ok else None
        import resource
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)
    except Exception:
        return None


def rl_versions():
    return {p: version(p) for p in RL_PACKAGES}


def md5_file(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def run_task(task, ctx, cfg=None):
    """Train one policy; write policies/<task_id>.npz atomically; return the record for the task JSON."""
    cache = ctx.setdefault("_slots", {})
    key = task["train_set"]
    if key not in cache:
        data = {e: ctx["data"][e] for e in task["train_exps"]}
        cache[key] = E.build_slots(data, key, ctx["table"], ctx["noise"], ctx["thresholds"], ctx["policy"],
                                   ctx["phi_thr"], ctx["t_start"])
    rl = load_config("rl")
    w, rec, _ = T.train_policy(cache[key], task["phi_cfg"], task["seed"], ctx["steps"], task["learning_rate"],
                               task["gamma"], window_s=ctx.get("window_s"), action_map=rl["action_map"],
                               normalize=rl["normalize"])      # (plan v7)
    out = Path(ctx["run_dir"]) / "policies" / (task["task_id"] + ".npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(task["task_id"] + ".partial.npz")
    w.to_npz(tmp)
    os.replace(tmp, out)
    rec.update(policy_file="policies/" + out.name, policy_md5=md5_file(out), peak_memory_mb=peak_memory_mb(),
               rl_packages=rl_versions(), n_slots=len(cache[key]))
    return rec
