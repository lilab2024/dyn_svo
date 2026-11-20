"""Calibration task list, shards, resumable execution and summary."""
import json
import os
import time
from pathlib import Path

import pandas as pd

from dyn_svo import provenance as pv
from dyn_svo.calibration import fit, segments
from dyn_svo.data import loader as ld
from dyn_svo.data import split as sp

CONFIGS = ("sim", "calibration", "data_split", "data_manifest")
REGIMES = ("I", "II")
VARIANTS = ("nodelay", "delay")


def folds(split=None):
    """(name, train experiments, held-out experiment or None): 5 dev LOEO folds + all-dev."""
    split = split if split is not None else sp.load_split()
    out = [("loeo_" + v, train, v) for train, v in sp.cv_folds(split)]
    return out + [("all_dev", sp.experiments("dev", split=split), None)]


def build_segment_bank(exps, segment_s, load=None):
    """Segments per (experiment, follower) and the instruction regime of each experiment."""
    load = load or ld.load_experiment
    bank, regimes = {}, {}
    for e in exps:
        df = load(e)
        regimes[e] = str(df["regime"].iloc[0])
        pairs = ld.follower_pairs(df)
        for veh in sorted(int(v) for v in pairs["veh_id"].unique()):
            seg = segments.vehicle_segments(pairs, veh, segment_s)
            if segments.n_segments(seg) > 0:
                bank[(e, veh)] = seg
    return bank, regimes


def enumerate_tasks(bank, regimes, fold_list):
    """Deterministic task list: fold x regime x (vehicles, then 'pooled') x variant."""
    tasks = []
    for fold_name, train, held in fold_list:
        for regime in REGIMES:
            train_r = [e for e in train if regimes.get(e) == regime]
            if not train_r:
                continue
            held_r = held if (held is not None and regimes.get(held) == regime) else None
            vehs = sorted({v for (e, v) in bank if e in train_r})
            for veh in [str(v) for v in vehs] + ["pooled"]:
                for variant in VARIANTS:
                    tasks.append({"task_id": "__".join([fold_name, regime, veh, variant]),
                                  "fold": fold_name, "regime": regime, "veh": veh,
                                  "variant": variant, "train": train_r, "held_out": held_r})
    return tasks


def shard(tasks, k, n):
    if not 0 <= k < n:
        raise ValueError("shard index out of range")
    return [t for i, t in enumerate(tasks) if i % n == k]


def task_segments(bank, task, which):
    exps = task["train"] if which == "train" else ([task["held_out"]] if task["held_out"] else [])
    keys = [(e, v) for (e, v) in sorted(bank)
            if e in exps and (task["veh"] == "pooled" or str(v) == task["veh"])]
    return segments.stack([bank[k] for k in keys]) if keys else None


def run_task(task, bank, cfg=None):
    train = task_segments(bank, task, "train")
    res = fit.fit_idm(train, with_delay=(task["variant"] == "delay"), cfg=cfg)
    held = task_segments(bank, task, "held")
    if held is not None:
        rmse, col = fit.spacing_rmse(held, res["params"], res["delay_steps"])
        res.update(held_rmse_m=rmse, held_collided_share=col, held_n_segments=segments.n_segments(held))
    else:
        res.update(held_rmse_m=None, held_collided_share=None, held_n_segments=0)
    return res


def write_json_atomic(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def run_shard(run_dir, tasks, bank, prov, cfg=None, max_tasks=None, progress_s=300, log=print,
              task_fn=None):
    """Run tasks not yet done; refuse to mix results from different code/settings."""
    task_fn = task_fn or run_task
    tdir = Path(run_dir) / "tasks"
    todo = []
    for t in tasks:
        p = tdir / (t["task_id"] + ".json")
        if p.exists():
            with open(p, encoding="utf-8") as f:
                old = json.load(f)
            if not pv.compatible(old["provenance"], prov):
                raise pv.ProvenanceMismatch("Existing result from other code/settings: " + p.name)
        else:
            todo.append(t)
    if max_tasks is not None:
        todo = todo[:max_tasks]
    log("tasks in shard: %d, already done: %d, to run now: %d"
        % (len(tasks), len(tasks) - len([t for t in tasks if not (tdir / (t["task_id"] + ".json")).exists()]),
           len(todo)))
    t0 = last = time.time()
    for i, t in enumerate(todo, 1):
        res = task_fn(t, bank, cfg)
        write_json_atomic(tdir / (t["task_id"] + ".json"), {"task": t, "result": res, "provenance": prov})
        now = time.time()
        if now - last >= progress_s or i == len(todo):
            el = now - t0
            log("done %d/%d | elapsed %.0f s | %.1f s/task | eta %.0f s"
                % (i, len(todo), el, el / i, el / i * (len(todo) - i)))
            last = now
    return len(todo)


def summarize(run_dir, n_expected=None):
    """Merge task JSONs into params.csv; return the frame and the number of missing tasks."""
    rows = []
    for p in sorted((Path(run_dir) / "tasks").glob("*.json")):
        with open(p, encoding="utf-8") as f:
            rec = json.load(f)
        t, r = rec["task"], rec["result"]
        row = {k: t[k] for k in ("task_id", "fold", "regime", "veh", "variant", "held_out")}
        row.update(r["params"])
        row.update({k: r[k] for k in ("delay_steps", "rmse_m", "collided_share", "n_segments",
                                      "held_rmse_m", "held_n_segments", "nfev", "seconds")})
        row.update(commit=rec["provenance"]["commit"], dirty=rec["provenance"]["dirty"],
                   data_check=rec["provenance"]["data_check"])
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(Path(run_dir) / "params.csv", index=False)
    missing = None if n_expected is None else n_expected - len(df)
    return df, missing
