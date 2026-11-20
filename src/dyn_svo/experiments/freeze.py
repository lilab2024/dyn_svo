"""Frozen test run: the freeze file, its checks against the development results,
the frozen task list. The test experiments are loaded only through scripts/evaluate.py --stage test."""
import hashlib
import json
from pathlib import Path

from dyn_svo.config import load_config
from dyn_svo.experiments import configs as cf

MATCHED = "fixed_match"


def load_freeze():
    return load_config("test_freeze")


def verify_freeze(freeze, results_root):
    """Problems (empty list = ok): every recorded md5 matches; each variant's best fixed and dynamic
    configuration equal the final selection of its source summary; phi of the matched control equals
    the source phi_match (4 decimals)."""
    root = Path(results_root)
    problems = []
    for rel, md5 in freeze["dev_sources"].items():
        p = root / rel
        if not p.exists():
            problems.append("missing " + rel)
        elif hashlib.md5(p.read_bytes()).hexdigest() != md5:
            problems.append("md5 differs " + rel)
    if problems:
        return problems
    for name, v in freeze["variants"].items():
        if v["selection_source"] not in freeze["dev_sources"]:
            problems.append(name + ": selection source not in dev_sources")
            continue
        s = json.loads((root / v["selection_source"]).read_text(encoding="utf-8"))
        if s.get("variant", "primary") != name:
            problems.append(name + ": source variant " + str(s.get("variant")))
        sel = s["final_selection"]
        if sel.get("mpc_fixed") != v["best_fixed"] or sel.get("mpc_dynamic") != v["dynamic"]:
            problems.append(name + ": selection differs from source " + str(sel))
        if (v["matched"] is None) != (v["dynamic"] is None):
            problems.append(name + ": matched control needs a dynamic selection ")
        if v["matched"] is not None:
            m = json.loads((root / v["matched"]["source"]).read_text(encoding="utf-8"))
            if m.get("dynamic") != v["dynamic"] or round(float(m["phi_match"]), 4) != float(v["matched"]["phi"]):
                problems.append(name + ": matched control differs from " + v["matched"]["source"])
        need = {"hdv", v["best_fixed"]} | ({v["dynamic"], MATCHED} if v["dynamic"] else set())
        if not need <= set(v["configs"]):
            problems.append(name + ": configs miss " + str(sorted(need - set(v["configs"]))))
    return problems


def config_by_name(freeze, variant, name):
    if name == MATCHED:
        return {"name": MATCHED, "family": "matched", "phi": float(freeze["variants"][variant]["matched"]["phi"]),
                "rank": [0]}
    by_name = {c["name"]: c for c in cf.all_configs()}
    return by_name[name]


def test_tasks(freeze, variant):
    """Frozen tasks of one variant: its configurations x the test experiments."""
    v = freeze["variants"][variant]
    return [{"task_id": n + "__" + e, "config": config_by_name(freeze, variant, n), "exp": e, "variant": variant,
             "stage": "test"}
            for n in v["configs"] for e in freeze["test_experiments"]]
