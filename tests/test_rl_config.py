"""config/rl.yaml matches analysis plan v6 and the v5 grid."""
import yaml

from dyn_svo.config import load_config
from dyn_svo.data.manifest import REPO_ROOT


def _front():
    return yaml.safe_load((REPO_ROOT / "config" / "decision_constants.yaml").read_text(encoding="utf-8"))


def test_rl_counts_match_plan():
    rl, front = load_config("rl"), _front()
    assert len(rl["fixed_phi_pi16"]) == front["rl_fixed_phi_points"]
    assert len(rl["dynamic_configs"]) == front["rl_dynamic_configs"]
    assert len(rl["training_seeds"]) == front["rl_seeds"] == len(set(rl["training_seeds"]))
    assert rl["noise_seed_offset"] > max(range(load_config("eval")["seeds"]))
    assert isinstance(rl["training_steps"], int) and rl["training_steps"] > 0   # fixed from the timing pilot
    assert rl["selected_hp"] == {"learning_rate": 3e-4, "gamma": 0.98}   # rl_tune_v2 and gate
    assert rl["action_map"] == "centred" and rl["normalize"] == "reward"   # (plan v7)


def test_rl_subsets_inside_mpc_grids():
    rl, co, mpc = load_config("rl"), load_config("cooperation"), load_config("mpc")
    assert set(rl["fixed_phi_pi16"]) <= set(range(mpc["fixed_phi_grid_points"]))
    la = co["leader_activity"]
    grid = {(float(w), float(lo), float(hi)) for w in la["window_s"] for lo, hi in la["phi_ranges_pi16"]}
    dyn = [(c["window_s"], c["phi_lo_pi16"], c["phi_hi_pi16"]) for c in rl["dynamic_configs"]]
    assert len(set(dyn)) == 5 and set(dyn) <= grid
    t = rl["tuning"]
    assert t["phi_pi16"] in rl["fixed_phi_pi16"] and len(t["learning_rate"]) * len(t["gamma"]) <= 6
