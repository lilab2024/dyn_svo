"""config/decision_constants.yaml agrees with the data split and holds the decision constants of the evaluation."""
from dyn_svo.config import load_config
from dyn_svo.data import split as sp


def test_constants_match_split_config():
    c = load_config("decision_constants")
    s = sp.load_split()
    assert c["dev_experiments"] == s["roles"]["dev"]
    assert c["test_experiments"] == s["roles"]["test"]
    assert c["drop_first_s"] == s["segments"]["drop_first_s"]


def test_key_numbers():
    c = load_config("decision_constants")
    assert c["warmup_s"] == 30
    assert c["fixed_phi_grid_points"] == 9 == c["dynamic_phi_configs"]
    assert c["rl_seeds"] == 10
    assert c["min_for_reduction"] == 0.05
    assert c["chain_speed_noninferiority"] == 0.98
    assert c["llm_phi_interval_s"] == 5
    assert c["plan_version"] == 7
    assert c["rl_fixed_phi_points"] == 5 == c["rl_dynamic_configs"]
