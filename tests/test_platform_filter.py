""" safety filter timing, filter floor buffer, upward jerk limit."""
import ast
import copy
from pathlib import Path

import numpy as np
import pytest

from dyn_svo.config import load_config
from dyn_svo.control import filter_check as fc
from dyn_svo.control import platform as pf
from dyn_svo.sim import idm

C = load_config("av")
B = C["safety"]["s_min_buffer_m"]
THRESHOLDS = (1.4945632197013647, 0.8008237432643556)   # guardrails_v1 min_gap I, II


class Fixed:
    def __init__(self, a=None, v=None):
        self.a, self.v = a, v

    def reset(self, info):
        pass

    def act(self, obs):
        return {"a": self.a} if self.a is not None else {"v": self.v}


@pytest.mark.parametrize("thr", THRESHOLDS)
def test_buffer_is_smallest_passing_multiple(thr):
    assert fc.smallest_buffer(thr) == pytest.approx(B)
    assert fc.worst_margin(thr, B) >= 0.0


def test_floor_threshold_and_convergence_time():
    p = pf.AVPlatform(Fixed(a=0.0), s_min=1.5)
    assert p.s_min == 1.5 and p.s_floor == pytest.approx(1.5 + B)
    assert p.t_conv == pytest.approx(C["safety"]["reaction_s"] - C["actuation_lag_s"])
    bad = copy.deepcopy(C)
    bad["actuation_lag_s"] = bad["safety"]["reaction_s"]
    with pytest.raises(ValueError):
        pf.AVPlatform(Fixed(a=0.0), s_min=1.0, cfg=bad)


def test_command_never_rises_faster_than_jerk_limit():
    rng = np.random.default_rng(45)
    step = C["jerk_max"] * idm.DT
    for _ in range(300):
        p = pf.AVPlatform(Fixed(a=rng.uniform(-3.5, 2.0)), s_min=1.0)
        for _ in range(5):
            before = p.a_cmd_prev
            p.act({"v": rng.uniform(0, 8), "gap": rng.uniform(0.5, 40), "v_leader": rng.uniform(0, 8)})
            assert p.a_cmd_prev <= before + step + 1e-12


def test_filter_braking_bypasses_jerk_limit_down_to_emergency_decel():
    p = pf.AVPlatform(Fixed(a=0.0), s_min=1.0)
    p.act({"v": 10.0, "gap": 1.2, "v_leader": 0.0})
    assert p.a_cmd_prev == pytest.approx(-C["safety"]["emergency_decel"])
    assert p.stats()["safety_share"] == 1.0


def test_evaluate_docstring_follows_run_name_rule():
    src = (Path(__file__).resolve().parents[1] / "scripts" / "evaluate.py").read_text(encoding="utf-8")
    cmds = [l.strip() for l in ast.get_docstring(ast.parse(src)).splitlines() if "scripts/evaluate.py" in l]
    assert cmds
    for line in cmds:
        assert ("--official" in line) != ("--run-name debug_" in line), line
