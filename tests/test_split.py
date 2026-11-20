import pytest
import yaml

from dyn_svo.data import split as sp


def test_repo_split_matches_d008():
    s = sp.load_split()
    assert s["roles"]["dev"] == ["A", "C", "E", "F", "G"]
    assert s["roles"]["test"] == ["B", "D", "H"]
    assert sorted(s["experiments"]) == list("ABCDEFGH")


def test_repo_segment_rules_match_d007():
    s = sp.load_split()
    assert s["segments"]["drop_first_s"] == 30
    assert s["segments"]["intervention_s"] == {"F": 59, "G": 312, "H": 191}
    assert s["segments"]["exclude_follower_after_intervention"] == [20]
    assert s["fuel"]["missing_full"] == {"D": [5, 6], "H": [4, 11]}
    assert s["ring_length_nominal_m"] == 260


def test_dev_is_default_and_unlocked():
    assert sp.experiments() == ["A", "C", "E", "F", "G"]
    assert sp.check_access(["A", "C"]) is False


@pytest.mark.parametrize("role", ["test", "all"])
def test_test_set_locked(role):
    with pytest.raises(sp.TestSetLocked):
        sp.experiments(role)


def test_test_set_with_explicit_flag():
    assert sp.experiments("test", allow_test=True) == ["B", "D", "H"]
    assert sp.check_access(["A", "H"], allow_test=True) is True


def test_mixed_request_locked():
    with pytest.raises(sp.TestSetLocked):
        sp.check_access(["A", "B"])


def test_unknown_role():
    with pytest.raises(ValueError):
        sp.experiments("train")


def test_cv_folds_never_touch_test():
    folds = sp.cv_folds()
    assert [v for _, v in folds] == ["A", "C", "E", "F", "G"]
    for train, val in folds:
        assert val not in train
        assert sorted(train + [val]) == ["A", "C", "E", "F", "G"]
        assert not set(train) & {"B", "D", "H"}


def _write(tmp_path, dev, test):
    s = {"experiments": list("ABCDEFGH"), "roles": {"dev": dev, "test": test},
         "cv": "leave_one_experiment_out"}
    p = tmp_path / "split.yaml"
    p.write_text(yaml.safe_dump(s), encoding="utf-8")
    return p


def test_overlap_rejected(tmp_path):
    with pytest.raises(ValueError):
        sp.load_split(_write(tmp_path, ["A", "B", "C", "E", "F", "G"], ["B", "D", "H"]))


def test_missing_experiment_rejected(tmp_path):
    with pytest.raises(ValueError):
        sp.load_split(_write(tmp_path, ["A", "C", "E", "F"], ["B", "D", "H"]))
