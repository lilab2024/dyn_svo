import pytest

from dyn_svo.data import manifest as mf


def test_repo_manifest_consistent():
    m = mf.load_manifest()
    assert m["data_dir"] == "shared_data/arizona_ring"
    assert m["hash"] == "md5" and m["sampling_hz"] == 30
    assert sorted(m["files"]) == ["exp" + e + ".csv" for e in "ABCDEFGH"]
    for info in m["files"].values():
        assert len(info["md5"]) == 32 and all(c in "0123456789abcdef" for c in info["md5"])
        assert info["rows"] == info["n_veh"] * info["samples_per_veh"]
        assert info["duration_s"] == info["samples_per_veh"] / 30


def test_data_root_requires_env(monkeypatch):
    monkeypatch.delenv(mf.ENV_VAR, raising=False)
    with pytest.raises(RuntimeError):
        mf.data_root()


def _fake(tmp_path):
    folder = tmp_path / "shared_data" / "arizona_ring"
    folder.mkdir(parents=True)
    files = {}
    for name, content in [("expA.csv", b"a,b" + bytes([10]) + b"1,2"), ("expB.csv", b"xyz")]:
        (folder / name).write_bytes(content)
        files[name] = {"bytes": len(content), "md5": mf.md5_of(folder / name)}
    return {"data_dir": "shared_data/arizona_ring", "files": files}, folder


def test_verify_ok(tmp_path):
    m, _ = _fake(tmp_path)
    assert mf.verify_data("size", m, tmp_path) == "size_only"
    assert mf.verify_data("full", m, tmp_path) == "full_md5"


def test_same_size_change_caught_only_by_full(tmp_path):
    m, folder = _fake(tmp_path)
    (folder / "expB.csv").write_bytes(b"xyZ")
    assert mf.verify_data("size", m, tmp_path) == "size_only"
    with pytest.raises(mf.DataIntegrityError):
        mf.verify_data("full", m, tmp_path)


def test_size_change_caught(tmp_path):
    m, folder = _fake(tmp_path)
    (folder / "expB.csv").write_bytes(b"xyzz")
    with pytest.raises(mf.DataIntegrityError):
        mf.verify_data("size", m, tmp_path)


def test_missing_file_caught(tmp_path):
    m, folder = _fake(tmp_path)
    (folder / "expA.csv").unlink()
    with pytest.raises(mf.DataIntegrityError):
        mf.verify_data("size", m, tmp_path)


def test_bad_mode(tmp_path):
    m, _ = _fake(tmp_path)
    with pytest.raises(ValueError):
        mf.verify_data("fast", m, tmp_path)


def test_data_file_path(tmp_path):
    m, folder = _fake(tmp_path)
    assert mf.data_file("A", m, tmp_path) == folder / "expA.csv"
    with pytest.raises(KeyError):
        mf.data_file("Z", m, tmp_path)
