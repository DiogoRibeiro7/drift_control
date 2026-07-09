import pandas as pd
import pytest

from drift_control.baseline_manager import BaselineManager


def test_save_and_load(tmp_path):
    data = pd.DataFrame({'a': [1, 2, 3]})
    manager = BaselineManager(directory=tmp_path)
    path = manager.save_baseline(data, name="data", version="1")
    assert path.endswith("data_v1.csv") or path.endswith("data_v1.parquet")
    loaded = manager.load_baseline("data", "1")
    pd.testing.assert_frame_equal(data, loaded)


def test_baseline_metadata_and_listing(tmp_path):
    data = pd.DataFrame({'a': [1, 2, 3]})
    manager = BaselineManager(directory=tmp_path)
    manager.save_baseline(data, name="data", version="1", owner="alice", training_job_id="job-1")
    ids = manager.list_baselines()
    assert "data@1" in ids
    meta = manager.get_metadata("data", "1")
    assert meta["owner"] == "alice"
    assert meta["training_job_id"] == "job-1"
    assert meta["row_count"] == 3
    assert meta["format"] in {"csv", "parquet"}
    assert isinstance(meta["dataset_sha256"], str)


def test_baseline_integrity_check_fails_on_tamper(tmp_path):
    data = pd.DataFrame({'a': [1, 2, 3]})
    manager = BaselineManager(directory=tmp_path)
    path = manager.save_baseline(data, name="data", version="1")
    # Tamper with persisted CSV after metadata hash is stored.
    pd.DataFrame({'a': [999]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Integrity check failed"):
        manager.load_baseline("data", "1", verify_integrity=True)


def test_delete_baseline_removes_files(tmp_path):
    data = pd.DataFrame({'a': [1, 2, 3]})
    manager = BaselineManager(directory=tmp_path)
    manager.save_baseline(data, name="data", version="1")
    manager.delete_baseline("data", "1")
    with pytest.raises(FileNotFoundError):
        manager.load_baseline("data", "1")
    with pytest.raises(FileNotFoundError):
        manager.get_metadata("data", "1")


def test_manager_uses_injected_store(tmp_path):
    calls = {"load": None}

    class _FakeStore:
        def save(self, **kwargs):
            return "fake-path"

        def load(self, *, name, version, verify_integrity=True):
            calls["load"] = (name, version, verify_integrity)
            return pd.DataFrame({"a": [1]})

        def list(self, *, name=None):
            return ["data@1"]

        def metadata(self, *, name, version):
            return {"name": name, "version": version}

        def delete(self, *, name, version):
            calls["delete"] = (name, version)

    manager = BaselineManager(directory=tmp_path, store=_FakeStore())
    loaded = manager.load_baseline("data", "1", verify_integrity=False)
    pd.testing.assert_frame_equal(loaded, pd.DataFrame({"a": [1]}))
    assert calls["load"] == ("data", "1", False)
