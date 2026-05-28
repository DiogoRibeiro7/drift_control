import pandas as pd
import pytest
from drift_control.baseline_manager import BaselineManager


def test_save_and_load(tmp_path):
    data = pd.DataFrame({'a': [1, 2, 3]})
    manager = BaselineManager(directory=tmp_path)
    path = manager.save_baseline(data, name="data", version="1")
    assert path.endswith("data_v1.csv")
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
    path = manager.save_baseline(data, name="data", version="1")
    manager.delete_baseline("data", "1")
    with pytest.raises(FileNotFoundError):
        manager.load_baseline("data", "1")
    with pytest.raises(FileNotFoundError):
        manager.get_metadata("data", "1")
