import pandas as pd
import pytest

from drift_control.baseline_store import LocalBaselineStore


def test_store_save_and_load(tmp_path):
    data = pd.DataFrame({"a": [1, 2, 3]})
    store = LocalBaselineStore(directory=tmp_path)
    path = store.save(data, name="data", version="1")
    assert path.endswith("data_v1.csv") or path.endswith("data_v1.parquet")
    loaded = store.load("data", "1")
    pd.testing.assert_frame_equal(data, loaded)


def test_store_metadata_listing_and_delete(tmp_path):
    data = pd.DataFrame({"a": [1, 2, 3]})
    store = LocalBaselineStore(directory=tmp_path)
    store.save(data, name="data", version="1", owner="alice", training_job_id="job-1")
    assert "data@1" in store.list()
    meta = store.metadata("data", "1")
    assert meta["owner"] == "alice"
    assert meta["training_job_id"] == "job-1"
    store.delete("data", "1")
    with pytest.raises(FileNotFoundError):
        store.load("data", "1")
