import pandas as pd
import pytest
from dataexcept import DataLoadingError, FileWriteError

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


def test_local_store_empty_csv_preserves_loading_cause(tmp_path):
    store = LocalBaselineStore(directory=tmp_path, default_format="csv")
    path = store.save(pd.DataFrame({"a": [1]}), name="data", version="1")
    (tmp_path / "data_v1.csv").write_text("", encoding="utf-8")

    with pytest.raises(DataLoadingError) as captured:
        store.load("data", "1", verify_integrity=False)

    assert captured.value.source == path
    assert isinstance(captured.value.__cause__, pd.errors.EmptyDataError)


def test_local_store_invalid_metadata_preserves_loading_cause(tmp_path):
    store = LocalBaselineStore(directory=tmp_path, default_format="csv")
    store.save(pd.DataFrame({"a": [1]}), name="data", version="1")
    meta_path = tmp_path / "data_v1.csv.meta.json"
    meta_path.write_text("{broken", encoding="utf-8")

    with pytest.raises(DataLoadingError) as captured:
        store.load("data", "1")

    assert captured.value.source == str(meta_path)
    assert isinstance(captured.value.__cause__, ValueError)


def test_local_store_invalid_metadata_encoding_preserves_loading_cause(tmp_path):
    store = LocalBaselineStore(directory=tmp_path, default_format="csv")
    store.save(pd.DataFrame({"a": [1]}), name="data", version="1")
    meta_path = tmp_path / "data_v1.csv.meta.json"
    meta_path.write_bytes(b"\xff")

    with pytest.raises(DataLoadingError) as error:
        store.metadata("data", "1")

    assert error.value.source == str(meta_path)
    assert isinstance(error.value.__cause__, UnicodeError)


def test_local_store_write_failure_preserves_original(tmp_path, monkeypatch):
    store = LocalBaselineStore(directory=tmp_path, default_format="csv")
    failure = PermissionError("read-only baseline directory")

    def fail_write(self: pd.DataFrame, path: str, *, index: bool) -> None:
        raise failure

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail_write)

    with pytest.raises(FileWriteError) as captured:
        store.save(pd.DataFrame({"a": [1]}), name="data", version="1")

    assert captured.value.path == str(tmp_path / "data_v1.csv")
    assert captured.value.__cause__ is failure


def test_local_store_directory_failure_preserves_destination(tmp_path):
    blocked = tmp_path / "blocked"
    blocked.write_text("ordinary file", encoding="utf-8")

    with pytest.raises(FileWriteError) as error:
        LocalBaselineStore(directory=blocked / "baselines")

    assert error.value.path == str(blocked / "baselines")
    assert isinstance(error.value.__cause__, OSError)


def test_local_store_delete_failure_preserves_destination(tmp_path, monkeypatch):
    store = LocalBaselineStore(directory=tmp_path, default_format="csv")
    path = store.save(pd.DataFrame({"a": [1]}), name="data", version="1")
    failure = PermissionError("read-only baseline directory")

    def fail_remove(target: str) -> None:
        raise failure

    monkeypatch.setattr("drift_control.baseline_store.os.remove", fail_remove)
    with pytest.raises(FileWriteError) as error:
        store.delete("data", "1")

    assert error.value.path == path
    assert error.value.__cause__ is failure
