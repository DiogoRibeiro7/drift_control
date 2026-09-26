import json

import pandas as pd
import pytest
from dataexcept import DataLoadingError, FileWriteError

from drift_control.baseline_store import S3BaselineStore


class _FakeS3Body:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data


class _FakeS3Paginator:
    def __init__(self, objects: dict[str, bytes]) -> None:
        self._objects = objects

    def paginate(self, Bucket: str, Prefix: str):  # noqa: N803
        keys = [k for k in self._objects.keys() if k.startswith(Prefix)]
        yield {"Contents": [{"Key": k} for k in keys]}


class _FakeS3Client:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def put_object(self, Bucket: str, Key: str, Body: bytes):  # noqa: N803
        self.objects[Key] = Body

    def get_object(self, Bucket: str, Key: str):  # noqa: N803
        return {"Body": _FakeS3Body(self.objects[Key])}

    def head_object(self, Bucket: str, Key: str):  # noqa: N803
        if Key not in self.objects:
            raise KeyError(Key)
        return {"Key": Key}

    def delete_object(self, Bucket: str, Key: str):  # noqa: N803
        self.objects.pop(Key, None)

    def get_paginator(self, op_name: str):
        assert op_name == "list_objects_v2"
        return _FakeS3Paginator(self.objects)


def test_s3_store_save_load_list_metadata_delete():
    data = pd.DataFrame({"a": [1, 2, 3]})
    fake = _FakeS3Client()
    store = S3BaselineStore(bucket="my-bucket", prefix="baselines", s3_client=fake)

    path = store.save(
        data, name="data", version="1", owner="alice", training_job_id="job-1", fmt="csv"
    )
    assert path == "s3://my-bucket/baselines/data_v1.csv"

    loaded = store.load("data", "1")
    pd.testing.assert_frame_equal(data, loaded)

    assert store.list() == ["data@1"]
    meta = store.metadata("data", "1")
    assert meta["owner"] == "alice"
    assert meta["training_job_id"] == "job-1"
    assert meta["format"] == "csv"

    store.delete("data", "1")
    with pytest.raises(FileNotFoundError):
        store.load("data", "1")


def test_s3_store_integrity_check_fails_on_tamper():
    data = pd.DataFrame({"a": [1, 2, 3]})
    fake = _FakeS3Client()
    store = S3BaselineStore(bucket="my-bucket", prefix="baselines", s3_client=fake)
    store.save(data, name="data", version="1", fmt="csv")

    fake.objects["baselines/data_v1.csv"] = b"a\n999\n"
    with pytest.raises(ValueError, match="Integrity check failed"):
        store.load("data", "1", verify_integrity=True)


def test_s3_download_failure_preserves_remote_source(monkeypatch):
    fake = _FakeS3Client()
    store = S3BaselineStore(bucket="my-bucket", s3_client=fake)
    store.save(pd.DataFrame({"a": [1]}), "data", "1", fmt="csv")
    failure = PermissionError("access denied")

    def fail_download(**kwargs):
        raise failure

    monkeypatch.setattr(fake, "get_object", fail_download)
    with pytest.raises(DataLoadingError) as error:
        store.load("data", "1")

    assert error.value.source == "s3://my-bucket/baselines/data_v1.csv"
    assert error.value.__cause__ is failure


def test_s3_malformed_data_and_metadata_identify_the_object():
    fake = _FakeS3Client()
    store = S3BaselineStore(bucket="my-bucket", s3_client=fake)
    store.save(pd.DataFrame({"a": [1]}), "data", "1", fmt="csv")

    fake.objects["baselines/data_v1.csv"] = b""
    with pytest.raises(DataLoadingError) as data_error:
        store.load("data", "1", verify_integrity=False)
    assert data_error.value.source.endswith("/data_v1.csv")
    assert isinstance(data_error.value.__cause__, pd.errors.EmptyDataError)

    fake.objects["baselines/data_v1.csv.meta.json"] = b"{"
    with pytest.raises(DataLoadingError) as meta_error:
        store.metadata("data", "1")
    assert meta_error.value.source.endswith("/data_v1.csv.meta.json")
    assert isinstance(meta_error.value.__cause__, json.JSONDecodeError)


def test_s3_upload_failure_preserves_destination(monkeypatch):
    fake = _FakeS3Client()
    store = S3BaselineStore(bucket="my-bucket", s3_client=fake)
    failure = PermissionError("access denied")

    def fail_upload(**kwargs):
        raise failure

    monkeypatch.setattr(fake, "put_object", fail_upload)
    with pytest.raises(FileWriteError) as error:
        store.save(pd.DataFrame({"a": [1]}), "data", "1", fmt="csv")

    assert error.value.path == "s3://my-bucket/baselines/data_v1.csv"
    assert error.value.__cause__ is failure


def test_s3_metadata_upload_failure_identifies_sidecar(monkeypatch):
    fake = _FakeS3Client()
    store = S3BaselineStore(bucket="my-bucket", s3_client=fake)
    original_upload = fake.put_object
    failure = PermissionError("metadata upload denied")

    def fail_sidecar(**kwargs):
        if kwargs["Key"].endswith(".meta.json"):
            raise failure
        return original_upload(**kwargs)

    monkeypatch.setattr(fake, "put_object", fail_sidecar)
    with pytest.raises(FileWriteError) as error:
        store.save(pd.DataFrame({"a": [1]}), "data", "1", fmt="csv")

    assert error.value.path.endswith("/data_v1.csv.meta.json")
    assert error.value.__cause__ is failure


def test_s3_head_failure_is_not_mistaken_for_missing_baseline(monkeypatch):
    fake = _FakeS3Client()
    store = S3BaselineStore(bucket="my-bucket", s3_client=fake)
    failure = PermissionError("head denied")

    def fail_head(**kwargs):
        raise failure

    monkeypatch.setattr(fake, "head_object", fail_head)
    with pytest.raises(DataLoadingError) as error:
        store.load("data", "1")

    assert error.value.source == "s3://my-bucket/baselines/data_v1.parquet"
    assert error.value.__cause__ is failure


def test_s3_list_and_delete_failures_preserve_object_location(monkeypatch):
    fake = _FakeS3Client()
    store = S3BaselineStore(bucket="my-bucket", s3_client=fake)
    store.save(pd.DataFrame({"a": [1]}), "data", "1", fmt="csv")
    failure = ConnectionError("S3 unavailable")

    def fail_list(operation):
        raise failure

    monkeypatch.setattr(fake, "get_paginator", fail_list)
    with pytest.raises(DataLoadingError) as list_error:
        store.list()
    assert list_error.value.source == "s3://my-bucket/baselines/"
    assert list_error.value.__cause__ is failure

    def fail_delete(**kwargs):
        raise failure

    monkeypatch.setattr(fake, "delete_object", fail_delete)
    with pytest.raises(FileWriteError) as delete_error:
        store.delete("data", "1")
    assert delete_error.value.path.endswith("/data_v1.csv")
    assert delete_error.value.__cause__ is failure
