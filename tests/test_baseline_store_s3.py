import pandas as pd
import pytest

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
