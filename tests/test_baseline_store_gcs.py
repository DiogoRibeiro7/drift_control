import pandas as pd
import pytest

from drift_control.baseline_store import GCSBaselineStore


class _FakeBlob:
    def __init__(self, name: str, objects: dict[str, bytes]) -> None:
        self.name = name
        self._objects = objects

    def upload_from_string(self, payload: bytes) -> None:
        self._objects[self.name] = payload

    def download_as_bytes(self) -> bytes:
        return self._objects[self.name]

    def exists(self) -> bool:
        return self.name in self._objects

    def delete(self) -> None:
        self._objects.pop(self.name, None)


class _FakeBucket:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def blob(self, name: str) -> _FakeBlob:
        return _FakeBlob(name, self.objects)

    def list_blobs(self, prefix: str):
        for key in sorted(self.objects.keys()):
            if key.startswith(prefix):
                yield _FakeBlob(key, self.objects)


class _FakeGCSClient:
    def __init__(self) -> None:
        self._bucket = _FakeBucket()

    def bucket(self, bucket_name: str) -> _FakeBucket:
        return self._bucket


def test_gcs_store_save_load_list_metadata_delete():
    data = pd.DataFrame({"a": [1, 2, 3]})
    fake = _FakeGCSClient()
    store = GCSBaselineStore(bucket="my-bucket", prefix="baselines", gcs_client=fake)

    path = store.save(data, name="data", version="1", owner="alice", training_job_id="job-1", fmt="csv")
    assert path == "gs://my-bucket/baselines/data_v1.csv"

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


def test_gcs_store_integrity_check_fails_on_tamper():
    data = pd.DataFrame({"a": [1, 2, 3]})
    fake = _FakeGCSClient()
    store = GCSBaselineStore(bucket="my-bucket", prefix="baselines", gcs_client=fake)
    store.save(data, name="data", version="1", fmt="csv")

    fake._bucket.objects["baselines/data_v1.csv"] = b"a\n999\n"
    with pytest.raises(ValueError, match="Integrity check failed"):
        store.load("data", "1", verify_integrity=True)
