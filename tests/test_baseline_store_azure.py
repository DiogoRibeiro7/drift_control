import pandas as pd
import pytest

from drift_control.baseline_store import AzureBlobBaselineStore


class _FakeDownloader:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def readall(self) -> bytes:
        return self._data


class _FakeBlobClient:
    def __init__(self, name: str, objects: dict[str, bytes]) -> None:
        self._name = name
        self._objects = objects

    def exists(self) -> bool:
        return self._name in self._objects


class _FakeBlobObj:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeContainerClient:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def upload_blob(self, name: str, data: bytes, overwrite: bool) -> None:
        self.objects[name] = data

    def download_blob(self, key: str) -> _FakeDownloader:
        return _FakeDownloader(self.objects[key])

    def get_blob_client(self, key: str) -> _FakeBlobClient:
        return _FakeBlobClient(key, self.objects)

    def list_blobs(self, name_starts_with: str):
        for key in sorted(self.objects.keys()):
            if key.startswith(name_starts_with):
                yield _FakeBlobObj(key)

    def delete_blob(self, key: str) -> None:
        self.objects.pop(key, None)


class _FakeBlobServiceClient:
    def __init__(self) -> None:
        self.container = _FakeContainerClient()

    def get_container_client(self, container_name: str) -> _FakeContainerClient:
        return self.container


def test_azure_store_save_load_list_metadata_delete():
    data = pd.DataFrame({"a": [1, 2, 3]})
    fake = _FakeBlobServiceClient()
    store = AzureBlobBaselineStore(container="my-container", prefix="baselines", blob_service_client=fake)

    path = store.save(data, name="data", version="1", owner="alice", training_job_id="job-1", fmt="csv")
    assert path == "azure://my-container/baselines/data_v1.csv"

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


def test_azure_store_integrity_check_fails_on_tamper():
    data = pd.DataFrame({"a": [1, 2, 3]})
    fake = _FakeBlobServiceClient()
    store = AzureBlobBaselineStore(container="my-container", prefix="baselines", blob_service_client=fake)
    store.save(data, name="data", version="1", fmt="csv")

    fake.container.objects["baselines/data_v1.csv"] = b"a\n999\n"
    with pytest.raises(ValueError, match="Integrity check failed"):
        store.load("data", "1", verify_integrity=True)
