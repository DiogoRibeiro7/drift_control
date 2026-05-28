import os
import re
import shutil
import subprocess
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9._-]+$")


def _validate_component(value: str, field: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    if not _SAFE_COMPONENT.match(value):
        raise ValueError(
            f"{field} may only contain letters, digits, '.', '_' and '-'; "
            f"got {value!r}"
        )


class BaselineManager:
    """Simple helper for saving and loading baseline datasets with versioning."""

    def __init__(self, directory: str = "baselines") -> None:
        self.directory = directory
        os.makedirs(self.directory, exist_ok=True)

    def _path(self, name: str, version: str) -> str:
        _validate_component(name, "name")
        _validate_component(version, "version")
        filename = f"{name}_v{version}.csv"
        return os.path.join(self.directory, filename)

    @staticmethod
    def _file_sha256(path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def _meta_path(self, name: str, version: str) -> str:
        return self._path(name, version) + ".meta.json"

    def save_baseline(
        self,
        data: pd.DataFrame,
        name: str,
        version: str,
        owner: str | None = None,
        training_job_id: str | None = None,
    ) -> str:
        """Save the baseline dataset and return the file path."""
        path = self._path(name, version)
        data.to_csv(path, index=False)
        meta = {
            "name": name,
            "version": version,
            "path": path,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "row_count": int(len(data)),
            "dataset_sha256": self._file_sha256(path),
            "owner": owner,
            "training_job_id": training_job_id,
        }
        with open(self._meta_path(name, version), "w", encoding="utf-8") as f:
            json.dump(meta, f)
        return path

    def load_baseline(self, name: str, version: str, verify_integrity: bool = True) -> pd.DataFrame:
        """Load a baseline dataset by name and version."""
        path = self._path(name, version)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Baseline {name} v{version} not found")
        if verify_integrity:
            meta = self.get_metadata(name, version)
            expected = str(meta.get("dataset_sha256", ""))
            actual = self._file_sha256(path)
            if expected and expected != actual:
                raise ValueError(
                    f"Integrity check failed for baseline {name} v{version}: "
                    f"expected {expected}, got {actual}"
                )
        return pd.read_csv(path)

    def list_baselines(self, name: str | None = None) -> list[str]:
        """List available baseline identifiers as 'name@version'."""
        out: list[str] = []
        for p in Path(self.directory).glob("*_v*.csv"):
            stem = p.stem
            if "_v" not in stem:
                continue
            n, v = stem.rsplit("_v", 1)
            if name is not None and n != name:
                continue
            out.append(f"{n}@{v}")
        return sorted(out)

    def get_metadata(self, name: str, version: str) -> dict:
        """Load metadata for a saved baseline."""
        meta_path = self._meta_path(name, version)
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata for baseline {name} v{version} not found")
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def delete_baseline(self, name: str, version: str) -> None:
        """Delete a baseline data file and metadata sidecar if present."""
        path = self._path(name, version)
        meta_path = self._meta_path(name, version)
        if os.path.exists(path):
            os.remove(path)
        if os.path.exists(meta_path):
            os.remove(meta_path)

    def save_with_dvc(self, data: pd.DataFrame, name: str, version: str) -> str:
        """Save the baseline dataset and track it with DVC."""
        path = self.save_baseline(data, name, version)
        if not shutil.which("dvc"):
            raise RuntimeError("dvc executable not found")
        subprocess.run(["dvc", "add", path], check=True)
        return path
