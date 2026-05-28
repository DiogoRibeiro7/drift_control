import os
import re
import shutil
import subprocess
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from typing import Literal

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

    def __init__(
        self,
        directory: str = "baselines",
        default_format: Literal["parquet", "csv"] = "parquet",
    ) -> None:
        self.directory = directory
        self.default_format = default_format
        os.makedirs(self.directory, exist_ok=True)

    def _path(self, name: str, version: str, fmt: Literal["parquet", "csv"] = "csv") -> str:
        _validate_component(name, "name")
        _validate_component(version, "version")
        ext = "parquet" if fmt == "parquet" else "csv"
        filename = f"{name}_v{version}.{ext}"
        return os.path.join(self.directory, filename)

    @staticmethod
    def _file_sha256(path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def _meta_path(self, name: str, version: str, fmt: Literal["parquet", "csv"] = "csv") -> str:
        return self._path(name, version, fmt=fmt) + ".meta.json"

    @staticmethod
    def _parquet_available() -> bool:
        try:
            import pyarrow  # type: ignore # noqa: F401
            return True
        except Exception:
            try:
                import fastparquet  # type: ignore # noqa: F401
                return True
            except Exception:
                return False

    def _resolve_existing_path(self, name: str, version: str) -> tuple[str, Literal["parquet", "csv"]]:
        p_parquet = self._path(name, version, fmt="parquet")
        p_csv = self._path(name, version, fmt="csv")
        if os.path.exists(p_parquet):
            return p_parquet, "parquet"
        if os.path.exists(p_csv):
            return p_csv, "csv"
        raise FileNotFoundError(f"Baseline {name} v{version} not found")

    def save_baseline(
        self,
        data: pd.DataFrame,
        name: str,
        version: str,
        owner: str | None = None,
        training_job_id: str | None = None,
        fmt: Literal["parquet", "csv"] | None = None,
    ) -> str:
        """Save the baseline dataset and return the file path."""
        selected = fmt or self.default_format
        if selected == "parquet" and not self._parquet_available():
            selected = "csv"
        path = self._path(name, version, fmt=selected)
        if selected == "parquet":
            data.to_parquet(path, index=False)
        else:
            data.to_csv(path, index=False)
        meta = {
            "name": name,
            "version": version,
            "path": path,
            "format": selected,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "row_count": int(len(data)),
            "dataset_sha256": self._file_sha256(path),
            "owner": owner,
            "training_job_id": training_job_id,
        }
        with open(self._meta_path(name, version, fmt=selected), "w", encoding="utf-8") as f:
            json.dump(meta, f)
        return path

    def load_baseline(self, name: str, version: str, verify_integrity: bool = True) -> pd.DataFrame:
        """Load a baseline dataset by name and version."""
        path, resolved_fmt = self._resolve_existing_path(name, version)
        if verify_integrity:
            meta = self.get_metadata(name, version)
            expected = str(meta.get("dataset_sha256", ""))
            actual = self._file_sha256(path)
            if expected and expected != actual:
                raise ValueError(
                    f"Integrity check failed for baseline {name} v{version}: "
                    f"expected {expected}, got {actual}"
                )
        if resolved_fmt == "parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path)

    def list_baselines(self, name: str | None = None) -> list[str]:
        """List available baseline identifiers as 'name@version'."""
        out: list[str] = []
        for p in list(Path(self.directory).glob("*_v*.csv")) + list(Path(self.directory).glob("*_v*.parquet")):
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
        meta_path_csv = self._meta_path(name, version, fmt="csv")
        meta_path_parquet = self._meta_path(name, version, fmt="parquet")
        meta_path = meta_path_parquet if os.path.exists(meta_path_parquet) else meta_path_csv
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata for baseline {name} v{version} not found")
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def delete_baseline(self, name: str, version: str) -> None:
        """Delete a baseline data file and metadata sidecar if present."""
        for fmt in ("csv", "parquet"):
            path = self._path(name, version, fmt=fmt)
            meta_path = self._meta_path(name, version, fmt=fmt)
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
