import shutil
import subprocess
import pandas as pd
from typing import Literal

from drift_control.baseline_store import BaselineStore, LocalBaselineStore


class BaselineManager:
    """Simple helper for saving and loading baseline datasets with versioning."""

    def __init__(
        self,
        directory: str = "baselines",
        default_format: Literal["parquet", "csv"] = "parquet",
        store: BaselineStore | None = None,
    ) -> None:
        self.directory = directory
        self.default_format = default_format
        self.store = store or LocalBaselineStore(directory=directory, default_format=default_format)

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
        return self.store.save(
            data=data,
            name=name,
            version=version,
            owner=owner,
            training_job_id=training_job_id,
            fmt=fmt,
        )

    def load_baseline(self, name: str, version: str, verify_integrity: bool = True) -> pd.DataFrame:
        """Load a baseline dataset by name and version."""
        return self.store.load(name=name, version=version, verify_integrity=verify_integrity)

    def list_baselines(self, name: str | None = None) -> list[str]:
        """List available baseline identifiers as 'name@version'."""
        return self.store.list(name=name)

    def get_metadata(self, name: str, version: str) -> dict:
        """Load metadata for a saved baseline."""
        return self.store.metadata(name=name, version=version)

    def delete_baseline(self, name: str, version: str) -> None:
        """Delete a baseline data file and metadata sidecar if present."""
        self.store.delete(name=name, version=version)

    def save_with_dvc(self, data: pd.DataFrame, name: str, version: str) -> str:
        """Save the baseline dataset and track it with DVC."""
        path = self.save_baseline(data, name, version)
        if not shutil.which("dvc"):
            raise RuntimeError("dvc executable not found")
        subprocess.run(["dvc", "add", path], check=True)
        return path
