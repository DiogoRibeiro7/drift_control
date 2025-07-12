import os
import pandas as pd

class BaselineManager:
    """Simple helper for saving and loading baseline datasets with versioning."""

    def __init__(self, directory: str = "baselines") -> None:
        self.directory = directory
        os.makedirs(self.directory, exist_ok=True)

    def _path(self, name: str, version: str) -> str:
        filename = f"{name}_v{version}.csv"
        return os.path.join(self.directory, filename)

    def save_baseline(self, data: pd.DataFrame, name: str, version: str) -> str:
        """Save the baseline dataset and return the file path."""
        path = self._path(name, version)
        data.to_csv(path, index=False)
        return path

    def load_baseline(self, name: str, version: str) -> pd.DataFrame:
        """Load a baseline dataset by name and version."""
        path = self._path(name, version)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Baseline {name} v{version} not found")
        return pd.read_csv(path)
