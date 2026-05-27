from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any


SUPPORTED_METHODS = {
    "psi",
    "ks",
    "mmd",
    "c2st",
    "cvm",
    "js",
    "wasserstein",
    "chi2cat",
    "tvdcat",
    "ensemble",
}
SUPPORTED_ENSEMBLE_METHODS = {"psi", "ks", "cvm", "js", "wasserstein", "chi2cat", "tvdcat"}
SUPPORTED_VOTE_MODES = {"majority", "any", "all"}
SUPPORTED_CORRECTIONS = {"none", "bonferroni", "bh"}


@dataclass(frozen=True)
class EnsembleConfig:
    methods: list[str]
    vote_mode: str = "majority"
    min_votes: int | None = None

    def __post_init__(self) -> None:
        unknown = [m for m in self.methods if m not in SUPPORTED_ENSEMBLE_METHODS]
        if unknown:
            raise ValueError(f"unknown methods: {unknown}")
        if self.vote_mode not in SUPPORTED_VOTE_MODES:
            raise ValueError("vote_mode must be one of: 'majority', 'any', 'all'")
        if self.min_votes is not None:
            if self.min_votes < 1 or self.min_votes > len(self.methods):
                raise ValueError("min_votes must be between 1 and the number of methods")


@dataclass(frozen=True)
class DriftCheckConfig:
    method: str = "psi"
    threshold: float | None = None
    correction: str = "none"
    ensemble: EnsembleConfig = field(
        default_factory=lambda: EnsembleConfig(methods=["psi", "ks", "cvm", "js"])
    )

    def __post_init__(self) -> None:
        if self.method not in SUPPORTED_METHODS:
            raise ValueError(f"unsupported method: {self.method}")
        if self.correction not in SUPPORTED_CORRECTIONS:
            raise ValueError("correction must be one of: none, bonferroni, bh")

    @staticmethod
    def from_cli(
        method: str,
        threshold: float | None,
        correction: str,
        ensemble_methods: str,
        vote_mode: str,
        min_votes: int | None,
    ) -> "DriftCheckConfig":
        methods = [m.strip() for m in ensemble_methods.split(",") if m.strip()]
        if not methods:
            methods = ["psi", "ks", "cvm", "js"]
        ensemble = EnsembleConfig(methods=methods, vote_mode=vote_mode, min_votes=min_votes)
        return DriftCheckConfig(
            method=method,
            threshold=threshold,
            correction=correction,
            ensemble=ensemble,
        )

    @staticmethod
    def from_file(path: str | Path) -> "DriftCheckConfig":
        """Load drift configuration from JSON, TOML, or YAML file."""
        file_path = Path(path)
        suffix = file_path.suffix.lower()
        text = file_path.read_text(encoding="utf-8")

        if suffix == ".json":
            raw = json.loads(text)
        elif suffix == ".toml":
            try:
                import tomllib as _toml_loader  # type: ignore[import-not-found]
            except ModuleNotFoundError:  # pragma: no cover
                import tomli as _toml_loader  # type: ignore[import-not-found]
            raw = _toml_loader.loads(text)
        elif suffix in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:
                raise ValueError(
                    "YAML config requires pyyaml. Install with: pip install pyyaml"
                ) from exc
            raw = yaml.safe_load(text)
        else:
            raise ValueError("config file must be .json, .toml, .yaml, or .yml")

        if not isinstance(raw, dict):
            raise ValueError("config file root must be a mapping/object")

        return DriftCheckConfig._from_mapping(raw)

    @staticmethod
    def from_env(prefix: str = "DRIFT_CONTROL_") -> "DriftCheckConfig":
        """Load drift configuration from environment variables."""
        method = os.getenv(f"{prefix}METHOD", "psi")
        threshold_raw = os.getenv(f"{prefix}THRESHOLD")
        threshold: float | None = None
        if threshold_raw is not None and threshold_raw != "":
            threshold = float(threshold_raw)
        ensemble_methods = os.getenv(f"{prefix}ENSEMBLE_METHODS", "psi,ks,cvm,js")
        correction = os.getenv(f"{prefix}CORRECTION", "none")
        vote_mode = os.getenv(f"{prefix}VOTE_MODE", "majority")
        min_votes_raw = os.getenv(f"{prefix}MIN_VOTES")
        min_votes: int | None = None
        if min_votes_raw is not None and min_votes_raw != "":
            min_votes = int(min_votes_raw)
        return DriftCheckConfig.from_cli(
            method=method,
            threshold=threshold,
            correction=correction,
            ensemble_methods=ensemble_methods,
            vote_mode=vote_mode,
            min_votes=min_votes,
        )

    @staticmethod
    def _from_mapping(raw: dict[str, Any]) -> "DriftCheckConfig":
        method = str(raw.get("method", "psi"))
        threshold_val = raw.get("threshold")
        threshold = float(threshold_val) if threshold_val is not None else None
        correction = str(raw.get("correction", "none"))
        ensemble_raw = raw.get("ensemble", {})
        if ensemble_raw is None:
            ensemble_raw = {}
        if not isinstance(ensemble_raw, dict):
            raise ValueError("ensemble config must be a mapping/object")

        methods_raw = ensemble_raw.get("methods", ["psi", "ks", "cvm", "js"])
        if not isinstance(methods_raw, list):
            raise ValueError("ensemble.methods must be a list")
        methods = [str(m) for m in methods_raw]

        vote_mode = str(ensemble_raw.get("vote_mode", "majority"))
        min_votes_raw = ensemble_raw.get("min_votes")
        min_votes = int(min_votes_raw) if min_votes_raw is not None else None
        ensemble = EnsembleConfig(methods=methods, vote_mode=vote_mode, min_votes=min_votes)
        return DriftCheckConfig(
            method=method,
            threshold=threshold,
            correction=correction,
            ensemble=ensemble,
        )
