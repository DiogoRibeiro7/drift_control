from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SUPPORTED_METHODS = {
    "psi",
    "ks",
    "mmd",
    "c2st",
    "energy",
    "cvm",
    "js",
    "wasserstein",
    "chi2cat",
    "tvdcat",
    "datetime",
    "ensemble",
}
SUPPORTED_ENSEMBLE_METHODS = {
    "psi",
    "ks",
    "cvm",
    "js",
    "wasserstein",
    "chi2cat",
    "tvdcat",
}
SUPPORTED_VOTE_MODES = {"majority", "any", "all", "stacking"}
SUPPORTED_CORRECTIONS = {"none", "bonferroni", "bh"}
DEFAULT_ENSEMBLE_METHODS = ["psi", "ks", "cvm", "js"]


def _default_ensemble_config() -> EnsembleConfig:
    return EnsembleConfig(methods=list(DEFAULT_ENSEMBLE_METHODS))


def _parse_ensemble_methods_csv(raw: str) -> list[str]:
    methods = [method.strip() for method in raw.split(",") if method.strip()]
    return methods or list(DEFAULT_ENSEMBLE_METHODS)


def _optional_float(raw: str | None) -> float | None:
    if raw is None or raw == "":
        return None
    return float(raw)


def _optional_int(raw: str | None) -> int | None:
    if raw is None or raw == "":
        return None
    return int(raw)


def _build_ensemble_config(
    *,
    methods: list[str],
    vote_mode: str,
    min_votes: int | None,
    stack_threshold: float,
) -> EnsembleConfig:
    return EnsembleConfig(
        methods=methods,
        vote_mode=vote_mode,
        min_votes=min_votes,
        stack_threshold=stack_threshold,
    )


def _load_config_mapping(path: str | Path) -> dict[str, Any]:
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
    return raw


def _ensemble_from_mapping(raw: Any) -> EnsembleConfig:
    ensemble_raw = {} if raw is None else raw
    if not isinstance(ensemble_raw, dict):
        raise ValueError("ensemble config must be a mapping/object")
    methods_raw = ensemble_raw.get("methods", list(DEFAULT_ENSEMBLE_METHODS))
    if not isinstance(methods_raw, list):
        raise ValueError("ensemble.methods must be a list")
    methods = [str(method) for method in methods_raw]
    vote_mode = str(ensemble_raw.get("vote_mode", "majority"))
    min_votes = (
        int(ensemble_raw["min_votes"]) if ensemble_raw.get("min_votes") is not None else None
    )
    stack_threshold = float(ensemble_raw.get("stack_threshold", 0.5))
    return _build_ensemble_config(
        methods=methods,
        vote_mode=vote_mode,
        min_votes=min_votes,
        stack_threshold=stack_threshold,
    )


@dataclass(frozen=True)
class EnsembleConfig:
    methods: list[str]
    vote_mode: str = "majority"
    min_votes: int | None = None
    stack_threshold: float = 0.5

    def __post_init__(self) -> None:
        unknown = [m for m in self.methods if m not in SUPPORTED_ENSEMBLE_METHODS]
        if unknown:
            raise ValueError(f"unknown methods: {unknown}")
        if self.vote_mode not in SUPPORTED_VOTE_MODES:
            raise ValueError("vote_mode must be one of: 'majority', 'any', 'all', 'stacking'")
        if self.min_votes is not None and (
            self.min_votes < 1 or self.min_votes > len(self.methods)
        ):
            raise ValueError("min_votes must be between 1 and the number of methods")
        if not (0 <= self.stack_threshold <= 1):
            raise ValueError("stack_threshold must be between 0 and 1")


@dataclass(frozen=True)
class DriftCheckConfig:
    method: str = "psi"
    threshold: float | None = None
    correction: str = "none"
    ensemble: EnsembleConfig = field(default_factory=_default_ensemble_config)

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
        stack_threshold: float = 0.5,
    ) -> DriftCheckConfig:
        ensemble = _build_ensemble_config(
            methods=_parse_ensemble_methods_csv(ensemble_methods),
            vote_mode=vote_mode,
            min_votes=min_votes,
            stack_threshold=stack_threshold,
        )
        return DriftCheckConfig(
            method=method,
            threshold=threshold,
            correction=correction,
            ensemble=ensemble,
        )

    @staticmethod
    def from_file(path: str | Path) -> DriftCheckConfig:
        """Load drift configuration from JSON, TOML, or YAML file."""
        return DriftCheckConfig._from_mapping(_load_config_mapping(path))

    @staticmethod
    def from_env(prefix: str = "DRIFT_CONTROL_") -> DriftCheckConfig:
        """Load drift configuration from environment variables."""
        return DriftCheckConfig.from_cli(
            method=os.getenv(f"{prefix}METHOD", "psi"),
            threshold=_optional_float(os.getenv(f"{prefix}THRESHOLD")),
            correction=os.getenv(f"{prefix}CORRECTION", "none"),
            ensemble_methods=os.getenv(
                f"{prefix}ENSEMBLE_METHODS", ",".join(DEFAULT_ENSEMBLE_METHODS)
            ),
            vote_mode=os.getenv(f"{prefix}VOTE_MODE", "majority"),
            min_votes=_optional_int(os.getenv(f"{prefix}MIN_VOTES")),
            stack_threshold=0.5,
        )

    @staticmethod
    def _from_mapping(raw: dict[str, Any]) -> DriftCheckConfig:
        method = str(raw.get("method", "psi"))
        threshold = float(raw["threshold"]) if raw.get("threshold") is not None else None
        correction = str(raw.get("correction", "none"))
        ensemble = _ensemble_from_mapping(raw.get("ensemble", {}))
        return DriftCheckConfig(
            method=method,
            threshold=threshold,
            correction=correction,
            ensemble=ensemble,
        )
