from __future__ import annotations

from dataclasses import dataclass, field


SUPPORTED_METHODS = {"psi", "ks", "mmd", "c2st", "cvm", "js", "wasserstein", "ensemble"}
SUPPORTED_ENSEMBLE_METHODS = {"psi", "ks", "cvm", "js", "wasserstein"}
SUPPORTED_VOTE_MODES = {"majority", "any", "all"}


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
    ensemble: EnsembleConfig = field(
        default_factory=lambda: EnsembleConfig(methods=["psi", "ks", "cvm", "js"])
    )

    def __post_init__(self) -> None:
        if self.method not in SUPPORTED_METHODS:
            raise ValueError(f"unsupported method: {self.method}")

    @staticmethod
    def from_cli(
        method: str,
        threshold: float | None,
        ensemble_methods: str,
        vote_mode: str,
        min_votes: int | None,
    ) -> "DriftCheckConfig":
        methods = [m.strip() for m in ensemble_methods.split(",") if m.strip()]
        if not methods:
            methods = ["psi", "ks", "cvm", "js"]
        ensemble = EnsembleConfig(methods=methods, vote_mode=vote_mode, min_votes=min_votes)
        return DriftCheckConfig(method=method, threshold=threshold, ensemble=ensemble)
