from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from .cvm_drift_detector import CVMDriftDetector
from .js_drift_detector import JensenShannonDriftDetector
from .ks_drift_detector import KSDriftDetector
from .psi_drift_detector import PSIDriftDetector
from .wasserstein_drift_detector import WassersteinDriftDetector


@dataclass(frozen=True)
class EnsembleColumnResult:
    drift_detected: bool
    votes: int
    required_votes: int
    method_results: dict[str, dict[str, float | bool]]


class EnsembleDriftDetector:
    """Run multiple univariate detectors per numeric column and vote on drift."""

    def __init__(
        self,
        methods: list[str] | None = None,
        vote_mode: str = "majority",
        min_votes: int | None = None,
    ) -> None:
        available = {"psi", "ks", "cvm", "js", "wasserstein"}
        selected = methods or ["psi", "ks", "cvm", "js"]
        unknown = [m for m in selected if m not in available]
        if unknown:
            raise ValueError(f"unknown methods: {unknown}")
        if vote_mode not in {"majority", "any", "all"}:
            raise ValueError("vote_mode must be one of: 'majority', 'any', 'all'")

        self.methods = selected
        self.vote_mode = vote_mode
        self.min_votes = min_votes

        self._builders: dict[str, Callable[[], object]] = {
            "psi": lambda: PSIDriftDetector(),
            "ks": lambda: KSDriftDetector(),
            "cvm": lambda: CVMDriftDetector(),
            "js": lambda: JensenShannonDriftDetector(),
            "wasserstein": lambda: WassersteinDriftDetector(n_permutations=100),
        }

    def _required_votes(self) -> int:
        if self.min_votes is not None:
            if self.min_votes < 1 or self.min_votes > len(self.methods):
                raise ValueError("min_votes must be between 1 and the number of methods")
            return self.min_votes
        if self.vote_mode == "any":
            return 1
        if self.vote_mode == "all":
            return len(self.methods)
        return (len(self.methods) // 2) + 1

    def detect_drift(self, df_prior: pd.DataFrame, df_post: pd.DataFrame) -> dict[str, EnsembleColumnResult]:
        if not isinstance(df_prior, pd.DataFrame) or not isinstance(df_post, pd.DataFrame):
            raise TypeError("df_prior and df_post must be pandas DataFrames")
        if set(df_prior.columns) != set(df_post.columns):
            raise ValueError("df_prior and df_post must have the same columns")

        required_votes = self._required_votes()
        results: dict[str, EnsembleColumnResult] = {}

        for col in sorted(df_prior.columns):
            prior = pd.to_numeric(df_prior[col], errors="raise")
            post = pd.to_numeric(df_post[col], errors="raise")
            if prior.isna().any() or post.isna().any():
                raise ValueError(f"column {col!r} contains null values")

            method_results: dict[str, dict[str, float | bool]] = {}
            votes = 0
            for method in self.methods:
                detector = self._builders[method]()
                if method == "wasserstein":
                    details = detector.detect_drift(prior.values, post.values, return_details=True)
                    drift = bool(details.drift_detected)
                    score = float(details.distance)
                    method_results[method] = {
                        "drift": drift,
                        "score": score,
                        "p_value": float(details.p_value),
                    }
                else:
                    drift, score = detector.detect_drift(prior.values, post.values)
                    method_results[method] = {"drift": bool(drift), "score": float(score)}
                votes += int(drift)

            results[col] = EnsembleColumnResult(
                drift_detected=votes >= required_votes,
                votes=votes,
                required_votes=required_votes,
                method_results=method_results,
            )

        return results
