from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol, cast

import pandas as pd

from .categorical_chi2_drift_detector import ChiSquareDriftDetector
from .categorical_tvd_drift_detector import TotalVariationDriftDetector
from .cvm_drift_detector import CVMDriftDetector
from .js_drift_detector import JensenShannonDriftDetector
from .ks_drift_detector import KSDriftDetector
from .psi_drift_detector import PSIDriftDetector
from .wasserstein_drift_detector import WassersteinDriftDetector
from .multiple_testing import adjust_pvalues
from .validation import (
    coerce_categorical_series,
    coerce_numeric_series,
    validate_matching_columns,
)


@dataclass(frozen=True)
class EnsembleColumnResult:
    drift_detected: bool
    votes: int
    required_votes: int
    method_results: dict[str, dict[str, float | bool]]


class _SimpleDetector(Protocol):
    def detect_drift(self, reference_data: Any, current_data: Any) -> tuple[bool, float]:
        ...


class _DetailedDetector(Protocol):
    def detect_drift(
        self, reference_data: Any, current_data: Any, return_details: bool = False
    ) -> Any:
        ...


class EnsembleDriftDetector:
    """Run multiple univariate detectors per numeric column and vote on drift."""

    def __init__(
        self,
        methods: list[str] | None = None,
        vote_mode: str = "majority",
        min_votes: int | None = None,
        correction: str = "none",
        stack_threshold: float = 0.5,
    ) -> None:
        available = {"psi", "ks", "cvm", "js", "wasserstein", "chi2cat", "tvdcat"}
        selected = methods or ["psi", "ks", "cvm", "js"]
        unknown = [m for m in selected if m not in available]
        if unknown:
            raise ValueError(f"unknown methods: {unknown}")
        if vote_mode not in {"majority", "any", "all", "stacking"}:
            raise ValueError("vote_mode must be one of: 'majority', 'any', 'all', 'stacking'")
        if correction not in {"none", "bonferroni", "bh"}:
            raise ValueError("correction must be one of: none, bonferroni, bh")
        if not (0 <= stack_threshold <= 1):
            raise ValueError("stack_threshold must be between 0 and 1")

        self.methods = selected
        self.vote_mode = vote_mode
        self.min_votes = min_votes
        self.correction = correction
        self.stack_threshold = float(stack_threshold)

        self._builders: dict[str, Callable[[], object]] = {
            "psi": lambda: PSIDriftDetector(),
            "ks": lambda: KSDriftDetector(),
            "cvm": lambda: CVMDriftDetector(),
            "js": lambda: JensenShannonDriftDetector(),
            "wasserstein": lambda: WassersteinDriftDetector(n_permutations=100),
            "chi2cat": lambda: ChiSquareDriftDetector(),
            "tvdcat": lambda: TotalVariationDriftDetector(),
        }

    def _required_votes(self) -> int:
        if self.vote_mode == "stacking":
            return 0
        if self.min_votes is not None:
            if self.min_votes < 1 or self.min_votes > len(self.methods):
                raise ValueError("min_votes must be between 1 and the number of methods")
            return self.min_votes
        if self.vote_mode == "any":
            return 1
        if self.vote_mode == "all":
            return len(self.methods)
        return (len(self.methods) // 2) + 1

    @staticmethod
    def _confidence(method: str, row: dict[str, float | bool], detector: object) -> float:
        p = row.get("p_value")
        if isinstance(p, (float, int)):
            return max(0.0, min(1.0, 1.0 - float(p)))
        score = float(row["score"])
        threshold = float(getattr(detector, "threshold", 1.0))
        if threshold <= 0:
            return 0.0
        return max(0.0, min(1.0, score / threshold))

    def detect_drift(self, df_prior: pd.DataFrame, df_post: pd.DataFrame) -> dict[str, EnsembleColumnResult]:
        if not isinstance(df_prior, pd.DataFrame) or not isinstance(df_post, pd.DataFrame):
            raise TypeError("df_prior and df_post must be pandas DataFrames")
        try:
            validate_matching_columns(df_prior, df_post)
        except ValueError as exc:
            raise ValueError("df_prior and df_post must have the same columns") from exc

        required_votes = self._required_votes()
        results: dict[str, EnsembleColumnResult] = {}
        raw_pvalues_by_method: dict[str, list[tuple[str, float]]] = {}

        for col in sorted(df_prior.columns):
            method_results: dict[str, dict[str, float | bool]] = {}
            votes = 0
            confidences: list[float] = []
            for method in self.methods:
                detector = self._builders[method]()
                if method in {"chi2cat", "tvdcat"}:
                    prior, post = coerce_categorical_series(
                        df_prior[col], df_post[col], column_name=col, method_name="ensemble"
                    )
                else:
                    prior, post = coerce_numeric_series(
                        df_prior[col], df_post[col], column_name=col, method_name="ensemble"
                    )
                if method == "wasserstein":
                    details = cast(_DetailedDetector, detector).detect_drift(
                        prior.values, post.values, return_details=True
                    )
                    drift = bool(details.drift_detected)
                    score = float(details.distance)
                    method_results[method] = {
                        "drift": drift,
                        "score": score,
                        "p_value": float(details.p_value),
                    }
                    raw_pvalues_by_method.setdefault(method, []).append((col, float(details.p_value)))
                else:
                    drift, score = cast(_SimpleDetector, detector).detect_drift(
                        prior.values, post.values
                    )
                    row: dict[str, float | bool] = {"drift": bool(drift), "score": float(score)}
                    if method in {"ks", "cvm", "chi2cat"}:
                        row["p_value"] = float(score)
                        raw_pvalues_by_method.setdefault(method, []).append((col, float(score)))
                    method_results[method] = row
                votes += int(drift)
                confidences.append(self._confidence(method, method_results[method], detector))

            stack_score = float(sum(confidences) / max(len(confidences), 1))
            if self.vote_mode == "stacking":
                drift_detected = bool(stack_score >= self.stack_threshold)
            else:
                drift_detected = bool(votes >= required_votes)
            method_results["_stacking"] = {
                "score": stack_score,
                "threshold": self.stack_threshold,
                "drift": drift_detected,
            }

            results[col] = EnsembleColumnResult(
                drift_detected=drift_detected,
                votes=votes,
                required_votes=required_votes,
                method_results=method_results,
            )

        if self.correction != "none":
            for method, items in raw_pvalues_by_method.items():
                cols = [c for c, _ in items]
                pvals = [p for _, p in items]
                adj = adjust_pvalues(pvals, method=self.correction)
                for col, adj_p in zip(cols, adj):
                    mr = results[col].method_results[method]
                    mr["p_value"] = float(adj_p)
                    mr["drift"] = bool(adj_p < 0.05)
                for col in cols:
                    votes = sum(
                        int(bool(v["drift"]))
                        for k, v in results[col].method_results.items()
                        if k != "_stacking"
                    )
                    confidences = [
                        self._confidence(m, results[col].method_results[m], self._builders[m]())
                        for m in self.methods
                    ]
                    stack_score = float(sum(confidences) / max(len(confidences), 1))
                    if self.vote_mode == "stacking":
                        drift_detected = bool(stack_score >= self.stack_threshold)
                    else:
                        drift_detected = bool(votes >= required_votes)
                    results[col].method_results["_stacking"] = {
                        "score": stack_score,
                        "threshold": self.stack_threshold,
                        "drift": drift_detected,
                    }
                    results[col] = EnsembleColumnResult(
                        drift_detected=drift_detected,
                        votes=votes,
                        required_votes=required_votes,
                        method_results=results[col].method_results,
                    )

        return results
