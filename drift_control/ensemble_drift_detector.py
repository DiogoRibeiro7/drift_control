from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import Any

import pandas as pd

from .multiple_testing import adjust_pvalues
from .unified_drift_detector import UnifiedDriftDetector
from .validation import (
    coerce_categorical_series,
    coerce_numeric_series,
    validate_matching_columns,
)

_AVAILABLE_METHODS = {"psi", "ks", "cvm", "js", "wasserstein", "chi2cat", "tvdcat"}


@dataclass(frozen=True)
class EnsembleColumnResult:
    drift_detected: bool
    votes: int
    required_votes: int
    method_results: dict[str, dict[str, float | bool]]

    def _repr_html_(self) -> str:
        detail_rows: list[str] = []
        for method, payload in sorted(self.method_results.items()):
            score = payload.get("score")
            p_value = payload.get("p_value")
            drift = bool(payload.get("drift"))
            score_str = f"{float(score):.6f}" if isinstance(score, (int, float)) else "-"
            pval_str = f"{float(p_value):.6f}" if isinstance(p_value, (int, float)) else "-"
            detail_rows.append(
                "<tr>"
                f"<td>{escape(method)}</td>"
                f"<td>{score_str}</td>"
                f"<td>{pval_str}</td>"
                f"<td>{'YES' if drift else 'NO'}</td>"
                "</tr>"
            )
        return (
            "<div>"
            "<table>"
            "<thead><tr><th>Drift</th><th>Votes</th><th>Required Votes</th></tr></thead>"
            "<tbody>"
            f"<tr><td>{'YES' if self.drift_detected else 'NO'}</td>"
            f"<td>{self.votes}</td><td>{self.required_votes}</td></tr>"
            "</tbody></table>"
            "<table>"
            "<thead><tr><th>Method</th><th>Score</th><th>P-Value</th><th>Drift</th></tr></thead>"
            f"<tbody>{''.join(detail_rows)}</tbody>"
            "</table>"
            "</div>"
        )


@dataclass(frozen=True)
class _ColumnScoreState:
    method_results: dict[str, dict[str, float | bool]]
    votes: int
    confidences: list[float]
    pvalues: list[tuple[str, float]]


class EnsembleDriftDetector:
    """Run multiple univariate detectors per numeric column and vote on drift.

    Scoring is delegated to the structured stack via ``UnifiedDriftDetector``.
    """

    def __init__(
        self,
        methods: list[str] | None = None,
        vote_mode: str = "majority",
        min_votes: int | None = None,
        correction: str = "none",
        stack_threshold: float = 0.5,
    ) -> None:
        selected = methods or ["psi", "ks", "cvm", "js"]
        unknown = [method for method in selected if method not in _AVAILABLE_METHODS]
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

    @staticmethod
    def _make(method: str) -> UnifiedDriftDetector:
        if method == "wasserstein":
            return UnifiedDriftDetector(method="wasserstein", n_permutations=100)
        return UnifiedDriftDetector(method=method)

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
    def _confidence(row: dict[str, float | bool], detector: UnifiedDriftDetector) -> float:
        p_value = row.get("p_value")
        if isinstance(p_value, (float, int)):
            return max(0.0, min(1.0, 1.0 - float(p_value)))
        score = float(row["score"])
        threshold = float(detector.threshold)
        if threshold <= 0:
            return 0.0
        return max(0.0, min(1.0, score / threshold))

    @staticmethod
    def _coerce_series(method: str, col: str, prior: Any, post: Any) -> tuple[Any, Any]:
        if method in {"chi2cat", "tvdcat"}:
            return coerce_categorical_series(prior, post, column_name=col, method_name="ensemble")
        return coerce_numeric_series(prior, post, column_name=col, method_name="ensemble")

    def _score_method(
        self,
        *,
        method: str,
        col: str,
        prior: Any,
        post: Any,
    ) -> tuple[dict[str, float | bool], float | None, float]:
        detector = self._make(method)
        prior_series, post_series = self._coerce_series(method, col, prior, post)
        result = detector.detect_drift(prior_series.values, post_series.values)
        row: dict[str, float | bool] = {
            "drift": bool(result.drift),
            "score": float(result.score),
        }
        p_value: float | None = None
        if result.p_value is not None:
            p_value = float(result.p_value)
            row["p_value"] = p_value
        return row, p_value, self._confidence(row, detector)

    def _score_column(self, col: str, prior: Any, post: Any) -> _ColumnScoreState:
        method_results: dict[str, dict[str, float | bool]] = {}
        pvalues: list[tuple[str, float]] = []
        confidences: list[float] = []
        votes = 0
        for method in self.methods:
            row, p_value, confidence = self._score_method(
                method=method,
                col=col,
                prior=prior,
                post=post,
            )
            method_results[method] = row
            votes += int(bool(row["drift"]))
            confidences.append(confidence)
            if p_value is not None:
                pvalues.append((method, p_value))
        return _ColumnScoreState(
            method_results=method_results,
            votes=votes,
            confidences=confidences,
            pvalues=pvalues,
        )

    def _decide(
        self,
        method_results: dict[str, dict[str, float | bool]],
        votes: int,
        confidences: list[float],
        required_votes: int,
    ) -> bool:
        stack_score = float(sum(confidences) / max(len(confidences), 1))
        drift_detected = (
            stack_score >= self.stack_threshold
            if self.vote_mode == "stacking"
            else votes >= required_votes
        )
        method_results["_stacking"] = {
            "score": stack_score,
            "threshold": self.stack_threshold,
            "drift": drift_detected,
        }
        return drift_detected

    def _build_column_result(
        self,
        *,
        state: _ColumnScoreState,
        required_votes: int,
    ) -> EnsembleColumnResult:
        drift_detected = self._decide(
            state.method_results,
            state.votes,
            state.confidences,
            required_votes,
        )
        return EnsembleColumnResult(
            drift_detected=drift_detected,
            votes=state.votes,
            required_votes=required_votes,
            method_results=state.method_results,
        )

    @staticmethod
    def _validate_frames(df_prior: pd.DataFrame, df_post: pd.DataFrame) -> None:
        if not isinstance(df_prior, pd.DataFrame) or not isinstance(df_post, pd.DataFrame):
            raise TypeError("df_prior and df_post must be pandas DataFrames")
        try:
            validate_matching_columns(df_prior, df_post)
        except ValueError as exc:
            raise ValueError("df_prior and df_post must have the same columns") from exc

    def detect_drift(
        self, df_prior: pd.DataFrame, df_post: pd.DataFrame
    ) -> dict[str, EnsembleColumnResult]:
        self._validate_frames(df_prior, df_post)
        required_votes = self._required_votes()
        results: dict[str, EnsembleColumnResult] = {}
        pvalues_by_method: dict[str, list[tuple[str, float]]] = {}
        for col in sorted(df_prior.columns):
            state = self._score_column(col, df_prior[col], df_post[col])
            for method, p_value in state.pvalues:
                pvalues_by_method.setdefault(method, []).append((col, p_value))
            results[col] = self._build_column_result(state=state, required_votes=required_votes)
        if self.correction != "none":
            self._apply_correction(results, pvalues_by_method, required_votes)
        return results

    def _state_from_method_results(
        self,
        method_results: dict[str, dict[str, float | bool]],
    ) -> _ColumnScoreState:
        votes = sum(
            int(bool(payload["drift"]))
            for method, payload in method_results.items()
            if method != "_stacking"
        )
        confidences = [
            self._confidence(method_results[method], self._make(method)) for method in self.methods
        ]
        pvalues = [
            (method, float(payload["p_value"]))
            for method, payload in method_results.items()
            if method != "_stacking" and isinstance(payload.get("p_value"), (float, int))
        ]
        return _ColumnScoreState(
            method_results=method_results,
            votes=votes,
            confidences=confidences,
            pvalues=pvalues,
        )

    def _apply_correction(
        self,
        results: dict[str, EnsembleColumnResult],
        pvalues_by_method: dict[str, list[tuple[str, float]]],
        required_votes: int,
    ) -> None:
        touched_columns: set[str] = set()
        for method, items in pvalues_by_method.items():
            columns = [column for column, _ in items]
            adjusted = adjust_pvalues([p for _, p in items], method=self.correction)
            for column, adjusted_p in zip(columns, adjusted, strict=True):
                touched_columns.add(column)
                method_result = results[column].method_results[method]
                method_result["p_value"] = float(adjusted_p)
                method_result["drift"] = bool(adjusted_p < 0.05)
        for column in sorted(touched_columns):
            state = self._state_from_method_results(results[column].method_results)
            results[column] = self._build_column_result(
                state=state,
                required_votes=required_votes,
            )


__all__ = ["EnsembleColumnResult", "EnsembleDriftDetector"]
