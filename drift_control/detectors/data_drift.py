"""Batch univariate data-drift detection (ROADMAP.md Phase 3).

``UnivariateDriftDetector`` is the first user-facing detector built on the
Phase 0 :class:`BaseDetector` contract and composing the Phase 2 distance
primitives. It scores each feature independently, applies multiple-testing
correction for p-value methods, and reports both per-feature results and an
aggregate decision.

Methods:

- ``ks``  -- Kolmogorov-Smirnov two-sample test (numeric, p-value)
- ``chi2`` -- chi-square homogeneity test (categorical, p-value)
- ``psi`` -- Population Stability Index (numeric, threshold; default 0.2)
- ``js``  -- Jensen-Shannon divergence (numeric, threshold; default 0.1)
- ``wasserstein`` -- 1D Wasserstein distance (numeric, threshold required)

The ``score``/``threshold``/``comparator`` triplet always reconciles with
``drift`` at both the per-feature and aggregate level.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import chi2_contingency, ks_2samp

from ..core.base import BaseDetector
from ..core.exceptions import NotFittedError, ValidationError
from ..core.result import DriftResult
from ..core.types import ArrayLike
from ..distances import (
    js_divergence,
    population_stability_index,
    to_histograms,
    wasserstein_distance,
)
from ..multiple_testing import adjust_pvalues
from ..preprocessing import coerce_observations

_PVALUE_METHODS = {"ks", "chi2"}
_THRESHOLD_METHODS = {"psi", "js", "wasserstein"}
_CATEGORICAL_METHODS = {"chi2"}
_DEFAULT_THRESHOLD: dict[str, float] = {"psi": 0.2, "js": 0.1}


def _score_ks(ref: np.ndarray, cur: np.ndarray) -> tuple[float, float | None]:
    res = ks_2samp(ref, cur)
    return float(res.statistic), float(res.pvalue)


def _score_chi2(ref: np.ndarray, cur: np.ndarray) -> tuple[float, float | None]:
    ref_s = ref.astype(str)
    cur_s = cur.astype(str)
    categories = np.unique(np.concatenate([ref_s, cur_s]))
    ref_counts = np.array([(ref_s == c).sum() for c in categories], dtype=float)
    cur_counts = np.array([(cur_s == c).sum() for c in categories], dtype=float)
    table = np.vstack([ref_counts, cur_counts])
    table = table[:, table.sum(axis=0) > 0]
    if table.shape[1] < 2:
        return 0.0, 1.0
    stat, p_value, _, _ = chi2_contingency(table, correction=False)
    return float(stat), float(p_value)


def _score_psi(ref: np.ndarray, cur: np.ndarray, bins: int) -> tuple[float, float | None]:
    return population_stability_index(ref, cur, bins=bins), None


def _score_js(ref: np.ndarray, cur: np.ndarray, bins: int) -> tuple[float, float | None]:
    ref_p, cur_p = to_histograms(ref, cur, bins=bins)
    return js_divergence(ref_p, cur_p), None


def _score_wasserstein(ref: np.ndarray, cur: np.ndarray) -> tuple[float, float | None]:
    return wasserstein_distance(ref, cur), None


class UnivariateDriftDetector(BaseDetector):
    """Feature-wise batch drift detector over a single scoring method."""

    def __init__(
        self,
        method: str = "ks",
        *,
        alpha: float = 0.05,
        threshold: float | None = None,
        bins: int = 10,
        correction: str = "bh",
        feature_names: list[str] | None = None,
    ) -> None:
        method = str(method)
        if method not in _PVALUE_METHODS | _THRESHOLD_METHODS:
            allowed = ", ".join(sorted(_PVALUE_METHODS | _THRESHOLD_METHODS))
            raise ValidationError(f"unknown method {method!r}; choose from: {allowed}")
        if not (0.0 < alpha < 1.0):
            raise ValidationError("alpha must be in the open interval (0, 1)")
        if correction not in {"none", "bonferroni", "bh"}:
            raise ValidationError("correction must be one of: none, bonferroni, bh")

        self.method = method
        self.alpha = float(alpha)
        self.bins = int(bins)
        self.correction = correction
        self.feature_names = list(feature_names) if feature_names is not None else None

        self.threshold: float | None
        if method in _THRESHOLD_METHODS:
            resolved = threshold if threshold is not None else _DEFAULT_THRESHOLD.get(method)
            if resolved is None:
                raise ValidationError(
                    f"method '{method}' has no default threshold; pass threshold="
                )
            self.threshold = float(resolved)
        else:
            self.threshold = None

        self._reference: np.ndarray | None = None

    @property
    def _is_categorical(self) -> bool:
        return self.method in _CATEGORICAL_METHODS

    def _columns(self, data: ArrayLike, name: str) -> np.ndarray:
        if self._is_categorical:
            arr: np.ndarray = np.asarray(data)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            if arr.ndim != 2:
                raise ValidationError(f"{name} must be 1D or 2D")
            if arr.shape[0] == 0:
                raise ValidationError(f"{name} must be non-empty")
            return arr
        return coerce_observations(data)

    def _feature_name(self, j: int) -> str:
        if self.feature_names is not None and j < len(self.feature_names):
            return self.feature_names[j]
        return f"feature_{j}"

    def _score_column(self, ref: np.ndarray, cur: np.ndarray) -> tuple[float, float | None]:
        if self.method == "ks":
            return _score_ks(ref, cur)
        if self.method == "chi2":
            return _score_chi2(ref, cur)
        if self.method == "psi":
            return _score_psi(ref, cur, self.bins)
        if self.method == "js":
            return _score_js(ref, cur, self.bins)
        return _score_wasserstein(ref, cur)

    def fit(self, reference_data: ArrayLike) -> UnivariateDriftDetector:
        self._reference = self._columns(reference_data, "reference_data")
        return self

    def detect_features(self, current_data: ArrayLike) -> list[DriftResult]:
        """Per-feature results (with multiple-testing correction for p-values)."""
        if self._reference is None:
            raise NotFittedError("call fit() before detect()")
        cur = self._columns(current_data, "current_data")
        n_features = self._reference.shape[1]
        if cur.shape[1] != n_features:
            raise ValidationError(
                f"feature count mismatch: reference has {n_features}, "
                f"current has {cur.shape[1]}"
            )

        scores: list[float] = []
        raw_pvalues: list[float | None] = []
        for j in range(n_features):
            score, p_value = self._score_column(self._reference[:, j], cur[:, j])
            scores.append(score)
            raw_pvalues.append(p_value)

        results: list[DriftResult] = []
        if self.method in _PVALUE_METHODS:
            # p-value methods always set a p-value; narrow None for the type checker.
            raw = [0.0 if p is None else float(p) for p in raw_pvalues]
            adjusted = adjust_pvalues(raw, self.correction)
            for j in range(n_features):
                results.append(
                    DriftResult(
                        method=self.method,
                        drift=adjusted[j] < self.alpha,
                        score=float(adjusted[j]),
                        threshold=self.alpha,
                        comparator="<",
                        p_value=float(adjusted[j]),
                        metadata={
                            "feature": self._feature_name(j),
                            "statistic": scores[j],
                            "raw_p_value": raw[j],
                            "correction": self.correction,
                        },
                    )
                )
        else:
            assert self.threshold is not None
            for j in range(n_features):
                results.append(
                    DriftResult(
                        method=self.method,
                        drift=scores[j] > self.threshold,
                        score=scores[j],
                        threshold=self.threshold,
                        comparator=">",
                        p_value=None,
                        metadata={"feature": self._feature_name(j)},
                    )
                )
        return results

    def detect(self, current_data: ArrayLike) -> DriftResult:
        """Aggregate decision: drift if any feature drifts (post-correction)."""
        features = self.detect_features(current_data)
        n_drifting = sum(1 for f in features if f.drift)
        metadata: dict[str, Any] = {
            "n_features": len(features),
            "n_drifting": n_drifting,
            "correction": self.correction,
            "features": [f.metadata.get("feature") for f in features],
            "drifting_features": [
                f.metadata.get("feature") for f in features if f.drift
            ],
        }
        if self.method in _PVALUE_METHODS:
            agg_score = min((f.score for f in features), default=1.0)
            return DriftResult(
                method=self.method,
                drift=n_drifting > 0,
                score=float(agg_score),
                threshold=self.alpha,
                comparator="<",
                p_value=float(agg_score),
                metadata=metadata,
            )
        assert self.threshold is not None
        agg_score = max((f.score for f in features), default=0.0)
        return DriftResult(
            method=self.method,
            drift=n_drifting > 0,
            score=float(agg_score),
            threshold=self.threshold,
            comparator=">",
            p_value=None,
            metadata=metadata,
        )


__all__ = ["UnivariateDriftDetector"]
