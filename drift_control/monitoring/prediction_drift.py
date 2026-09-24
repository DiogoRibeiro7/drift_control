"""Prediction-drift monitoring: model output distribution shift (no labels needed)."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import chi2_contingency

from ..core.exceptions import NotFittedError, ValidationError
from ..core.result import DriftResult
from ..core.types import ArrayLike
from ..distances import population_stability_index


def _confidence_entropy(proba: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    p: np.ndarray = np.asarray(proba, dtype=float)
    if p.ndim == 1:
        p = np.column_stack([1.0 - p, p])
    if p.ndim != 2:
        raise ValidationError("proba must be 1D (binary) or 2D (per-class)")
    p = np.clip(p, 1e-12, 1.0)
    confidence: np.ndarray = p.max(axis=1)
    entropy: np.ndarray = -np.sum(p * np.log(p), axis=1)
    return confidence, entropy


def _class_distribution_signal(
    reference_predictions: np.ndarray,
    current_predictions: np.ndarray,
    *,
    alpha: float,
) -> tuple[float, bool, int]:
    ref_s = reference_predictions.astype(str)
    cur_s = current_predictions.astype(str)
    categories = np.unique(np.concatenate([ref_s, cur_s]))
    ref_counts = np.array([(ref_s == category).sum() for category in categories], dtype=float)
    cur_counts = np.array([(cur_s == category).sum() for category in categories], dtype=float)
    table = np.vstack([ref_counts, cur_counts])
    table = table[:, table.sum(axis=0) > 0]
    if table.shape[1] < 2:
        return 1.0, False, int(categories.size)
    class_p = float(chi2_contingency(table, correction=False)[1])
    return class_p, class_p < alpha, int(categories.size)


def _base_metadata(
    *,
    class_p_value: float,
    n_classes: int,
    class_drift: bool,
) -> dict[str, Any]:
    return {
        "class_distribution_p_value": class_p_value,
        "n_classes": n_classes,
        "class_drift": class_drift,
        "confidence_drift": False,
    }


def _maybe_add_confidence_signal(
    metadata: dict[str, Any],
    *,
    reference_confidence: np.ndarray | None,
    current_proba: ArrayLike | None,
    psi_threshold: float,
    bins: int,
) -> bool:
    if current_proba is None or reference_confidence is None:
        return False
    current_confidence, current_entropy = _confidence_entropy(current_proba)
    confidence_psi = population_stability_index(reference_confidence, current_confidence, bins=bins)
    confidence_drift = confidence_psi > psi_threshold
    metadata["confidence_psi"] = float(confidence_psi)
    metadata["mean_entropy"] = float(current_entropy.mean())
    metadata["confidence_drift"] = confidence_drift
    return confidence_drift


class PredictionDriftMonitor:
    """Detects shift in a model's predictions vs a reference batch.

    Combines predicted-class-distribution drift (chi-square) with, when
    probabilities are supplied, confidence drift (PSI on the top-class
    probability) and a predictive-entropy summary. ``detect`` flags drift if
    either signal fires; the ``score``/``threshold`` fields track the class
    distribution test, with the confidence signal in metadata.
    """

    def __init__(self, *, alpha: float = 0.05, psi_threshold: float = 0.2, bins: int = 10) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValidationError("alpha must be in (0, 1)")
        if psi_threshold <= 0.0:
            raise ValidationError("psi_threshold must be > 0")
        self.alpha = float(alpha)
        self.psi_threshold = float(psi_threshold)
        self.bins = int(bins)
        self._ref_pred: np.ndarray | None = None
        self._ref_conf: np.ndarray | None = None

    def fit(
        self, predictions: ArrayLike, *, proba: ArrayLike | None = None
    ) -> PredictionDriftMonitor:
        ref: np.ndarray = np.asarray(predictions).ravel()
        if ref.size == 0:
            raise ValidationError("predictions must be non-empty")
        self._ref_pred = ref
        self._ref_conf = _confidence_entropy(proba)[0] if proba is not None else None
        return self

    def detect(self, predictions: ArrayLike, *, proba: ArrayLike | None = None) -> DriftResult:
        if self._ref_pred is None:
            raise NotFittedError("call fit() before detect()")
        cur: np.ndarray = np.asarray(predictions).ravel()
        class_p, class_drift, n_classes = _class_distribution_signal(
            self._ref_pred, cur, alpha=self.alpha
        )
        metadata = _base_metadata(
            class_p_value=class_p,
            n_classes=n_classes,
            class_drift=class_drift,
        )
        confidence_drift = _maybe_add_confidence_signal(
            metadata,
            reference_confidence=self._ref_conf,
            current_proba=proba,
            psi_threshold=self.psi_threshold,
            bins=self.bins,
        )

        return DriftResult.new(
            drift_detected=bool(class_drift or confidence_drift),
            score=class_p,
            threshold=self.alpha,
            p_value=class_p,
            method="prediction_drift",
            comparator="<",
            metadata=metadata,
        )


__all__ = ["PredictionDriftMonitor"]
