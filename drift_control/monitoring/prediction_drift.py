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


class PredictionDriftMonitor:
    """Detects shift in a model's predictions vs a reference batch.

    Combines predicted-class-distribution drift (chi-square) with, when
    probabilities are supplied, confidence drift (PSI on the top-class
    probability) and a predictive-entropy summary. ``detect`` flags drift if
    either signal fires; the ``score``/``threshold`` fields track the class
    distribution test, with the confidence signal in metadata.
    """

    def __init__(
        self, *, alpha: float = 0.05, psi_threshold: float = 0.2, bins: int = 10
    ) -> None:
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

    def detect(
        self, predictions: ArrayLike, *, proba: ArrayLike | None = None
    ) -> DriftResult:
        if self._ref_pred is None:
            raise NotFittedError("call fit() before detect()")
        cur: np.ndarray = np.asarray(predictions).ravel()

        ref_s = self._ref_pred.astype(str)
        cur_s = cur.astype(str)
        categories = np.unique(np.concatenate([ref_s, cur_s]))
        ref_counts = np.array([(ref_s == c).sum() for c in categories], dtype=float)
        cur_counts = np.array([(cur_s == c).sum() for c in categories], dtype=float)
        table = np.vstack([ref_counts, cur_counts])
        table = table[:, table.sum(axis=0) > 0]
        if table.shape[1] < 2:
            class_p = 1.0
        else:
            class_p = float(chi2_contingency(table, correction=False)[1])
        class_drift = class_p < self.alpha

        metadata: dict[str, Any] = {
            "class_distribution_p_value": class_p,
            "n_classes": int(categories.size),
            "class_drift": class_drift,
            "confidence_drift": False,
        }

        confidence_drift = False
        if proba is not None and self._ref_conf is not None:
            cur_conf, cur_entropy = _confidence_entropy(proba)
            conf_psi = population_stability_index(
                self._ref_conf, cur_conf, bins=self.bins
            )
            confidence_drift = conf_psi > self.psi_threshold
            metadata["confidence_psi"] = float(conf_psi)
            metadata["mean_entropy"] = float(cur_entropy.mean())
            metadata["confidence_drift"] = confidence_drift

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
