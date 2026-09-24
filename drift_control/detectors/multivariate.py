"""Multivariate two-sample drift detection on the core contract.

``MultivariateDriftDetector`` captures joint / non-linear distribution shift that
feature-wise tests miss, by composing the permutation-calibrated ``distances``
primitives. Decision is ``statistic > calibrated_threshold`` (equivalently the
permutation p-value below ``alpha``); the score/threshold/comparator triplet
reconciles with ``drift``.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from ..core.base import BaseDetector
from ..core.exceptions import NotFittedError, ValidationError
from ..core.result import DriftResult
from ..core.types import ArrayLike
from ..distances import energy_permutation_test, mmd_permutation_test
from ..preprocessing import coerce_observations


def _c2st_permutation_test(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_permutations: int,
    test_size: float,
    alpha: float,
    random_state: int,
) -> tuple[float, float, float]:
    """Classifier two-sample test: ROC-AUC of reference-vs-current, calibrated.

    Returns ``(roc_auc, p_value, calibrated_threshold)``. Drift means the
    classifier separates the samples better than the permutation null.
    """
    data = np.vstack([x, y])
    labels = np.concatenate([np.zeros(x.shape[0]), np.ones(y.shape[0])])

    def fit_auc(features: np.ndarray, target: np.ndarray) -> float:
        f_train, f_test, t_train, t_test = train_test_split(
            features,
            target,
            test_size=test_size,
            random_state=random_state,
            stratify=target,
        )
        clf = LogisticRegression(max_iter=1000)
        clf.fit(f_train, t_train)
        return float(roc_auc_score(t_test, clf.predict_proba(f_test)[:, 1]))

    observed = fit_auc(data, labels)
    rng = np.random.default_rng(random_state)
    null = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        null[i] = fit_auc(data, rng.permutation(labels))
    threshold = float(np.quantile(null, 1.0 - alpha))
    p_value = float((1.0 + np.sum(null >= observed)) / (1.0 + n_permutations))
    return observed, p_value, threshold


class MultivariateDriftDetector(BaseDetector):
    """Permutation-calibrated multivariate drift via MMD, energy, or classifier.

    :param method: ``"mmd"`` (RBF-kernel MMD^2), ``"energy"``, or ``"c2st"``
        (classifier two-sample test / covariate shift, ROC-AUC of a logistic
        regression distinguishing reference from current).
    :param gamma: RBF bandwidth for ``method="mmd"`` (``None`` = median heuristic).
    :param test_size: held-out fraction for ``method="c2st"``.
    """

    def __init__(
        self,
        method: str = "mmd",
        *,
        alpha: float = 0.05,
        n_permutations: int = 200,
        gamma: float | None = None,
        test_size: float = 0.3,
        random_state: int = 42,
    ) -> None:
        if method not in {"mmd", "energy", "c2st"}:
            raise ValidationError("method must be 'mmd', 'energy', or 'c2st'")
        if not 0.0 < alpha < 1.0:
            raise ValidationError("alpha must be in (0, 1)")
        if n_permutations < 1:
            raise ValidationError("n_permutations must be >= 1")
        if not 0.0 < test_size < 1.0:
            raise ValidationError("test_size must be in (0, 1)")
        self.method = method
        self.alpha = float(alpha)
        self.n_permutations = int(n_permutations)
        self.gamma = gamma
        self.test_size = float(test_size)
        self.random_state = int(random_state)
        self._reference: np.ndarray | None = None

    def fit(self, reference_data: ArrayLike) -> MultivariateDriftDetector:
        self._reference = coerce_observations(reference_data)
        return self

    def detect(self, current_data: ArrayLike) -> DriftResult:
        if self._reference is None:
            raise NotFittedError("call fit() before detect()")
        current = coerce_observations(current_data)
        if current.shape[1] != self._reference.shape[1]:
            raise ValidationError(
                f"feature count mismatch: reference has {self._reference.shape[1]}, "
                f"current has {current.shape[1]}"
            )

        if self.method == "mmd":
            stat, p_value, threshold = mmd_permutation_test(
                self._reference,
                current,
                n_permutations=self.n_permutations,
                gamma=self.gamma,
                alpha=self.alpha,
                random_state=self.random_state,
            )
        elif self.method == "energy":
            stat, p_value, threshold = energy_permutation_test(
                self._reference,
                current,
                n_permutations=self.n_permutations,
                alpha=self.alpha,
                random_state=self.random_state,
            )
        else:
            if self._reference.shape[0] < 2 or current.shape[0] < 2:
                raise ValidationError("c2st requires at least 2 samples per side")
            stat, p_value, threshold = _c2st_permutation_test(
                self._reference,
                current,
                n_permutations=self.n_permutations,
                test_size=self.test_size,
                alpha=self.alpha,
                random_state=self.random_state,
            )

        return DriftResult.new(
            drift_detected=stat > threshold,
            score=stat,
            threshold=threshold,
            p_value=p_value,
            method=self.method,
            comparator=">",
            metadata={
                "alpha": self.alpha,
                "calibrated_threshold": threshold,
                "n_permutations": self.n_permutations,
            },
        )


__all__ = ["MultivariateDriftDetector"]
