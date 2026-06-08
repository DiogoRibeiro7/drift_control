"""Multivariate two-sample drift detection on the core contract.

``MultivariateDriftDetector`` captures joint / non-linear distribution shift that
feature-wise tests miss, by composing the permutation-calibrated ``distances``
primitives. Decision is ``statistic > calibrated_threshold`` (equivalently the
permutation p-value below ``alpha``); the score/threshold/comparator triplet
reconciles with ``drift``.
"""

from __future__ import annotations

import numpy as np

from ..core.base import BaseDetector
from ..core.exceptions import NotFittedError, ValidationError
from ..core.result import DriftResult
from ..core.types import ArrayLike
from ..distances import energy_permutation_test, mmd_permutation_test
from ..preprocessing import coerce_observations


class MultivariateDriftDetector(BaseDetector):
    """Permutation-calibrated multivariate drift via MMD or energy distance.

    :param method: ``"mmd"`` (RBF-kernel MMD^2) or ``"energy"``.
    :param gamma: RBF bandwidth for ``method="mmd"`` (``None`` = median heuristic).
    """

    def __init__(
        self,
        method: str = "mmd",
        *,
        alpha: float = 0.05,
        n_permutations: int = 200,
        gamma: float | None = None,
        random_state: int = 42,
    ) -> None:
        if method not in {"mmd", "energy"}:
            raise ValidationError("method must be 'mmd' or 'energy'")
        if not 0.0 < alpha < 1.0:
            raise ValidationError("alpha must be in (0, 1)")
        if n_permutations < 1:
            raise ValidationError("n_permutations must be >= 1")
        self.method = method
        self.alpha = float(alpha)
        self.n_permutations = int(n_permutations)
        self.gamma = gamma
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
        else:
            stat, p_value, threshold = energy_permutation_test(
                self._reference,
                current,
                n_permutations=self.n_permutations,
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
