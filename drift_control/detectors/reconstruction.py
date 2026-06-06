"""Reconstruction-error drift detection (ROADMAP.md Phase 10).

Fit a PCA subspace on the reference, then watch the per-sample reconstruction
error. A two-sample KS test compares the reference and current error
distributions, so shifts that push mass off the learned manifold are flagged.

Note: by construction this is most sensitive to change *orthogonal* to the
retained principal subspace; pair it with a distributional detector for shifts
along high-variance directions.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ..core.base import BaseDetector
from ..core.exceptions import NotFittedError, ValidationError
from ..core.result import DriftResult
from ..core.types import ArrayLike
from ..preprocessing import coerce_observations


class PCAReconstructionDriftDetector(BaseDetector):
    """Detect multivariate drift via PCA reconstruction error.

    :param n_components: passed to :class:`sklearn.decomposition.PCA`; ``None``
        retains 90% of the reference variance.
    :param alpha: significance level for the KS test on error distributions.
    :param standardize: z-score features on the reference before PCA.
    """

    def __init__(
        self,
        *,
        n_components: int | float | None = None,
        alpha: float = 0.05,
        standardize: bool = False,
        random_state: int = 42,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValidationError("alpha must be in (0, 1)")
        self.n_components = n_components
        self.alpha = float(alpha)
        self.standardize = bool(standardize)
        self.random_state = int(random_state)
        self._pca: PCA | None = None
        self._scaler: StandardScaler | None = None
        self._ref_errors: np.ndarray | None = None

    def _prepare(self, data: ArrayLike, name: str, *, fit: bool) -> np.ndarray:
        x = coerce_observations(data)
        if x.shape[1] < 2:
            raise ValidationError(f"{name} needs >= 2 features for PCA reconstruction")
        if self.standardize:
            if fit:
                self._scaler = StandardScaler().fit(x)
            assert self._scaler is not None
            x = self._scaler.transform(x)
        return x

    def _errors(self, x: np.ndarray) -> np.ndarray:
        assert self._pca is not None
        reconstructed = self._pca.inverse_transform(self._pca.transform(x))
        errors: np.ndarray = np.sum((x - reconstructed) ** 2, axis=1)
        return errors

    def fit(self, reference_data: ArrayLike) -> PCAReconstructionDriftDetector:
        x = self._prepare(reference_data, "reference_data", fit=True)
        n_components = self.n_components if self.n_components is not None else 0.9
        pca = PCA(n_components=n_components, random_state=self.random_state)
        pca.fit(x)
        self._pca = pca
        self._ref_errors = self._errors(x)
        return self

    def detect(self, current_data: ArrayLike) -> DriftResult:
        if self._pca is None or self._ref_errors is None:
            raise NotFittedError("call fit() before detect()")
        cur = self._prepare(current_data, "current_data", fit=False)
        cur_errors = self._errors(cur)
        result = ks_2samp(self._ref_errors, cur_errors)
        p_value = float(result.pvalue)
        return DriftResult.new(
            drift_detected=p_value < self.alpha,
            score=p_value,
            threshold=self.alpha,
            p_value=p_value,
            method="pca_reconstruction",
            comparator="<",
            metadata={
                "ks_statistic": float(result.statistic),
                "n_components": int(self._pca.n_components_),
                "ref_mean_error": float(np.mean(self._ref_errors)),
                "cur_mean_error": float(np.mean(cur_errors)),
            },
        )


__all__ = ["PCAReconstructionDriftDetector"]
