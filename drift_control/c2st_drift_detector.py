from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from .result_schema import DriftResult


@dataclass(frozen=True)
class C2STResult:
    drift_detected: bool
    roc_auc: float
    p_value: float
    threshold: float


class C2STDriftDetector:
    """Classifier two-sample test with permutation-calibrated ROC-AUC."""

    def __init__(
        self,
        alpha: float = 0.05,
        n_permutations: int = 100,
        test_size: float = 0.3,
        random_state: int = 42,
    ) -> None:
        if not (0 < alpha < 1):
            raise ValueError("alpha must be between 0 and 1")
        if n_permutations < 50:
            raise ValueError("n_permutations must be >= 50")
        if not (0 < test_size < 1):
            raise ValueError("test_size must be between 0 and 1")
        self.alpha = float(alpha)
        self.n_permutations = int(n_permutations)
        self.test_size = float(test_size)
        self.random_state = int(random_state)

    @staticmethod
    def _as_2d_array(x, name: str) -> np.ndarray:
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.ndim != 2:
            raise ValueError(f"{name} must be 1D or 2D array-like")
        if arr.shape[0] < 2:
            raise ValueError(f"{name} must contain at least 2 samples")
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} must contain only finite values")
        return arr

    def _fit_auc(self, X: np.ndarray, y: np.ndarray) -> float:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)[:, 1]
        return float(roc_auc_score(y_test, proba))

    def detect_drift(self, reference_data, current_data, return_details: bool = False):
        ref = self._as_2d_array(reference_data, "reference_data")
        cur = self._as_2d_array(current_data, "current_data")
        if ref.shape[1] != cur.shape[1]:
            raise ValueError(
                "reference_data and current_data must have the same number of features"
            )

        X = np.vstack([ref, cur])
        y = np.concatenate([np.zeros(ref.shape[0]), np.ones(cur.shape[0])])

        observed_auc = self._fit_auc(X, y)

        rng = np.random.default_rng(self.random_state)
        null_aucs = np.empty(self.n_permutations, dtype=float)
        for i in range(self.n_permutations):
            y_perm = rng.permutation(y)
            null_aucs[i] = self._fit_auc(X, y_perm)

        threshold = float(np.quantile(null_aucs, 1.0 - self.alpha))
        p_value = float((1.0 + np.sum(null_aucs >= observed_auc)) / (1.0 + self.n_permutations))
        drift = bool(observed_auc > threshold)

        if return_details:
            return C2STResult(
                drift_detected=drift,
                roc_auc=observed_auc,
                p_value=p_value,
                threshold=threshold,
            )
        return drift, observed_auc

    def detect_drift_result(self, reference_data, current_data) -> DriftResult:
        details = self.detect_drift(reference_data, current_data, return_details=True)
        return DriftResult(
            method="c2st",
            drift=bool(details.drift_detected),
            score=float(details.roc_auc),
            p_value=float(details.p_value),
            threshold=float(self.alpha),
            comparator="<",
            metadata={"calibrated_threshold": float(details.threshold)},
        )
