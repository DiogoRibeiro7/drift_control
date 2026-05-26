"""Scikit-learn compatible drift monitoring transformer."""

from typing import Dict, Any
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .psi_drift_detector import PSIDriftDetector


class DriftMonitor(BaseEstimator, TransformerMixin):
    """A scikit-learn compatible transformer for monitoring data drift.

    ``transform`` passes input through unchanged but records the last drift
    scores on ``drift_results_`` so the estimator slots into a ``Pipeline``.
    For uses outside a pipeline prefer :meth:`score_drift`, which returns
    the scores directly without mutating instance state.
    """

    def __init__(self, detector: Any | None = None) -> None:
        self.detector = detector or PSIDriftDetector()
        self.baseline_: pd.DataFrame | None = None
        self.drift_results_: Dict[str, Dict[str, Any]] | None = None
        self.feature_names_in_: pd.Index | None = None
        self.n_features_in_: int | None = None

    def fit(self, X: pd.DataFrame, y: Any = None):
        """Store the baseline dataset."""
        X_df = pd.DataFrame(X).copy()
        self.baseline_ = X_df
        self.feature_names_in_ = X_df.columns
        self.n_features_in_ = len(X_df.columns)
        return self

    def _check_input(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.baseline_ is None:
            raise ValueError("Call fit before scoring drift")
        X_df = pd.DataFrame(X)
        if self.n_features_in_ != len(X_df.columns):
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {len(X_df.columns)}."
            )
        if self.feature_names_in_ is None:
            raise ValueError("Missing fit schema metadata.")
        if list(X_df.columns) != list(self.feature_names_in_):
            expected = list(self.feature_names_in_)
            got = list(X_df.columns)
            raise ValueError(
                f"Feature schema mismatch. Expected columns in order {expected}, got {got}."
            )
        return X_df

    def score_drift(self, X: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Score drift against the baseline without mutating instance state."""
        X_df = self._check_input(X)
        results: Dict[str, Dict[str, Any]] = {}
        assert self.baseline_ is not None
        for col in self.baseline_.columns:
            drift, score = self.detector.detect_drift(self.baseline_[col], X_df[col])
            results[col] = {"drift": drift, "score": score}
        return results

    def transform(self, X: pd.DataFrame, y: Any = None):
        """Check drift against the baseline and return data unchanged.

        Side-effect: stores results on ``self.drift_results_``. Use
        :meth:`score_drift` if you need a pure call.
        """
        self.drift_results_ = self.score_drift(X)
        return X
