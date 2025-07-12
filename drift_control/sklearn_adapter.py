from typing import Dict, Any
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .psi_drift_detector import PSIDriftDetector

class DriftMonitor(BaseEstimator, TransformerMixin):
    """A scikit-learn compatible transformer for monitoring data drift."""

    def __init__(self, detector: Any | None = None) -> None:
        self.detector = detector or PSIDriftDetector()
        self.baseline_: pd.DataFrame | None = None
        self.drift_results_: Dict[str, Dict[str, Any]] | None = None

    def fit(self, X: pd.DataFrame, y: Any = None):
        """Store the baseline dataset."""
        self.baseline_ = pd.DataFrame(X).copy()
        return self

    def transform(self, X: pd.DataFrame, y: Any = None):
        """Check drift against the baseline and return data unchanged."""
        if self.baseline_ is None:
            raise ValueError("Call fit before transform")
        X_df = pd.DataFrame(X)
        results: Dict[str, Dict[str, Any]] = {}
        for col in self.baseline_.columns:
            drift, score = self.detector.detect_drift(self.baseline_[col], X_df[col])
            results[col] = {"drift": drift, "score": score}
        self.drift_results_ = results
        return X
