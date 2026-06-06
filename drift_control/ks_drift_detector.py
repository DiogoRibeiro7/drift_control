import numpy as np
from scipy.stats import ks_2samp

from .result_schema import DriftResult


class KSDriftDetector:
    """Detect drift using the Kolmogorov-Smirnov test."""

    def __init__(self, alpha: float = 0.05) -> None:
        """Create detector with a significance level."""
        if not (0 < alpha < 1):
            raise ValueError("alpha must be between 0 and 1")
        self.alpha = alpha

    def calculate_pvalue(self, reference, current) -> float:
        """Return the p-value of the KS test."""
        ref = np.asarray(reference, dtype=float)
        cur = np.asarray(current, dtype=float)
        _, pvalue = ks_2samp(ref, cur)
        return float(pvalue)

    def detect_drift(self, reference, current):
        """Return whether drift is detected and the p-value."""
        pvalue = self.calculate_pvalue(reference, current)
        return pvalue < self.alpha, pvalue

    def detect_drift_result(self, reference, current) -> DriftResult:
        drift, pvalue = self.detect_drift(reference, current)
        return DriftResult(
            method="ks",
            drift=bool(drift),
            score=float(pvalue),
            p_value=float(pvalue),
            threshold=float(self.alpha),
            comparator="<",
            metadata={},
        )
