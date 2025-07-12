import numpy as np
from scipy.stats import ks_2samp

class KSDriftDetector:
    """Detect drift using the Kolmogorov-Smirnov test."""

    def __init__(self, alpha: float = 0.05) -> None:
        """Create detector with a significance level."""
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
