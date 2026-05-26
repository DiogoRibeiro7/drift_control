import numpy as np
from scipy.stats import cramervonmises_2samp


class CVMDriftDetector:
    """Detect drift using the two-sample Cramer-von Mises test."""

    def __init__(self, alpha: float = 0.05) -> None:
        if not (0 < alpha < 1):
            raise ValueError("alpha must be between 0 and 1")
        self.alpha = alpha

    def calculate_pvalue(self, reference, current) -> float:
        ref = np.asarray(reference, dtype=float).ravel()
        cur = np.asarray(current, dtype=float).ravel()
        if ref.size == 0 or cur.size == 0:
            raise ValueError("reference and current must be non-empty")
        result = cramervonmises_2samp(ref, cur)
        return float(result.pvalue)

    def detect_drift(self, reference, current):
        pvalue = self.calculate_pvalue(reference, current)
        return pvalue < self.alpha, pvalue
