class DriftDetector:
    """Paired-sample mean-squared-error check.

    This detector compares two arrays element-wise and flags drift when
    their MSE exceeds a threshold. It does **not** compare distributions;
    it requires the two inputs to be paired observations of equal length
    (e.g. predictions vs. ground truth, or sensor readings before/after a
    deterministic transform). For genuine distribution drift use
    :class:`PSIDriftDetector`, :class:`KSDriftDetector`, or
    :class:`CovariateShiftDetector`.
    """

    def __init__(self, threshold: float) -> None:
        """Create detector with an error threshold."""
        if threshold < 0:
            raise ValueError("threshold must be non-negative")
        self.threshold = threshold

    def detect_drift(self, reference_data, current_data):
        """Return whether drift is detected and the calculated error."""
        import numpy as np

        ref = np.asarray(reference_data, dtype=float).ravel()
        cur = np.asarray(current_data, dtype=float).ravel()
        if ref.size == 0 or cur.size == 0:
            raise ValueError("reference_data and current_data must be non-empty")
        if ref.shape != cur.shape:
            raise ValueError(
                f"reference_data and current_data must have the same shape; got {ref.shape} and {cur.shape}"
            )
        if not np.isfinite(ref).all() or not np.isfinite(cur).all():
            raise ValueError("reference_data and current_data must contain only finite values")
        error = np.mean((ref - cur) ** 2)
        drift_detected = error > self.threshold
        return drift_detected, error
