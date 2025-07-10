class DriftDetector:
    """Simple drift detector based on mean squared error."""

    def __init__(self, threshold: float) -> None:
        """Create detector with an error threshold."""
        self.threshold = threshold

    def detect_drift(self, reference_data, current_data):
        """Return whether drift is detected and the calculated error."""
        import numpy as np

        ref = np.array(reference_data)
        cur = np.array(current_data)
        error = np.mean((ref - cur) ** 2)
        drift_detected = error > self.threshold
        return drift_detected, error
