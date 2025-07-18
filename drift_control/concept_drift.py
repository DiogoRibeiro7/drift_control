from typing import Iterable
from river.drift.binary import DDM, EDDM


class DDMDetector:
    """Wrapper around river's DDM concept drift detector."""

    def __init__(self) -> None:
        self.ddm = DDM()

    def update(self, value: float) -> bool:
        """Feed a performance metric and return True if drift detected."""
        self.ddm.update(value)
        return self.ddm.drift_detected


class EDDMDetector:
    """Wrapper around river's EDDM concept drift detector."""

    def __init__(self) -> None:
        self.eddm = EDDM()

    def update(self, value: float) -> bool:
        """Feed a performance metric and return True if drift detected."""
        self.eddm.update(value)
        return self.eddm.drift_detected


class AccuracyMonitor:
    """Track prediction accuracy over time and detect concept drift."""

    def __init__(self, detector: object | None = None) -> None:
        self.detector = detector or DDMDetector()

    def update(self, y_true: Iterable, y_pred: Iterable) -> bool:
        """Update the monitor with a batch of predictions.

        Returns True if concept drift is detected by the underlying detector.
        """
        import numpy as np

        true_arr = np.asarray(list(y_true))
        pred_arr = np.asarray(list(y_pred))
        if true_arr.shape != pred_arr.shape:
            raise ValueError("y_true and y_pred must have the same shape")

        drift = False
        for correct in (true_arr == pred_arr):
            drift = self.detector.update(float(not correct))
            if drift:
                break
        return drift
