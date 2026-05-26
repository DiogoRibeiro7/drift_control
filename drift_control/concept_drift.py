from typing import Iterable

try:
    from river.drift.binary import DDM, EDDM
except ImportError:  # pragma: no cover - exercised in minimal installs
    DDM = None
    EDDM = None


class DDMDetector:
    """Wrapper around river's DDM concept drift detector."""

    def __init__(self) -> None:
        if DDM is None:
            raise ImportError(
                "river is required for DDMDetector. Install with: pip install 'drift-control[concept]'"
            )
        self.ddm = DDM()

    def update(self, value: float) -> bool:
        """Feed a performance metric and return True if drift detected."""
        self.ddm.update(value)
        return self.ddm.drift_detected


class EDDMDetector:
    """Wrapper around river's EDDM concept drift detector."""

    def __init__(self) -> None:
        if EDDM is None:
            raise ImportError(
                "river is required for EDDMDetector. Install with: pip install 'drift-control[concept]'"
            )
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

        Every sample in the batch is fed to the underlying detector so its
        internal state stays in sync with the observed stream. Returns True
        if drift was detected at any point during this batch.
        """
        import numpy as np

        true_arr = np.asarray(list(y_true))
        pred_arr = np.asarray(list(y_pred))
        if true_arr.shape != pred_arr.shape:
            raise ValueError("y_true and y_pred must have the same shape")

        drift_any = False
        for correct in (true_arr == pred_arr):
            if self.detector.update(float(not correct)):
                drift_any = True
        return drift_any
