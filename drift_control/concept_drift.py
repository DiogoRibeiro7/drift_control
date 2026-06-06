from collections.abc import Iterable

try:
    from river.drift import ADWIN, KSWIN, PageHinkley
    from river.drift.binary import DDM, EDDM
except ImportError:  # pragma: no cover - exercised in minimal installs
    ADWIN = None
    KSWIN = None
    PageHinkley = None
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


class ADWINDetector:
    """Wrapper around river's ADWIN concept drift detector."""

    def __init__(self, delta: float = 0.002) -> None:
        if ADWIN is None:
            raise ImportError(
                "river is required for ADWINDetector. Install with: pip install 'drift-control[concept]'"
            )
        self.adwin = ADWIN(delta=delta)

    def update(self, value: float) -> bool:
        """Feed a performance metric and return True if drift detected."""
        self.adwin.update(value)
        return self.adwin.drift_detected


class PageHinkleyDetector:
    """Wrapper around river's Page-Hinkley concept drift detector."""

    def __init__(
        self,
        min_instances: int = 30,
        delta: float = 0.005,
        threshold: float = 50.0,
        alpha: float = 0.9999,
    ) -> None:
        if PageHinkley is None:
            raise ImportError(
                "river is required for PageHinkleyDetector. Install with: pip install 'drift-control[concept]'"
            )
        self.page_hinkley = PageHinkley(
            min_instances=min_instances,
            delta=delta,
            threshold=threshold,
            alpha=alpha,
        )

    def update(self, value: float) -> bool:
        """Feed a performance metric and return True if drift detected."""
        self.page_hinkley.update(value)
        return self.page_hinkley.drift_detected


class KSWINDetector:
    """Wrapper around river's KSWIN concept drift detector."""

    def __init__(
        self,
        alpha: float = 0.005,
        window_size: int = 100,
        stat_size: int = 30,
        seed: int | None = None,
    ) -> None:
        if KSWIN is None:
            raise ImportError(
                "river is required for KSWINDetector. Install with: pip install 'drift-control[concept]'"
            )
        self.kswin = KSWIN(
            alpha=alpha,
            window_size=window_size,
            stat_size=stat_size,
            seed=seed,
        )

    def update(self, value: float) -> bool:
        """Feed a scalar value and return True if drift detected."""
        self.kswin.update(value)
        return self.kswin.drift_detected


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
