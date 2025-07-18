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
