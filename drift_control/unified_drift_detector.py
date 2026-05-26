from __future__ import annotations

from typing import Any

from .cvm_drift_detector import CVMDriftDetector
from .js_drift_detector import JensenShannonDriftDetector
from .ks_drift_detector import KSDriftDetector
from .mmd_drift_detector import MMDDriftDetector
from .psi_drift_detector import PSIDriftDetector
from .result_schema import DriftResult
from .wasserstein_drift_detector import WassersteinDriftDetector


class UnifiedDriftDetector:
    """Facade providing a consistent DriftResult across detector methods."""

    def __init__(self, method: str = "psi", **kwargs: Any) -> None:
        self.method = method
        self.kwargs = kwargs

        if method == "psi":
            self.detector = PSIDriftDetector(**kwargs)
            self.threshold = float(self.detector.threshold)
            self.comparator = ">"
        elif method == "ks":
            self.detector = KSDriftDetector(**kwargs)
            self.threshold = float(self.detector.alpha)
            self.comparator = "<"
        elif method == "cvm":
            self.detector = CVMDriftDetector(**kwargs)
            self.threshold = float(self.detector.alpha)
            self.comparator = "<"
        elif method == "js":
            self.detector = JensenShannonDriftDetector(**kwargs)
            self.threshold = float(self.detector.threshold)
            self.comparator = ">"
        elif method == "wasserstein":
            self.detector = WassersteinDriftDetector(**kwargs)
            self.threshold = float(self.detector.alpha)
            self.comparator = "<"
        elif method == "mmd":
            self.detector = MMDDriftDetector(**kwargs)
            self.threshold = float(self.detector.alpha)
            self.comparator = "<"
        else:
            raise ValueError(
                "method must be one of: psi, ks, cvm, js, wasserstein, mmd"
            )

    def detect_drift(self, reference_data, current_data) -> DriftResult:
        if self.method == "wasserstein":
            details = self.detector.detect_drift(reference_data, current_data, return_details=True)
            return DriftResult(
                method=self.method,
                drift=bool(details.drift_detected),
                score=float(details.distance),
                p_value=float(details.p_value),
                threshold=self.threshold,
                comparator=self.comparator,
                metadata={"calibrated_threshold": float(details.threshold)},
            )

        if self.method == "mmd":
            details = self.detector.detect_drift(reference_data, current_data, return_details=True)
            return DriftResult(
                method=self.method,
                drift=bool(details.drift_detected),
                score=float(details.mmd2),
                p_value=float(details.p_value),
                threshold=self.threshold,
                comparator=self.comparator,
                metadata={"calibrated_threshold": float(details.threshold)},
            )

        drift, score = self.detector.detect_drift(reference_data, current_data)
        p_value = float(score) if self.method in {"ks", "cvm"} else None
        return DriftResult(
            method=self.method,
            drift=bool(drift),
            score=float(score),
            p_value=p_value,
            threshold=self.threshold,
            comparator=self.comparator,
            metadata={},
        )
