from __future__ import annotations

from typing import Any, Protocol, Union, cast

from .c2st_drift_detector import C2STDriftDetector
from .categorical_chi2_drift_detector import ChiSquareDriftDetector
from .categorical_tvd_drift_detector import TotalVariationDriftDetector
from .cvm_drift_detector import CVMDriftDetector
from .js_drift_detector import JensenShannonDriftDetector
from .ks_drift_detector import KSDriftDetector
from .mmd_drift_detector import MMDDriftDetector
from .psi_drift_detector import PSIDriftDetector
from .result_schema import DriftResult
from .wasserstein_drift_detector import WassersteinDriftDetector


class _SimpleDetector(Protocol):
    def detect_drift(self, reference_data: Any, current_data: Any) -> tuple[bool, float]:
        ...


class _DetailedDetector(Protocol):
    def detect_drift(
        self, reference_data: Any, current_data: Any, return_details: bool = False
    ) -> Any:
        ...


class UnifiedDriftDetector:
    """Facade providing a consistent DriftResult across detector methods."""

    def __init__(self, method: str = "psi", **kwargs: Any) -> None:
        self.method = method
        self.kwargs = kwargs
        self.detector: Union[_SimpleDetector, _DetailedDetector]
        self.threshold: float
        self.comparator: str

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
        elif method == "c2st":
            self.detector = C2STDriftDetector(**kwargs)
            self.threshold = float(self.detector.alpha)
            self.comparator = "<"
        elif method == "chi2cat":
            self.detector = ChiSquareDriftDetector(**kwargs)
            self.threshold = float(self.detector.alpha)
            self.comparator = "<"
        elif method == "tvdcat":
            self.detector = TotalVariationDriftDetector(**kwargs)
            self.threshold = float(self.detector.threshold)
            self.comparator = ">"
        else:
            raise ValueError(
                "method must be one of: psi, ks, cvm, js, wasserstein, mmd, c2st, chi2cat, tvdcat"
            )

    def detect_drift(self, reference_data: Any, current_data: Any) -> DriftResult:
        if self.method == "wasserstein":
            details = cast(_DetailedDetector, self.detector).detect_drift(
                reference_data, current_data, return_details=True
            )
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
            details = cast(_DetailedDetector, self.detector).detect_drift(
                reference_data, current_data, return_details=True
            )
            return DriftResult(
                method=self.method,
                drift=bool(details.drift_detected),
                score=float(details.mmd2),
                p_value=float(details.p_value),
                threshold=self.threshold,
                comparator=self.comparator,
                metadata={"calibrated_threshold": float(details.threshold)},
            )

        if self.method == "c2st":
            details = cast(_DetailedDetector, self.detector).detect_drift(
                reference_data, current_data, return_details=True
            )
            return DriftResult(
                method=self.method,
                drift=bool(details.drift_detected),
                score=float(details.roc_auc),
                p_value=float(details.p_value),
                threshold=self.threshold,
                comparator=self.comparator,
                metadata={"calibrated_threshold": float(details.threshold)},
            )

        drift, score = cast(_SimpleDetector, self.detector).detect_drift(
            reference_data, current_data
        )
        p_value = float(score) if self.method in {"ks", "cvm", "chi2cat"} else None
        return DriftResult(
            method=self.method,
            drift=bool(drift),
            score=float(score),
            p_value=p_value,
            threshold=self.threshold,
            comparator=self.comparator,
            metadata={},
        )
