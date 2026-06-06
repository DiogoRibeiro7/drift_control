"""Bridge legacy detectors onto the core ``BaseDetector`` contract.

The flat ``*_drift_detector`` classes are stateless: ``detect_drift_result(ref,
current)``. This adapter wraps any of them as a :class:`BaseDetector`
(``fit(reference)`` then ``detect(current)``), so legacy detectors compose with
the structured machinery — ``DriftReport``, retraining policies, and future
ensembles — without modifying each class.
"""

from __future__ import annotations

from typing import Any

from ..core.base import BaseDetector
from ..core.exceptions import NotFittedError, ValidationError
from ..core.result import DriftResult


class LegacyDetectorAdapter(BaseDetector):
    """Expose a legacy ``detect_drift_result`` detector as a ``BaseDetector``."""

    def __init__(self, detector: Any) -> None:
        if not callable(getattr(detector, "detect_drift_result", None)):
            raise ValidationError(
                "detector must expose detect_drift_result(reference, current)"
            )
        self.detector = detector
        self._reference: Any = None
        self._fitted = False

    def fit(self, reference_data: Any) -> LegacyDetectorAdapter:
        self._reference = reference_data
        self._fitted = True
        return self

    def detect(self, new_data: Any) -> DriftResult:
        if not self._fitted:
            raise NotFittedError("call fit() before detect()")
        result: DriftResult = self.detector.detect_drift_result(
            self._reference, new_data
        )
        return result


def as_base_detector(detector: Any) -> LegacyDetectorAdapter:
    """Wrap a legacy detector so it satisfies the core fit/detect contract."""
    return LegacyDetectorAdapter(detector)


__all__ = ["LegacyDetectorAdapter", "as_base_detector"]
