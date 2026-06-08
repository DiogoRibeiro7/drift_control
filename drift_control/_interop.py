"""Interop between the legacy detector interface and the core contract.

Entry points (``StreamMonitor``, the sklearn ``DriftMonitor``) historically
scored a column with ``detector.detect_drift(reference, current) -> (bool,
float)``. ``score_pair`` accepts that legacy interface *or* a core
:class:`~drift_control.core.base.BaseDetector` (``fit(reference).detect(current)
-> DriftResult``), so the structured detectors — and legacy ones wrapped via
``as_base_detector`` — work on those paths unchanged.
"""

from __future__ import annotations

from typing import Any


def score_pair(detector: Any, reference: Any, current: Any) -> tuple[bool, float]:
    """Return ``(drift, score)`` for one reference/current pair.

    Dispatches on the detector's interface: legacy ``detect_drift`` if present,
    otherwise the core ``fit``/``detect`` contract.
    """
    detect_drift = getattr(detector, "detect_drift", None)
    if callable(detect_drift):
        drift, score = detect_drift(reference, current)
        return bool(drift), float(score)
    result = detector.fit(reference).detect(current)
    return bool(result.drift_detected), float(result.score)


__all__ = ["score_pair"]
