"""Helpers for driving online detectors over a sequence."""

from __future__ import annotations

from collections.abc import Iterable

from ..core.base import OnlineDetector


def run_online(detector: OnlineDetector, values: Iterable[float]) -> list[int]:
    """Feed ``values`` to an :class:`OnlineDetector`, returning drift indices.

    Returns the positions at which ``update`` reported drift. The package's
    online detectors restart their own statistics after signalling, so a single
    pass detects multiple successive change points.
    """
    return [
        i
        for i, value in enumerate(values)
        if detector.update(float(value)).drift_detected
    ]


__all__ = ["run_online"]
