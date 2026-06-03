"""Detectors built on the core contracts and distance primitives.

See ``ROADMAP.md``. New detectors live here and compose ``drift_control.distances``;
the existing flat ``*_drift_detector`` modules remain until they are migrated.
"""

from __future__ import annotations

from .data_drift import UnivariateDriftDetector

__all__ = ["UnivariateDriftDetector"]
