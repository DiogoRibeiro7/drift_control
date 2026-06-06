"""Detectors built on the core contracts and distance primitives.

See ``ROADMAP.md``. New detectors live here and compose ``drift_control.distances``;
the existing flat ``*_drift_detector`` modules remain until they are migrated.
"""

from __future__ import annotations

from .change_point import (
    EWMAChart,
    SegmentationResult,
    ShewhartChart,
    binary_segmentation,
    window_based_change_detection,
)
from .concept_drift import CUSUM, DDM, EDDM, PageHinkley
from .data_drift import UnivariateDriftDetector
from .reconstruction import PCAReconstructionDriftDetector

__all__ = [
    "UnivariateDriftDetector",
    "DDM",
    "EDDM",
    "PageHinkley",
    "CUSUM",
    "ShewhartChart",
    "EWMAChart",
    "SegmentationResult",
    "binary_segmentation",
    "window_based_change_detection",
    "PCAReconstructionDriftDetector",
]
