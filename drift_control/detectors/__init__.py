"""Detectors built on the core contracts and distance primitives.

See ``ROADMAP.md``. All detectors live here and compose
``drift_control.distances``; the legacy flat ``*_drift_detector`` modules have
been removed.
"""

from __future__ import annotations

from .change_point import (
    EWMAChart,
    SegmentationResult,
    ShewhartChart,
    binary_segmentation,
    pelt,
    window_based_change_detection,
)
from .concept_drift import CUSUM, DDM, EDDM, KSWIN, PageHinkley
from .data_drift import UnivariateDriftDetector
from .datetime_drift import DateTimeDriftDetector
from .ensemble import DetectorEnsemble
from .mixed_type import MixedTypeDriftDetector
from .multivariate import MultivariateDriftDetector
from .online import run_online
from .reconstruction import PCAReconstructionDriftDetector

__all__ = [
    "UnivariateDriftDetector",
    "DateTimeDriftDetector",
    "DDM",
    "EDDM",
    "PageHinkley",
    "CUSUM",
    "KSWIN",
    "ShewhartChart",
    "EWMAChart",
    "SegmentationResult",
    "binary_segmentation",
    "pelt",
    "window_based_change_detection",
    "run_online",
    "PCAReconstructionDriftDetector",
    "MultivariateDriftDetector",
    "MixedTypeDriftDetector",
    "DetectorEnsemble",
]
