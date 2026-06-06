"""Migration: legacy detectors bridged onto the core BaseDetector contract."""

import numpy as np
import pytest

import drift_control
from drift_control.core import BaseDetector, DriftResult
from drift_control.core.exceptions import NotFittedError, ValidationError
from drift_control.detectors import LegacyDetectorAdapter, as_base_detector
from drift_control.ks_drift_detector import KSDriftDetector
from drift_control.monitoring import DriftReport
from drift_control.psi_drift_detector import PSIDriftDetector

RNG = np.random.default_rng(0)


def test_adapter_is_a_base_detector():
    adapted = as_base_detector(KSDriftDetector(alpha=0.05))
    assert isinstance(adapted, BaseDetector)
    assert isinstance(adapted, LegacyDetectorAdapter)


def test_fit_detect_returns_core_result():
    adapted = as_base_detector(KSDriftDetector(alpha=0.05))
    assert adapted.fit(RNG.normal(0, 1, 500)) is adapted
    result = adapted.detect(RNG.normal(2.0, 1, 500))
    assert isinstance(result, DriftResult)
    assert result.method == "ks"
    assert result.drift_detected is True


def test_detect_before_fit_raises():
    with pytest.raises(NotFittedError):
        as_base_detector(KSDriftDetector()).detect([1, 2, 3])


def test_rejects_object_without_detect_drift_result():
    with pytest.raises(ValidationError):
        as_base_detector(object())


def test_bridges_legacy_detectors_into_drift_report():
    ref = RNG.normal(0, 1, 400)
    cur = RNG.normal(1.5, 1, 400)
    results = [
        as_base_detector(KSDriftDetector()).fit(ref).detect(cur),
        as_base_detector(PSIDriftDetector(threshold=0.2)).fit(ref).detect(cur),
    ]
    report = DriftReport.from_results(results, names=["ks", "psi"])
    assert report.n_total == 2
    assert report.any_drift is True


def test_exposed_at_package_root():
    from drift_control import detectors

    assert drift_control.as_base_detector is detectors.as_base_detector
