import pytest
from drift_control.concept_drift import DDMDetector, EDDMDetector


def test_ddm_detects_drift():
    detector = DDMDetector()
    drift = False
    for i in range(100):
        drift = detector.update(0.1)
    for i in range(100):
        drift = detector.update(1.0)
        if drift:
            break
    assert drift


def test_eddm_detects_drift():
    detector = EDDMDetector()
    drift = False
    for i in range(150):
        drift = detector.update(0.1)
    for i in range(150):
        drift = detector.update(1.0)
        if drift:
            break
    assert drift
