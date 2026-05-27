import numpy as np

from drift_control.energy_drift_detector import EnergyDriftDetector, EnergyResult


def test_energy_detector_detects_shift_with_details():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, size=(120, 2))
    cur = rng.normal(0.8, 1, size=(120, 2))
    detector = EnergyDriftDetector(alpha=0.05, n_permutations=80, random_state=7)
    result = detector.detect_drift(ref, cur, return_details=True)
    assert isinstance(result, EnergyResult)
    assert result.p_value >= 0.0
    assert result.p_value <= 1.0
    assert result.threshold >= 0.0


def test_energy_detector_simple_api_returns_tuple():
    ref = [0.0, 1.0, 2.0, 3.0]
    cur = [2.0, 3.0, 4.0, 5.0]
    detector = EnergyDriftDetector(alpha=0.05, n_permutations=20, random_state=1)
    drift, score = detector.detect_drift(ref, cur)
    assert isinstance(drift, bool)
    assert isinstance(score, float)
