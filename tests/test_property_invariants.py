import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from drift_control.js_drift_detector import JensenShannonDriftDetector
from drift_control.ks_drift_detector import KSDriftDetector
from drift_control.psi_drift_detector import PSIDriftDetector
from drift_control.wasserstein_drift_detector import WassersteinDriftDetector

_float_list = st.lists(
    st.integers(min_value=-1000, max_value=1000).map(float),
    min_size=30,
    max_size=80,
)


@given(_float_list)
@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_psi_identical_samples_have_near_zero_score(values):
    arr = np.asarray(values, dtype=float)
    detector = PSIDriftDetector(threshold=0.2)
    drift, score = detector.detect_drift(arr, arr.copy())
    assert drift is False
    assert score < 1e-9


@given(_float_list)
@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_js_identical_samples_have_near_zero_score(values):
    arr = np.asarray(values, dtype=float)
    detector = JensenShannonDriftDetector(threshold=0.1)
    drift, score = detector.detect_drift(arr, arr.copy())
    assert drift is False
    assert score < 1e-9


@given(_float_list)
@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_ks_identical_samples_have_high_pvalue(values):
    arr = np.asarray(values, dtype=float)
    detector = KSDriftDetector(alpha=0.05)
    drift, p_value = detector.detect_drift(arr, arr.copy())
    assert drift is False
    assert p_value >= 0.99


@given(_float_list)
@settings(
    max_examples=15,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_wasserstein_identical_samples_have_zero_distance(values):
    arr = np.asarray(values, dtype=float)
    detector = WassersteinDriftDetector(alpha=0.05, n_permutations=50, random_state=7)
    details = detector.detect_drift(arr, arr.copy(), return_details=True)
    assert details.drift_detected is False
    assert details.distance == 0.0
