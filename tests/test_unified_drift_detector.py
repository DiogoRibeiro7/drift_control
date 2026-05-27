import numpy as np
import pytest

from drift_control.result_schema import DriftResult
from drift_control.unified_drift_detector import UnifiedDriftDetector


def test_unified_psi_result_schema():
    detector = UnifiedDriftDetector(method="psi")
    result = detector.detect_drift([1, 2, 3, 4, 5], [10, 11, 12, 13, 14])
    assert isinstance(result, DriftResult)
    assert result.method == "psi"
    assert result.comparator == ">"
    assert result.p_value is None


def test_unified_ks_contains_pvalue():
    detector = UnifiedDriftDetector(method="ks", alpha=0.05)
    result = detector.detect_drift([1, 2, 3, 4, 5], [10, 11, 12, 13, 14])
    assert result.method == "ks"
    assert result.p_value is not None
    assert result.comparator == "<"


def test_unified_wasserstein_contains_calibrated_metadata():
    detector = UnifiedDriftDetector(method="wasserstein", alpha=0.05, n_permutations=80)
    result = detector.detect_drift([1, 2, 3, 4, 5], [10, 11, 12, 13, 14])
    assert result.method == "wasserstein"
    assert result.p_value is not None
    assert "calibrated_threshold" in result.metadata


def test_unified_mmd_multivariate():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, size=(120, 2))
    cur = rng.normal(1.0, 1, size=(120, 2))
    detector = UnifiedDriftDetector(method="mmd", alpha=0.05, n_permutations=80)
    result = detector.detect_drift(ref, cur)
    assert result.method == "mmd"
    assert result.p_value is not None


def test_unified_rejects_unknown_method():
    with pytest.raises(ValueError, match="method must be one of"):
        UnifiedDriftDetector(method="not-real")
import numpy as np

from drift_control.unified_drift_detector import UnifiedDriftDetector


def test_unified_c2st_multivariate():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, size=(120, 3))
    cur = rng.normal(0.9, 1, size=(120, 3))
    detector = UnifiedDriftDetector(method='c2st', alpha=0.05, n_permutations=60)
    out = detector.detect_drift(ref, cur)
    assert out.method == 'c2st'
    assert out.p_value is not None
    assert 'calibrated_threshold' in out.metadata
