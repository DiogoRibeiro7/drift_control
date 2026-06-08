"""MultivariateDriftDetector: permutation-calibrated MMD / energy on the contract."""

import numpy as np
import pytest

import drift_control
from drift_control.core import BaseDetector, DriftResult
from drift_control.core.exceptions import NotFittedError, ValidationError
from drift_control.detectors import MultivariateDriftDetector
from drift_control.distances import energy_permutation_test

RNG = np.random.default_rng(0)


def _reconciles(r):
    return (r.score > r.threshold) == r.drift


# --- contract / validation --------------------------------------------------

def test_is_a_base_detector():
    assert isinstance(MultivariateDriftDetector(), BaseDetector)


def test_validation():
    with pytest.raises(ValidationError):
        MultivariateDriftDetector(method="kldiv")
    with pytest.raises(ValidationError):
        MultivariateDriftDetector(alpha=0.0)
    with pytest.raises(NotFittedError):
        MultivariateDriftDetector().detect(np.zeros((10, 3)))


def test_feature_count_mismatch():
    det = MultivariateDriftDetector(method="energy").fit(np.zeros((20, 3)))
    with pytest.raises(ValidationError, match="feature count mismatch"):
        det.detect(np.zeros((20, 4)))


# --- MMD --------------------------------------------------------------------

def test_mmd_no_drift_and_drift():
    ref = RNG.normal(0, 1, (150, 3))
    det = MultivariateDriftDetector(method="mmd", n_permutations=100, random_state=1).fit(ref)
    no = det.detect(RNG.normal(0, 1, (150, 3)))
    yes = det.detect(RNG.normal(1.2, 1, (150, 3)))
    assert no.drift is False and yes.drift is True
    assert _reconciles(no) and _reconciles(yes)
    assert no.method == "mmd" and no.p_value is not None


# --- energy -----------------------------------------------------------------

def test_energy_no_drift_and_drift():
    ref = RNG.normal(0, 1, (150, 3))
    det = MultivariateDriftDetector(method="energy", n_permutations=100, random_state=1).fit(ref)
    assert det.detect(RNG.normal(0, 1, (150, 3))).drift is False
    yes = det.detect(RNG.normal(1.5, 1, (150, 3)))
    assert yes.drift is True
    assert _reconciles(yes)


def test_energy_permutation_test_primitive():
    ref = RNG.normal(0, 1, (120, 2))
    stat, p_same, thr = energy_permutation_test(ref, RNG.normal(0, 1, (120, 2)), n_permutations=100)
    stat2, p_diff, thr2 = energy_permutation_test(
        ref, RNG.normal(2.0, 1, (120, 2)), n_permutations=100
    )
    assert p_diff < p_same
    assert stat2 > thr2  # drifted statistic exceeds its calibrated threshold


# --- determinism + integration ----------------------------------------------

def test_deterministic_with_seed():
    ref = RNG.normal(0, 1, (80, 2))
    cur = RNG.normal(0.5, 1, (80, 2))
    a = MultivariateDriftDetector(n_permutations=60, random_state=7).fit(ref).detect(cur)
    b = MultivariateDriftDetector(n_permutations=60, random_state=7).fit(ref).detect(cur)
    assert (a.score, a.p_value, a.threshold) == (b.score, b.p_value, b.threshold)


def test_flows_into_drift_report():
    from drift_control.monitoring import DriftReport

    ref = RNG.normal(0, 1, (120, 3))
    result = MultivariateDriftDetector(method="energy", n_permutations=80).fit(ref).detect(
        RNG.normal(2.0, 1, (120, 3))
    )
    report = DriftReport.from_results([result], names=["multivariate"])
    assert report.any_drift is True
    assert isinstance(result, DriftResult)


def test_exposed_at_package_root():
    from drift_control import detectors

    assert drift_control.MultivariateDriftDetector is detectors.MultivariateDriftDetector
