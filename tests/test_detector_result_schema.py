import numpy as np

from drift_control.c2st_drift_detector import C2STDriftDetector
from drift_control.categorical_chi2_drift_detector import ChiSquareDriftDetector
from drift_control.categorical_tvd_drift_detector import TotalVariationDriftDetector
from drift_control.cvm_drift_detector import CVMDriftDetector
from drift_control.energy_drift_detector import EnergyDriftDetector
from drift_control.js_drift_detector import JensenShannonDriftDetector
from drift_control.ks_drift_detector import KSDriftDetector
from drift_control.mmd_drift_detector import MMDDriftDetector
from drift_control.psi_drift_detector import PSIDriftDetector
from drift_control.result_schema import DriftResult
from drift_control.wasserstein_drift_detector import WassersteinDriftDetector


def _assert_common(result: DriftResult) -> None:
    assert isinstance(result, DriftResult)
    assert isinstance(result.drift, bool)
    assert isinstance(result.score, float)
    assert isinstance(result.threshold, float)
    assert result.comparator in {"<", ">"}
    _assert_reconciles(result)


def _assert_reconciles(result: DriftResult) -> None:
    """The (score, comparator, threshold) triplet must reproduce ``drift``.

    Without this, a detector can report a decision that contradicts the
    machine-readable fields downstream consumers rely on.
    """
    if result.comparator == ">":
        expected = result.score > result.threshold
    else:
        expected = result.score < result.threshold
    assert expected == result.drift, (
        f"{result.method}: score={result.score} {result.comparator} "
        f"threshold={result.threshold} does not match drift={result.drift}"
    )


def test_univariate_detectors_expose_common_result_schema():
    ref = np.array([0, 1, 2, 3, 4, 5], dtype=float)
    cur = np.array([1, 2, 3, 4, 5, 6], dtype=float)

    for detector in [
        PSIDriftDetector(),
        KSDriftDetector(),
        CVMDriftDetector(),
        JensenShannonDriftDetector(),
        WassersteinDriftDetector(n_permutations=50, random_state=0),
    ]:
        _assert_common(detector.detect_drift_result(ref, cur))


def test_categorical_detectors_expose_common_result_schema():
    ref = np.array(["a", "a", "b", "c"])
    cur = np.array(["a", "b", "c", "c"])

    for detector in [ChiSquareDriftDetector(), TotalVariationDriftDetector()]:
        _assert_common(detector.detect_drift_result(ref, cur))


def test_multivariate_detectors_expose_common_result_schema():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, size=(40, 3))
    cur = rng.normal(0.4, 1, size=(40, 3))

    for detector in [
        MMDDriftDetector(n_permutations=50, random_state=0),
        C2STDriftDetector(n_permutations=50, random_state=0),
        EnergyDriftDetector(n_permutations=50, random_state=0),
    ]:
        _assert_common(detector.detect_drift_result(ref, cur))
