"""Phase 10: PCA reconstruction-error drift detector."""

import numpy as np
import pytest

import drift_control
from drift_control.core import BaseDetector, DriftResult
from drift_control.core.exceptions import NotFittedError, ValidationError
from drift_control.detectors import PCAReconstructionDriftDetector

RNG = np.random.default_rng(0)
_W = RNG.normal(size=(2, 6))  # 2D latent -> 6D observed


def _manifold(n, noise):
    z = RNG.normal(size=(n, 2))
    return z @ _W + RNG.normal(0.0, noise, size=(n, 6))


# --- contract ---------------------------------------------------------------

def test_is_a_base_detector():
    assert isinstance(PCAReconstructionDriftDetector(), BaseDetector)


def test_requires_two_features():
    det = PCAReconstructionDriftDetector()
    with pytest.raises(ValidationError):
        det.fit(np.zeros((10, 1)))


def test_detect_before_fit():
    with pytest.raises(NotFittedError):
        PCAReconstructionDriftDetector().detect(np.zeros((10, 3)))


def test_invalid_alpha():
    with pytest.raises(ValidationError):
        PCAReconstructionDriftDetector(alpha=1.0)


# --- detection --------------------------------------------------------------

def test_no_drift_same_manifold():
    ref = _manifold(400, 0.05)
    det = PCAReconstructionDriftDetector(n_components=2).fit(ref)
    result = det.detect(_manifold(400, 0.05))
    assert isinstance(result, DriftResult)
    assert result.drift is False
    assert (result.score < result.threshold) == result.drift  # reconciles


def test_drift_when_mass_moves_off_manifold():
    ref = _manifold(400, 0.05)
    det = PCAReconstructionDriftDetector(n_components=2).fit(ref)
    result = det.detect(_manifold(400, 0.8))  # large off-manifold noise
    assert result.drift is True
    assert result.metadata["cur_mean_error"] > result.metadata["ref_mean_error"]
    assert (result.score < result.threshold) == result.drift


def test_default_variance_components_runs_and_standardize():
    ref = _manifold(300, 0.1)
    det = PCAReconstructionDriftDetector(standardize=True).fit(ref)  # n_components=None -> 0.9
    result = det.detect(_manifold(300, 0.1))
    assert det._pca is not None
    assert result.metadata["n_components"] >= 1


# --- exposure ---------------------------------------------------------------

def test_exposed_at_package_root():
    from drift_control import detectors

    assert (
        drift_control.PCAReconstructionDriftDetector
        is detectors.PCAReconstructionDriftDetector
    )
