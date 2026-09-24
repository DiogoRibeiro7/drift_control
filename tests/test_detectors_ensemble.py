"""DetectorEnsemble: voting over core detectors."""

import numpy as np
import pytest

import drift_control
from drift_control.core import BaseDetector, DriftResult
from drift_control.core.exceptions import ValidationError
from drift_control.detectors import (
    DetectorEnsemble,
    UnivariateDriftDetector,
)

RNG = np.random.default_rng(0)


class _Stub(BaseDetector):
    """Member that always votes a fixed decision (for testing vote logic)."""

    def __init__(self, drift: bool) -> None:
        self._drift = drift

    def fit(self, reference_data):
        return self

    def detect(self, new_data):
        return DriftResult.new(
            drift_detected=self._drift, score=1.0 if self._drift else 0.0, threshold=0.5
        )


def _ensemble(votes, vote):
    return DetectorEnsemble([_Stub(v) for v in votes], vote=vote).fit(None)


# --- contract / validation --------------------------------------------------


def test_is_a_base_detector():
    assert isinstance(DetectorEnsemble([_Stub(True)]), BaseDetector)


def test_validation():
    with pytest.raises(ValidationError):
        DetectorEnsemble([])
    with pytest.raises(ValidationError):
        DetectorEnsemble([_Stub(True)], vote="weighted")
    with pytest.raises(ValidationError):
        DetectorEnsemble([_Stub(True)], names=["a", "b"])


# --- vote logic + reconciliation -------------------------------------------


@pytest.mark.parametrize(
    "votes,vote,expected",
    [
        ([True, True, False], "any", True),
        ([False, False, False], "any", False),
        ([True, True, False], "majority", True),  # 2/3
        ([True, False, False], "majority", False),  # 1/3
        ([True, True, True], "all", True),
        ([True, True, False], "all", False),
    ],
)
def test_vote_modes(votes, vote, expected):
    result = _ensemble(votes, vote).detect(None)
    assert result.drift is expected
    assert (result.score > result.threshold) == result.drift  # reconciles
    assert result.metadata["n_drifting"] == sum(votes)
    assert result.metadata["vote"] == vote


def test_metadata_lists_members_with_names():
    ens = DetectorEnsemble([_Stub(True), _Stub(False)], names=["a", "b"]).fit(None)
    members = ens.detect(None).metadata["members"]
    assert [m["name"] for m in members] == ["a", "b"]
    assert [m["drift"] for m in members] == [True, False]


# --- integration with real detectors ---------------------------------------


def test_real_detectors_members():
    ref = RNG.normal(0, 1, 400)
    ensemble = DetectorEnsemble(
        [
            UnivariateDriftDetector(method="ks"),
            UnivariateDriftDetector(method="psi", threshold=0.2),
            UnivariateDriftDetector(method="js", threshold=0.1),
        ],
        vote="majority",
        names=["ks", "psi", "js"],
    ).fit(ref)
    assert ensemble.detect(RNG.normal(0, 1, 400)).drift is False
    assert ensemble.detect(RNG.normal(2.0, 1, 400)).drift is True


def test_flows_into_drift_report():
    from drift_control.monitoring import DriftReport

    ref = RNG.normal(0, 1, 300)
    result = (
        DetectorEnsemble([UnivariateDriftDetector(method="ks")])
        .fit(ref)
        .detect(RNG.normal(3.0, 1, 300))
    )
    assert DriftReport.from_results([result], names=["ensemble"]).any_drift is True


def test_exposed_at_package_root():
    from drift_control import detectors

    assert drift_control.DetectorEnsemble is detectors.DetectorEnsemble
