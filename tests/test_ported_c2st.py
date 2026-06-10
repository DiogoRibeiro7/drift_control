"""Ported C2ST (classifier two-sample / covariate shift) on MultivariateDriftDetector."""

import numpy as np
import pytest

from drift_control.core.exceptions import ValidationError
from drift_control.detectors import MultivariateDriftDetector

RNG = np.random.default_rng(0)


def test_c2st_no_drift_and_drift():
    ref = RNG.normal(0, 1, (200, 4))
    det = MultivariateDriftDetector(method="c2st", n_permutations=60, random_state=1).fit(ref)
    no = det.detect(RNG.normal(0, 1, (200, 4)))
    yes = det.detect(RNG.normal(1.5, 1, (200, 4)))
    assert no.drift is False and yes.drift is True
    assert no.method == "c2st" and no.p_value is not None
    assert (yes.score > yes.threshold) == yes.drift  # AUC > calibrated threshold


def test_c2st_validation():
    with pytest.raises(ValidationError):
        MultivariateDriftDetector(method="c2st", test_size=1.5)
    det = MultivariateDriftDetector(method="c2st", n_permutations=60).fit(np.zeros((1, 3)))
    with pytest.raises(ValidationError):
        det.detect(np.zeros((1, 3)))  # < 2 samples per side


def test_c2st_deterministic():
    ref = RNG.normal(0, 1, (120, 3))
    cur = RNG.normal(0.8, 1, (120, 3))
    a = MultivariateDriftDetector(method="c2st", n_permutations=50, random_state=7).fit(ref).detect(cur)
    b = MultivariateDriftDetector(method="c2st", n_permutations=50, random_state=7).fit(ref).detect(cur)
    assert (a.score, a.p_value, a.threshold) == (b.score, b.p_value, b.threshold)
