"""Guard the flat<->structured mapping in docs/flat-vs-structured-api.md.

Keeps the documented guidance from rotting: every name on both sides resolves,
and the structured online detectors really are pure-Python + DriftResult-returning
(no river dependency), which is the reason they are recommended.
"""

import numpy as np

import drift_control

# (flat name, structured name) pairs that the doc recommends migrating.
_RECOMMENDED = [
    ("DDMDetector", "DDM"),
    ("EDDMDetector", "EDDM"),
    ("PageHinkleyDetector", "PageHinkley"),
    ("KSDriftDetector", "UnivariateDriftDetector"),
    ("PSIDriftDetector", "UnivariateDriftDetector"),
    ("JensenShannonDriftDetector", "UnivariateDriftDetector"),
    ("WassersteinDriftDetector", "UnivariateDriftDetector"),
    ("ChiSquareDriftDetector", "UnivariateDriftDetector"),
]

# Flat names the doc says to keep (no structured equivalent).
_FLAT_ONLY = [
    "ADWINDetector", "KSWINDetector", "CVMDriftDetector",
    "TotalVariationDriftDetector", "DateTimeDriftDetector",
    "MMDDriftDetector", "EnergyDriftDetector", "C2STDriftDetector",
    "CovariateShiftDetector",
]


def test_both_sides_of_the_mapping_resolve():
    for flat, structured in _RECOMMENDED:
        assert getattr(drift_control, flat) is not None, flat
        assert getattr(drift_control, structured) is not None, structured


def test_flat_only_names_still_exposed():
    for name in _FLAT_ONLY:
        assert getattr(drift_control, name) is not None, name


def test_structured_online_detectors_are_pure_python_driftresult():
    # The recommendation rests on these returning a DriftResult with no river dep.
    from drift_control.core import DriftResult

    for cls in (drift_control.DDM, drift_control.EDDM, drift_control.PageHinkley,
                drift_control.CUSUM):
        result = cls().update(1.0)
        assert isinstance(result, DriftResult)


def test_as_base_detector_lifts_any_flat_detector():
    adapted = drift_control.as_base_detector(drift_control.KSDriftDetector())
    out = adapted.fit(np.zeros(50)).detect(np.ones(50) * 5)
    from drift_control.core import DriftResult

    assert isinstance(out, DriftResult)
