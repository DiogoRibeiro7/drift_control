"""Guard the public detector surface documented in docs/detectors.md.

The legacy flat ``*DriftDetector`` modules were removed in favour of a single
structured track. This keeps the documented guidance from rotting: every
canonical name resolves at the package root, the ported niche methods are
reachable through ``UnifiedDriftDetector``, and the online concept-drift
detectors really are pure-Python + ``DriftResult``-returning (no river dep).
"""

import numpy as np

import drift_control

# Canonical structured names the docs point users at.
_STRUCTURED = [
    "UnivariateDriftDetector",
    "MultivariateDriftDetector",
    "MixedTypeDriftDetector",
    "DateTimeDriftDetector",
    "DetectorEnsemble",
    "PCAReconstructionDriftDetector",
    "DDM",
    "EDDM",
    "PageHinkley",
    "CUSUM",
    "KSWIN",
    "UnifiedDriftDetector",
]

# Niche methods with no dedicated class, reachable through UnifiedDriftDetector.
_UNIFIED_METHODS = ["cvm", "tvdcat", "c2st", "datetime"]


def test_structured_names_resolve():
    for name in _STRUCTURED:
        assert getattr(drift_control, name) is not None, name


def test_unified_exposes_ported_niche_methods():
    for method in _UNIFIED_METHODS:
        detector = drift_control.UnifiedDriftDetector(method=method)
        assert detector.method == method


def test_structured_online_detectors_are_pure_python_driftresult():
    # The recommendation rests on these returning a DriftResult with no river dep.
    from drift_control.core import DriftResult

    for cls in (drift_control.DDM, drift_control.EDDM, drift_control.PageHinkley,
                drift_control.CUSUM):
        result = cls().update(1.0)
        assert isinstance(result, DriftResult)


def test_univariate_detector_returns_driftresult():
    from drift_control.core import DriftResult

    out = (
        drift_control.UnivariateDriftDetector(method="ks")
        .fit(np.zeros(50))
        .detect(np.ones(50) * 5)
    )
    assert isinstance(out, DriftResult)
