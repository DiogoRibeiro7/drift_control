import numpy as np
import pytest

from drift_control.result_schema import DriftResult
from drift_control.unified_drift_detector import UnifiedDriftDetector


class _SpyTelemetry:
    def __init__(self):
        self.latency_calls = 0
        self.error_calls = 0
        self.drift_rate_calls = 0
        self.span_calls = 0

    def record_latency(self, value_ms, attributes=None):
        self.latency_calls += 1

    def record_error(self, attributes=None):
        self.error_calls += 1

    def record_drift_rate(self, value, attributes=None):
        self.drift_rate_calls += 1

    class _SpanCtx:
        def __init__(self, owner):
            self.owner = owner

        def __enter__(self):
            self.owner.span_calls += 1
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def start_span(self, name, attributes=None):
        return _SpyTelemetry._SpanCtx(self)


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


def test_unified_psi_bootstrap_ci_metadata():
    detector = UnifiedDriftDetector(method="psi", ci_bootstrap_samples=40, ci_random_state=0)
    result = detector.detect_drift([1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7])
    assert "score_ci" in result.metadata
    ci = result.metadata["score_ci"]
    assert ci["lo"] <= ci["hi"]
    assert 0 < ci["level"] < 1


def test_unified_js_bootstrap_ci_metadata():
    detector = UnifiedDriftDetector(method="js", ci_bootstrap_samples=30, ci_random_state=1)
    result = detector.detect_drift([1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7])
    assert "score_ci" in result.metadata


def test_unified_wasserstein_bootstrap_ci_metadata():
    detector = UnifiedDriftDetector(
        method="wasserstein",
        alpha=0.05,
        n_permutations=60,
        ci_bootstrap_samples=30,
        ci_random_state=2,
    )
    result = detector.detect_drift([1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7])
    assert "calibrated_threshold" in result.metadata
    assert "score_ci" in result.metadata


def test_unified_warns_on_small_sample_sizes():
    detector = UnifiedDriftDetector(method="ks", alpha=0.05)
    with pytest.warns(UserWarning, match="recommended"):
        detector.detect_drift([1, 2, 3], [1, 2, 4])


def test_unified_raises_below_two_samples():
    detector = UnifiedDriftDetector(method="ks", alpha=0.05)
    with pytest.raises(ValueError, match="at least 2 samples"):
        detector.detect_drift([1], [1, 2, 3])



def test_unified_c2st_multivariate():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, size=(120, 3))
    cur = rng.normal(0.9, 1, size=(120, 3))
    detector = UnifiedDriftDetector(method='c2st', alpha=0.05, n_permutations=60)
    out = detector.detect_drift(ref, cur)
    assert out.method == 'c2st'
    assert out.p_value is not None
    assert 'calibrated_threshold' in out.metadata


def test_unified_energy_multivariate():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, size=(120, 3))
    cur = rng.normal(0.9, 1, size=(120, 3))
    detector = UnifiedDriftDetector(method='energy', alpha=0.05, n_permutations=60, random_state=7)
    out = detector.detect_drift(ref, cur)
    assert out.method == 'energy'
    assert out.p_value is not None
    assert 'calibrated_threshold' in out.metadata


def test_unified_chi2cat():
    detector = UnifiedDriftDetector(method='chi2cat', alpha=0.05)
    out = detector.detect_drift(['a', 'a', 'b', 'c'], ['c', 'c', 'c', 'b'])
    assert out.method == 'chi2cat'
    assert out.p_value is not None


def test_unified_tvdcat():
    detector = UnifiedDriftDetector(method='tvdcat', threshold=0.1)
    out = detector.detect_drift(['a', 'a', 'b', 'c'], ['c', 'c', 'c', 'b'])
    assert out.method == 'tvdcat'
    assert out.p_value is None


def test_unified_telemetry_records_latency_and_drift_rate():
    spy = _SpyTelemetry()
    detector = UnifiedDriftDetector(method="psi", telemetry=spy)
    _ = detector.detect_drift([1, 2, 3, 4, 5], [10, 11, 12, 13, 14])
    assert spy.latency_calls == 1
    assert spy.drift_rate_calls == 1
    assert spy.error_calls == 0
    assert spy.span_calls == 1


def test_unified_telemetry_records_errors():
    spy = _SpyTelemetry()
    detector = UnifiedDriftDetector(method="ks", alpha=0.05, telemetry=spy)
    with pytest.raises(ValueError):
        detector.detect_drift([1], [1, 2, 3])
    assert spy.error_calls == 1
    assert spy.latency_calls == 1


def test_unified_accepts_polars_like_series_for_psi():
    class _FakePolarsSeries:
        __module__ = "polars.series.series"

        def __init__(self, values):
            self._values = np.asarray(values, dtype=float)

        def to_numpy(self):
            return self._values

    detector = UnifiedDriftDetector(method="psi")
    out = detector.detect_drift(
        _FakePolarsSeries([1, 2, 3, 4, 5, 6]),
        _FakePolarsSeries([2, 3, 4, 5, 6, 7]),
    )
    assert out.method == "psi"
    assert isinstance(out.score, float)


def test_unified_accepts_polars_like_series_for_wasserstein():
    class _FakePolarsSeries:
        __module__ = "polars.series.series"

        def __init__(self, values):
            self._values = np.asarray(values, dtype=float)

        def to_numpy(self):
            return self._values

    detector = UnifiedDriftDetector(method="wasserstein", alpha=0.05, n_permutations=50)
    out = detector.detect_drift(
        _FakePolarsSeries([1, 2, 3, 4, 5, 6]),
        _FakePolarsSeries([2, 3, 4, 5, 6, 7]),
    )
    assert out.method == "wasserstein"
    assert out.p_value is not None


def test_unified_accepts_polars_like_series_for_js():
    class _FakePolarsSeries:
        __module__ = "polars.series.series"

        def __init__(self, values):
            self._values = np.asarray(values, dtype=float)

        def to_numpy(self):
            return self._values

    detector = UnifiedDriftDetector(method="js", threshold=0.05)
    out = detector.detect_drift(
        _FakePolarsSeries([1, 2, 3, 4, 5, 6]),
        _FakePolarsSeries([2, 3, 4, 5, 6, 7]),
    )
    assert out.method == "js"
    assert isinstance(out.score, float)


def test_unified_accepts_polars_like_frame_for_c2st():
    class _FakePolarsFrame:
        __module__ = "polars.dataframe.frame"

        def __init__(self, values):
            self._values = np.asarray(values, dtype=float)

        def to_numpy(self):
            return self._values

    rng = np.random.default_rng(0)
    ref = _FakePolarsFrame(rng.normal(0, 1, size=(120, 3)))
    cur = _FakePolarsFrame(rng.normal(0.8, 1, size=(120, 3)))
    detector = UnifiedDriftDetector(method="c2st", alpha=0.05, n_permutations=50)
    out = detector.detect_drift(ref, cur)
    assert out.method == "c2st"
    assert out.p_value is not None


def test_unified_accepts_pyarrow_like_series_for_ks():
    class _FakeArrowArray:
        __module__ = "pyarrow.lib"

        def __init__(self, values):
            self._values = np.asarray(values, dtype=float)

        def to_numpy(self):
            return self._values

    detector = UnifiedDriftDetector(method="ks", alpha=0.05)
    out = detector.detect_drift(
        _FakeArrowArray([1, 2, 3, 4, 5, 6]),
        _FakeArrowArray([2, 3, 4, 5, 6, 7]),
    )
    assert out.method == "ks"
    assert out.p_value is not None


def test_unified_accepts_pyarrow_like_table_for_energy():
    class _FakeArrowTable:
        __module__ = "pyarrow.lib"

        def __init__(self, values):
            self._values = np.asarray(values, dtype=float)

        def to_numpy(self):
            return self._values

    rng = np.random.default_rng(0)
    ref = _FakeArrowTable(rng.normal(0, 1, size=(120, 3)))
    cur = _FakeArrowTable(rng.normal(0.8, 1, size=(120, 3)))
    detector = UnifiedDriftDetector(method="energy", alpha=0.05, n_permutations=50, random_state=7)
    out = detector.detect_drift(ref, cur)
    assert out.method == "energy"
    assert out.p_value is not None


def test_unified_metadata_includes_execution_backend_for_arrow_psi():
    class _FakeArrowArray:
        __module__ = "pyarrow.lib"

        def __init__(self, values):
            self._values = np.asarray(values, dtype=float)

        def to_numpy(self):
            return self._values

    detector = UnifiedDriftDetector(method="psi", strategy="uniform")
    out = detector.detect_drift(
        _FakeArrowArray([1, 2, 3, 4, 5, 6]),
        _FakeArrowArray([2, 3, 4, 5, 6, 7]),
    )
    assert out.metadata.get("execution_backend") in {"numpy", "pyarrow"}


def test_unified_metadata_includes_execution_backend_for_numpy_psi():
    detector = UnifiedDriftDetector(method="psi", strategy="uniform")
    out = detector.detect_drift([1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7])
    assert out.metadata.get("execution_backend") == "numpy"


def test_unified_datetime_method():
    ref = np.array(
        ["2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z", "2026-01-01T02:00:00Z", "2026-01-01T03:00:00Z"]
    )
    cur = np.array(
        ["2026-01-02T10:00:00Z", "2026-01-02T11:00:00Z", "2026-01-02T12:00:00Z", "2026-01-02T13:00:00Z"]
    )
    detector = UnifiedDriftDetector(method="datetime", threshold=0.01)
    out = detector.detect_drift(ref, cur)
    assert out.method == "datetime"
    assert out.p_value is None
