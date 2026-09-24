"""CLI-facing facade over the structured detectors.

``UnifiedDriftDetector`` maps the CLI method names to the structured
``drift_control.detectors`` / ``distances`` implementations and returns a
``DriftResult``, while keeping the cross-cutting concerns the CLI relies on
(telemetry spans, sample-size checks, optional bootstrap CIs, columnar input
conversion). It no longer depends on the legacy flat detector modules.
"""

from __future__ import annotations

import time
import warnings
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import numpy as np

from .detectors import (
    DateTimeDriftDetector,
    MultivariateDriftDetector,
    UnivariateDriftDetector,
)
from .distances import (
    js_divergence,
    population_stability_index,
    to_histograms,
    wasserstein_distance,
    wasserstein_permutation_test,
)
from .result_schema import DriftResult

_UNIVARIATE = {
    "psi": "psi",
    "js": "js",
    "ks": "ks",
    "cvm": "cvm",
    "chi2cat": "chi2",
    "tvdcat": "tvd",
}
_MULTIVARIATE = {"mmd": "mmd", "energy": "energy", "c2st": "c2st"}
_THRESHOLD_DEFAULT = {"psi": 0.2, "js": 0.1, "tvdcat": 0.1, "datetime": 0.2}
_P_VALUE_METHODS = {"ks", "cvm", "chi2cat"}
_ALL_METHODS = set(_UNIVARIATE) | set(_MULTIVARIATE) | {"wasserstein", "datetime"}
_CI_METHODS = {"psi", "js", "wasserstein"}
_RECOMMENDED_MIN_SAMPLES = {
    "psi": 100,
    "js": 100,
    "wasserstein": 50,
    "ks": 20,
    "cvm": 20,
    "mmd": 50,
    "c2st": 50,
    "energy": 50,
    "chi2cat": 50,
    "tvdcat": 50,
    "datetime": 20,
}


@dataclass(frozen=True)
class _MethodPolicy:
    threshold: float
    comparator: str


def _validate_method(method: str) -> None:
    if method not in _ALL_METHODS:
        raise ValueError("method must be one of: " + ", ".join(sorted(_ALL_METHODS)))


def _resolve_method_policy(
    *,
    method: str,
    alpha: float,
    threshold: Any,
) -> _MethodPolicy:
    if method in _THRESHOLD_DEFAULT:
        return _MethodPolicy(
            threshold=float(threshold) if threshold is not None else _THRESHOLD_DEFAULT[method],
            comparator=">",
        )
    if method in _P_VALUE_METHODS:
        return _MethodPolicy(threshold=alpha, comparator="<")
    return _MethodPolicy(threshold=alpha, comparator=">")


def _maybe_convert_columnar(data: Any) -> Any:
    mod = getattr(getattr(data, "__class__", None), "__module__", "")
    if (
        isinstance(mod, str)
        and (mod.startswith("polars.") or mod.startswith("pyarrow."))
        and hasattr(data, "to_numpy")
    ):
        return data.to_numpy()
    return data


def _bootstrap_statistic(
    method: str,
    ref_sample: np.ndarray,
    cur_sample: np.ndarray,
    bins: int,
) -> float:
    if method == "psi":
        return population_stability_index(ref_sample, cur_sample, bins=bins)
    if method == "js":
        ref_hist, cur_hist = to_histograms(ref_sample, cur_sample, bins=bins)
        return js_divergence(ref_hist, cur_hist)
    return wasserstein_distance(ref_sample, cur_sample)


class UnifiedDriftDetector:
    """Consistent ``DriftResult`` across methods, backed by the structured stack."""

    def __init__(self, method: str = "psi", **kwargs: Any) -> None:
        self.method = method
        self.telemetry = kwargs.pop("telemetry", None)
        self.ci_bootstrap_samples = int(kwargs.pop("ci_bootstrap_samples", 0))
        self.ci_level = float(kwargs.pop("ci_level", 0.95))
        self.ci_random_state = int(kwargs.pop("ci_random_state", 42))
        if self.ci_bootstrap_samples < 0:
            raise ValueError("ci_bootstrap_samples must be >= 0")
        if not (0 < self.ci_level < 1):
            raise ValueError("ci_level must be between 0 and 1")
        _validate_method(method)

        self.alpha = float(kwargs.get("alpha", 0.05))
        self.n_permutations = int(kwargs.get("n_permutations", 200))
        self.random_state = int(kwargs.get("random_state", 42))
        self.bins = int(kwargs.get("bins", 10))
        self._policy = _resolve_method_policy(
            method=method,
            alpha=self.alpha,
            threshold=kwargs.get("threshold"),
        )
        self.threshold = self._policy.threshold
        self.comparator = self._policy.comparator

    @staticmethod
    def recommended_min_samples(method: str) -> int:
        """Recommended minimum samples per side for stable estimates."""
        return _RECOMMENDED_MIN_SAMPLES.get(method, 20)

    def _check_sample_sizes(self, reference_data: Any, current_data: Any) -> None:
        ref = np.asarray(reference_data)
        cur = np.asarray(current_data)
        n_ref = int(ref.shape[0]) if ref.ndim > 0 else int(ref.size)
        n_cur = int(cur.shape[0]) if cur.ndim > 0 else int(cur.size)
        if n_ref < 2 or n_cur < 2:
            raise ValueError(
                f"{self.method} requires at least 2 samples per side; got {n_ref} and {n_cur}"
            )
        rec = self.recommended_min_samples(self.method)
        if n_ref < rec or n_cur < rec:
            warnings.warn(
                f"{self.method} is being run with small sample sizes "
                f"(reference={n_ref}, current={n_cur}); recommended >= {rec} per side.",
                UserWarning,
                stacklevel=2,
            )

    def _bootstrap_ci(self, reference_data: Any, current_data: Any) -> tuple[float, float] | None:
        if self.ci_bootstrap_samples <= 0 or self.method not in _CI_METHODS:
            return None
        ref = np.asarray(reference_data, dtype=float).ravel()
        cur = np.asarray(current_data, dtype=float).ravel()
        if ref.size < 2 or cur.size < 2:
            return None
        rng = np.random.default_rng(self.ci_random_state)
        stats: list[float] = []
        for _ in range(self.ci_bootstrap_samples):
            ref_sample = ref[rng.integers(0, ref.size, ref.size)]
            cur_sample = cur[rng.integers(0, cur.size, cur.size)]
            stats.append(_bootstrap_statistic(self.method, ref_sample, cur_sample, self.bins))
        alpha_tail = 1.0 - self.ci_level
        return (
            float(np.quantile(stats, alpha_tail / 2.0)),
            float(np.quantile(stats, 1.0 - alpha_tail / 2.0)),
        )

    def _score(self, reference_data: Any, current_data: Any) -> DriftResult:
        method = self.method
        metadata: dict[str, Any] = {}
        if method in _UNIVARIATE:
            inner = (
                UnivariateDriftDetector(
                    method=_UNIVARIATE[method],
                    alpha=self.alpha,
                    threshold=self.threshold if method in _THRESHOLD_DEFAULT else None,
                    bins=self.bins,
                    correction="none",
                )
                .fit(reference_data)
                .detect(current_data)
            )
            score = inner.score
            drift = inner.drift
            p_value = inner.p_value
            threshold = self.threshold
        elif method in _MULTIVARIATE:
            inner = (
                MultivariateDriftDetector(
                    method=_MULTIVARIATE[method],
                    alpha=self.alpha,
                    n_permutations=self.n_permutations,
                    random_state=self.random_state,
                )
                .fit(reference_data)
                .detect(current_data)
            )
            score = inner.score
            drift = inner.drift
            p_value = inner.p_value
            threshold = float(inner.threshold) if inner.threshold is not None else self.alpha
            metadata = {"alpha": self.alpha, "calibrated_threshold": threshold}
        elif method == "wasserstein":
            score, p_value, threshold = wasserstein_permutation_test(
                reference_data,
                current_data,
                n_permutations=self.n_permutations,
                alpha=self.alpha,
                random_state=self.random_state,
            )
            drift = score > threshold
            metadata = {"alpha": self.alpha, "calibrated_threshold": threshold}
        else:
            inner = (
                DateTimeDriftDetector(threshold=self.threshold)
                .fit(reference_data)
                .detect(current_data)
            )
            score = inner.score
            drift = inner.drift
            p_value = inner.p_value
            threshold = self.threshold
        return DriftResult(
            method=method,
            drift=bool(drift),
            score=float(score),
            p_value=None if p_value is None else float(p_value),
            threshold=float(threshold),
            comparator=self.comparator,
            metadata=metadata,
        )

    def _span_context(self) -> Any:
        span_ctx: Any = nullcontext()
        if self.telemetry is not None:
            start_span = getattr(self.telemetry, "start_span", None)
            if callable(start_span):
                span_ctx = start_span(
                    "drift_control.detect_drift",
                    {"component": "detector", "method": self.method},
                )
        return span_ctx

    def _record_success(self, result: DriftResult) -> None:
        if self.telemetry is not None:
            self.telemetry.record_drift_rate(
                1.0 if result.drift else 0.0,
                {"component": "detector", "method": self.method},
            )

    def _record_error(self) -> None:
        if self.telemetry is not None:
            self.telemetry.record_error({"component": "detector", "method": self.method})

    def _record_latency(self, started: float) -> None:
        if self.telemetry is not None:
            self.telemetry.record_latency(
                (time.perf_counter() - started) * 1000.0,
                {"component": "detector", "method": self.method},
            )

    def detect_drift(self, reference_data: Any, current_data: Any) -> DriftResult:
        started = time.perf_counter()
        reference_data = _maybe_convert_columnar(reference_data)
        current_data = _maybe_convert_columnar(current_data)
        with self._span_context():
            try:
                self._check_sample_sizes(reference_data, current_data)
                result = self._score(reference_data, current_data)
                ci = self._bootstrap_ci(reference_data, current_data)
                if ci is not None:
                    result.metadata["score_ci"] = {
                        "lo": ci[0],
                        "hi": ci[1],
                        "level": self.ci_level,
                    }
                self._record_success(result)
                return result
            except Exception:
                self._record_error()
                raise
            finally:
                self._record_latency(started)


__all__ = ["UnifiedDriftDetector"]
