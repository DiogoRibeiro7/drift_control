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

# CLI method -> structured univariate method (numeric or categorical).
_UNIVARIATE = {"psi": "psi", "js": "js", "ks": "ks", "cvm": "cvm",
               "chi2cat": "chi2", "tvdcat": "tvd"}
_MULTIVARIATE = {"mmd": "mmd", "energy": "energy", "c2st": "c2st"}
_THRESHOLD_DEFAULT = {"psi": 0.2, "js": 0.1, "tvdcat": 0.1, "datetime": 0.2}
_PVALUE_METHODS = {"ks", "cvm", "chi2cat"}
_CALIBRATED = {"mmd", "energy", "c2st", "wasserstein"}
_ALL_METHODS = set(_UNIVARIATE) | set(_MULTIVARIATE) | {"wasserstein", "datetime"}
_CI_METHODS = {"psi", "js", "wasserstein"}


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
        if method not in _ALL_METHODS:
            raise ValueError(
                "method must be one of: " + ", ".join(sorted(_ALL_METHODS))
            )

        self.alpha = float(kwargs.get("alpha", 0.05))
        self.n_permutations = int(kwargs.get("n_permutations", 200))
        self.random_state = int(kwargs.get("random_state", 42))
        self.bins = int(kwargs.get("bins", 10))
        threshold = kwargs.get("threshold")

        if method in _THRESHOLD_DEFAULT:
            self.threshold = (
                float(threshold) if threshold is not None else _THRESHOLD_DEFAULT[method]
            )
            self.comparator = ">"
        elif method in _PVALUE_METHODS:
            self.threshold = self.alpha
            self.comparator = "<"
        else:  # calibrated (mmd/energy/c2st/wasserstein)
            self.threshold = self.alpha
            self.comparator = ">"

    @staticmethod
    def recommended_min_samples(method: str) -> int:
        """Recommended minimum samples per side for stable estimates."""
        rec = {
            "psi": 100, "js": 100, "wasserstein": 50, "ks": 20, "cvm": 20,
            "mmd": 50, "c2st": 50, "energy": 50, "chi2cat": 50, "tvdcat": 50,
            "datetime": 20,
        }
        return rec.get(method, 20)

    def _check_sample_sizes(self, reference_data: Any, current_data: Any) -> None:
        ref = np.asarray(reference_data)
        cur = np.asarray(current_data)
        n_ref = int(ref.shape[0]) if ref.ndim > 0 else int(ref.size)
        n_cur = int(cur.shape[0]) if cur.ndim > 0 else int(cur.size)
        if n_ref < 2 or n_cur < 2:
            raise ValueError(
                f"{self.method} requires at least 2 samples per side; "
                f"got {n_ref} and {n_cur}"
            )
        rec = self.recommended_min_samples(self.method)
        if n_ref < rec or n_cur < rec:
            warnings.warn(
                f"{self.method} is being run with small sample sizes "
                f"(reference={n_ref}, current={n_cur}); recommended >= {rec} per side.",
                UserWarning,
                stacklevel=2,
            )

    @staticmethod
    def _maybe_convert_columnar(data: Any) -> Any:
        mod = getattr(getattr(data, "__class__", None), "__module__", "")
        if (
            isinstance(mod, str)
            and (mod.startswith("polars.") or mod.startswith("pyarrow."))
            and hasattr(data, "to_numpy")
        ):
            return data.to_numpy()
        return data

    def _bootstrap_ci(
        self, reference_data: Any, current_data: Any
    ) -> tuple[float, float] | None:
        if self.ci_bootstrap_samples <= 0 or self.method not in _CI_METHODS:
            return None
        ref = np.asarray(reference_data, dtype=float).ravel()
        cur = np.asarray(current_data, dtype=float).ravel()
        if ref.size < 2 or cur.size < 2:
            return None
        rng = np.random.default_rng(self.ci_random_state)
        stats: list[float] = []
        for _ in range(self.ci_bootstrap_samples):
            ref_s = ref[rng.integers(0, ref.size, ref.size)]
            cur_s = cur[rng.integers(0, cur.size, cur.size)]
            if self.method == "psi":
                stats.append(population_stability_index(ref_s, cur_s, bins=self.bins))
            elif self.method == "js":
                ref_p, cur_p = to_histograms(ref_s, cur_s, bins=self.bins)
                stats.append(js_divergence(ref_p, cur_p))
            else:
                stats.append(wasserstein_distance(ref_s, cur_s))
        a = 1.0 - self.ci_level
        return float(np.quantile(stats, a / 2.0)), float(np.quantile(stats, 1.0 - a / 2.0))

    def _score(self, reference_data: Any, current_data: Any) -> DriftResult:
        method = self.method
        metadata: dict[str, Any] = {}
        if method in _UNIVARIATE:
            detector = UnivariateDriftDetector(
                method=_UNIVARIATE[method],
                alpha=self.alpha,
                threshold=self.threshold if method in _THRESHOLD_DEFAULT else None,
                bins=self.bins,
                correction="none",
            )
            inner = detector.fit(reference_data).detect(current_data)
            score, drift, p_value = inner.score, inner.drift, inner.p_value
            threshold = self.threshold
        elif method in _MULTIVARIATE:
            inner = MultivariateDriftDetector(
                method=_MULTIVARIATE[method],
                alpha=self.alpha,
                n_permutations=self.n_permutations,
                random_state=self.random_state,
            ).fit(reference_data).detect(current_data)
            score, drift, p_value = inner.score, inner.drift, inner.p_value
            threshold = float(inner.threshold) if inner.threshold is not None else self.alpha
            metadata = {"alpha": self.alpha, "calibrated_threshold": threshold}
        elif method == "wasserstein":
            score, p_value, threshold = wasserstein_permutation_test(
                reference_data, current_data, n_permutations=self.n_permutations,
                alpha=self.alpha, random_state=self.random_state,
            )
            drift = score > threshold
            metadata = {"alpha": self.alpha, "calibrated_threshold": threshold}
        else:  # datetime
            inner = DateTimeDriftDetector(threshold=self.threshold).fit(
                reference_data
            ).detect(current_data)
            score, drift, p_value = inner.score, inner.drift, inner.p_value
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

    def detect_drift(self, reference_data: Any, current_data: Any) -> DriftResult:
        started = time.perf_counter()
        reference_data = self._maybe_convert_columnar(reference_data)
        current_data = self._maybe_convert_columnar(current_data)
        span_ctx: Any = nullcontext()
        if self.telemetry is not None:
            start_span = getattr(self.telemetry, "start_span", None)
            if callable(start_span):
                span_ctx = start_span(
                    "drift_control.detect_drift",
                    {"component": "detector", "method": self.method},
                )
        with span_ctx:
            try:
                self._check_sample_sizes(reference_data, current_data)
                result = self._score(reference_data, current_data)
                ci = self._bootstrap_ci(reference_data, current_data)
                if ci is not None:
                    result.metadata["score_ci"] = {
                        "lo": ci[0], "hi": ci[1], "level": self.ci_level
                    }
                if self.telemetry is not None:
                    self.telemetry.record_drift_rate(
                        1.0 if result.drift else 0.0,
                        {"component": "detector", "method": self.method},
                    )
                return result
            except Exception:
                if self.telemetry is not None:
                    self.telemetry.record_error(
                        {"component": "detector", "method": self.method}
                    )
                raise
            finally:
                if self.telemetry is not None:
                    self.telemetry.record_latency(
                        (time.perf_counter() - started) * 1000.0,
                        {"component": "detector", "method": self.method},
                    )


__all__ = ["UnifiedDriftDetector"]
