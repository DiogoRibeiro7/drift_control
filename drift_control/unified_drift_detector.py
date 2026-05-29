from __future__ import annotations

from typing import Any, Protocol, Union, cast
import numpy as np
import time
import warnings
from contextlib import nullcontext

from .c2st_drift_detector import C2STDriftDetector
from .categorical_chi2_drift_detector import ChiSquareDriftDetector
from .categorical_tvd_drift_detector import TotalVariationDriftDetector
from .cvm_drift_detector import CVMDriftDetector
from .energy_drift_detector import EnergyDriftDetector
from .datetime_drift_detector import DateTimeDriftDetector
from .js_drift_detector import JensenShannonDriftDetector
from .ks_drift_detector import KSDriftDetector
from .mmd_drift_detector import MMDDriftDetector
from .psi_drift_detector import PSIDriftDetector
from .result_schema import DriftResult
from .wasserstein_drift_detector import WassersteinDriftDetector


class _SimpleDetector(Protocol):
    def detect_drift(self, reference_data: Any, current_data: Any) -> tuple[bool, float]:
        ...


class _DetailedDetector(Protocol):
    def detect_drift(
        self, reference_data: Any, current_data: Any, return_details: bool = False
    ) -> Any:
        ...


class UnifiedDriftDetector:
    """Facade providing a consistent DriftResult across detector methods."""

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
        self.kwargs = kwargs
        self.detector: Union[_SimpleDetector, _DetailedDetector]
        self.threshold: float
        self.comparator: str

        if method == "psi":
            self.detector = PSIDriftDetector(**kwargs)
            self.threshold = float(self.detector.threshold)
            self.comparator = ">"
        elif method == "ks":
            self.detector = KSDriftDetector(**kwargs)
            self.threshold = float(self.detector.alpha)
            self.comparator = "<"
        elif method == "cvm":
            self.detector = CVMDriftDetector(**kwargs)
            self.threshold = float(self.detector.alpha)
            self.comparator = "<"
        elif method == "js":
            self.detector = JensenShannonDriftDetector(**kwargs)
            self.threshold = float(self.detector.threshold)
            self.comparator = ">"
        elif method == "wasserstein":
            self.detector = WassersteinDriftDetector(**kwargs)
            self.threshold = float(self.detector.alpha)
            self.comparator = "<"
        elif method == "mmd":
            self.detector = MMDDriftDetector(**kwargs)
            self.threshold = float(self.detector.alpha)
            self.comparator = "<"
        elif method == "c2st":
            self.detector = C2STDriftDetector(**kwargs)
            self.threshold = float(self.detector.alpha)
            self.comparator = "<"
        elif method == "energy":
            self.detector = EnergyDriftDetector(**kwargs)
            self.threshold = float(self.detector.alpha)
            self.comparator = "<"
        elif method == "chi2cat":
            self.detector = ChiSquareDriftDetector(**kwargs)
            self.threshold = float(self.detector.alpha)
            self.comparator = "<"
        elif method == "tvdcat":
            self.detector = TotalVariationDriftDetector(**kwargs)
            self.threshold = float(self.detector.threshold)
            self.comparator = ">"
        elif method == "datetime":
            self.detector = DateTimeDriftDetector(**kwargs)
            self.threshold = float(self.detector.threshold)
            self.comparator = ">"
        else:
            raise ValueError(
                "method must be one of: psi, ks, cvm, js, wasserstein, mmd, c2st, energy, chi2cat, tvdcat, datetime"
            )

    @staticmethod
    def recommended_min_samples(method: str) -> int:
        """Recommended minimum samples per side for stable estimates."""
        rec = {
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
        return rec.get(method, 20)

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

    def _bootstrap_ci(self, reference_data: Any, current_data: Any) -> tuple[float, float] | None:
        if self.ci_bootstrap_samples <= 0:
            return None
        if self.method not in {"psi", "js", "wasserstein"}:
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
                score = float(cast(Any, self.detector).calculate_psi(ref_s, cur_s))
            elif self.method == "js":
                score = float(cast(Any, self.detector).calculate_js_distance(ref_s, cur_s))
            else:
                _, score = cast(_SimpleDetector, self.detector).detect_drift(ref_s, cur_s)
                score = float(score)
            stats.append(score)

        alpha = 1.0 - self.ci_level
        lo = float(np.quantile(stats, alpha / 2.0))
        hi = float(np.quantile(stats, 1.0 - alpha / 2.0))
        return lo, hi

    def detect_drift(self, reference_data: Any, current_data: Any) -> DriftResult:
        started = time.perf_counter()
        reference_data = self._maybe_convert_columnar(reference_data)
        current_data = self._maybe_convert_columnar(current_data)
        span_ctx = nullcontext()
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
                if self.method == "wasserstein":
                    details = cast(_DetailedDetector, self.detector).detect_drift(
                        reference_data, current_data, return_details=True
                    )
                    metadata_ws: dict[str, Any] = {"calibrated_threshold": float(details.threshold)}
                    ci_ws = self._bootstrap_ci(reference_data, current_data)
                    if ci_ws is not None:
                        metadata_ws["score_ci"] = {"lo": ci_ws[0], "hi": ci_ws[1], "level": self.ci_level}
                    result = DriftResult(
                        method=self.method,
                        drift=bool(details.drift_detected),
                        score=float(details.distance),
                        p_value=float(details.p_value),
                        threshold=self.threshold,
                        comparator=self.comparator,
                        metadata=metadata_ws,
                    )
                elif self.method == "mmd":
                    details = cast(_DetailedDetector, self.detector).detect_drift(
                        reference_data, current_data, return_details=True
                    )
                    result = DriftResult(
                        method=self.method,
                        drift=bool(details.drift_detected),
                        score=float(details.mmd2),
                        p_value=float(details.p_value),
                        threshold=self.threshold,
                        comparator=self.comparator,
                        metadata={"calibrated_threshold": float(details.threshold)},
                    )
                elif self.method in {"c2st", "energy"}:
                    details = cast(_DetailedDetector, self.detector).detect_drift(
                        reference_data, current_data, return_details=True
                    )
                    score = float(details.roc_auc) if self.method == "c2st" else float(details.energy_distance)
                    result = DriftResult(
                        method=self.method,
                        drift=bool(details.drift_detected),
                        score=score,
                        p_value=float(details.p_value),
                        threshold=self.threshold,
                        comparator=self.comparator,
                        metadata={"calibrated_threshold": float(details.threshold)},
                    )
                else:
                    drift, score = cast(_SimpleDetector, self.detector).detect_drift(
                        reference_data, current_data
                    )
                    p_value = float(score) if self.method in {"ks", "cvm", "chi2cat"} else None
                    metadata_simple: dict[str, Any] = {}
                    backend = getattr(self.detector, "last_backend", None)
                    if isinstance(backend, str):
                        metadata_simple["execution_backend"] = backend
                    ci_simple = self._bootstrap_ci(reference_data, current_data)
                    if ci_simple is not None:
                        metadata_simple["score_ci"] = {
                            "lo": ci_simple[0],
                            "hi": ci_simple[1],
                            "level": self.ci_level,
                        }
                    result = DriftResult(
                        method=self.method,
                        drift=bool(drift),
                        score=float(score),
                        p_value=p_value,
                        threshold=self.threshold,
                        comparator=self.comparator,
                        metadata=metadata_simple,
                    )
                if self.telemetry is not None:
                    self.telemetry.record_drift_rate(
                        1.0 if result.drift else 0.0,
                        {"component": "detector", "method": self.method},
                    )
                return result
            except Exception:
                if self.telemetry is not None:
                    self.telemetry.record_error({"component": "detector", "method": self.method})
                raise
            finally:
                if self.telemetry is not None:
                    self.telemetry.record_latency(
                        (time.perf_counter() - started) * 1000.0,
                        {"component": "detector", "method": self.method},
                    )
