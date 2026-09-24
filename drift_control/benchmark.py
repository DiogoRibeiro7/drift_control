from __future__ import annotations

import asyncio
import time
import tracemalloc
from collections.abc import AsyncIterable, Callable, Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any

import numpy as np

from .detectors import UnivariateDriftDetector
from .distances import bin_edges
from .stream_monitor import StreamMonitor
from .telemetry import DriftTelemetry
from .unified_drift_detector import UnifiedDriftDetector

ScenarioFn = Callable[[np.random.Generator, int], tuple[np.ndarray, np.ndarray, bool]]


@dataclass(frozen=True)
class BenchmarkResult:
    method: str
    scenario: str
    n_trials: int
    expected_drift: bool
    drift_rate: float
    avg_score: float
    avg_latency_ms: float


@dataclass(frozen=True)
class LargeScaleParityResult:
    method: str
    small_n: int
    large_n: int
    small_score: float
    large_score: float
    abs_score_delta: float
    same_drift_flag: bool
    large_latency_ms: float


@dataclass(frozen=True)
class StreamingSmokeResult:
    batches_processed: int
    drift_events: int
    callback_events: int
    final_baseline_rows: int
    final_columns: list[str]
    early_phase_drift_rate: float
    middle_phase_drift_rate: float
    late_phase_drift_rate: float


@dataclass(frozen=True)
class MassiveScaleBenchmarkResult:
    method: str
    effective_rows: int
    chunk_rows: int
    n_chunks: int
    reference_score: float
    massive_score: float
    abs_score_delta: float
    elapsed_seconds: float
    throughput_rows_per_second: float
    peak_memory_mb: float
    execution_backend: str
    within_tolerance: bool
    within_runtime_budget: bool
    within_memory_budget: bool


@dataclass(frozen=True)
class _ScenarioRunSummary:
    drift_rate: float
    avg_score: float
    avg_latency_ms: float
    expected_drift: bool


@dataclass(frozen=True)
class _MassiveScaleConfig:
    effective_rows: int
    chunk_rows: int
    small_n: int
    score_tolerance: float
    runtime_budget_seconds: float
    memory_budget_mb: float
    backend_preference: str
    random_seed_offset: int


def _scenario_no_drift(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray, bool]:
    ref = rng.normal(loc=0.0, scale=1.0, size=n)
    cur = rng.normal(loc=0.0, scale=1.0, size=n)
    return ref, cur, False


def _scenario_mean_shift(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray, bool]:
    ref = rng.normal(loc=0.0, scale=1.0, size=n)
    cur = rng.normal(loc=0.8, scale=1.0, size=n)
    return ref, cur, True


def _scenario_variance_shift(
    rng: np.random.Generator, n: int
) -> tuple[np.ndarray, np.ndarray, bool]:
    ref = rng.normal(loc=0.0, scale=1.0, size=n)
    cur = rng.normal(loc=0.0, scale=1.8, size=n)
    return ref, cur, True


def _scenario_tail_shift(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray, bool]:
    ref = rng.normal(loc=0.0, scale=1.0, size=n)
    cur = rng.standard_t(df=3, size=n)
    return ref, cur, True


def _scenario_cat_no_drift(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray, bool]:
    categories = np.array(["a", "b", "c"])
    ref = rng.choice(categories, size=n, p=[0.4, 0.4, 0.2])
    cur = rng.choice(categories, size=n, p=[0.4, 0.4, 0.2])
    return ref, cur, False


def _scenario_cat_shift(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray, bool]:
    categories = np.array(["a", "b", "c"])
    ref = rng.choice(categories, size=n, p=[0.5, 0.4, 0.1])
    cur = rng.choice(categories, size=n, p=[0.1, 0.3, 0.6])
    return ref, cur, True


def _default_numeric_scenarios() -> dict[str, ScenarioFn]:
    return {
        "no_drift": _scenario_no_drift,
        "mean_shift": _scenario_mean_shift,
        "variance_shift": _scenario_variance_shift,
        "tail_shift": _scenario_tail_shift,
    }


def _default_categorical_scenarios() -> dict[str, ScenarioFn]:
    return {
        "cat_no_drift": _scenario_cat_no_drift,
        "cat_shift": _scenario_cat_shift,
    }


def _detector_for_method(method: str) -> UnifiedDriftDetector:
    if method in {"ks", "cvm"}:
        return UnifiedDriftDetector(method=method, alpha=0.05)
    if method in {"mmd", "wasserstein", "c2st", "energy"}:
        return UnifiedDriftDetector(method=method, alpha=0.05, n_permutations=80, random_state=7)
    if method == "chi2cat":
        return UnifiedDriftDetector(method=method, alpha=0.05)
    if method == "tvdcat":
        return UnifiedDriftDetector(method=method, threshold=0.1)
    return UnifiedDriftDetector(method=method, threshold=0.2 if method == "psi" else 0.1)


def _selected_scenarios(
    method: str,
    scenarios: dict[str, ScenarioFn],
) -> dict[str, ScenarioFn]:
    if method in {"chi2cat", "tvdcat"}:
        return _default_categorical_scenarios()
    return scenarios


@contextmanager
def _telemetry_span(
    telemetry: DriftTelemetry | None,
    name: str,
    attributes: dict[str, object],
) -> Iterator[object]:
    span_ctx = nullcontext()
    if telemetry is not None:
        start_span = getattr(telemetry, "start_span", None)
        if callable(start_span):
            span_ctx = start_span(name, attributes)
    with span_ctx as span:
        yield span


def _record_benchmark_error(
    telemetry: DriftTelemetry | None,
    *,
    method: str,
    scenario_name: str,
) -> None:
    if telemetry is not None:
        telemetry.record_error(
            {"component": "benchmark", "method": method, "scenario": scenario_name}
        )


def _run_scenario_trials(
    *,
    detector: UnifiedDriftDetector,
    scenario_fn: ScenarioFn,
    rng: np.random.Generator,
    sample_size: int,
    n_trials: int,
    telemetry: DriftTelemetry | None,
    method: str,
    scenario_name: str,
) -> _ScenarioRunSummary:
    drift_flags: list[bool] = []
    scores: list[float] = []
    latencies_ms: list[float] = []
    expected: bool | None = None
    for _ in range(n_trials):
        ref, cur, exp = scenario_fn(rng, sample_size)
        expected = exp if expected is None else expected
        started = time.perf_counter()
        try:
            result = detector.detect_drift(ref, cur)
        except Exception:
            _record_benchmark_error(telemetry, method=method, scenario_name=scenario_name)
            raise
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        drift_flags.append(bool(result.drift))
        scores.append(float(result.score))
        latencies_ms.append(float(elapsed_ms))
        if telemetry is not None:
            telemetry.record_latency(
                elapsed_ms,
                {"component": "benchmark", "method": method, "scenario": scenario_name},
            )
    return _ScenarioRunSummary(
        drift_rate=float(np.mean(drift_flags)),
        avg_score=float(np.mean(scores)),
        avg_latency_ms=float(np.mean(latencies_ms)),
        expected_drift=bool(expected),
    )


def _validate_large_scale_parity(
    rows: list[LargeScaleParityResult],
    tolerance: float,
) -> None:
    for row in rows:
        if row.abs_score_delta > tolerance:
            raise AssertionError(
                f"Large-scale parity failed for {row.method}: "
                f"abs_score_delta={row.abs_score_delta:.6f} > tolerance={tolerance:.6f}"
            )
        if not row.same_drift_flag:
            raise AssertionError(
                f"Large-scale parity drift decision mismatch for {row.method}: "
                "small and large runs disagree."
            )


def _phase_drift_rates(drift_flags: list[bool]) -> tuple[float, float, float]:
    third = max(len(drift_flags) // 3, 1)
    early = drift_flags[:third]
    middle = drift_flags[third : 2 * third]
    late = drift_flags[2 * third :]
    early_rate = float(np.mean(early)) if early else 0.0
    middle_rate = float(np.mean(middle)) if middle else 0.0
    late_rate = float(np.mean(late)) if late else 0.0
    return early_rate, middle_rate, late_rate


def _validate_phase_behavior(
    early_rate: float,
    middle_rate: float,
    late_rate: float,
) -> None:
    tol = 0.2
    if middle_rate < early_rate - tol:
        raise AssertionError(
            "Streaming smoke failed: middle-phase drift rate should be >= early-phase rate."
        )
    if late_rate > middle_rate + tol:
        raise AssertionError(
            "Streaming smoke failed: late-phase drift rate should be <= middle-phase rate."
        )


def _build_massive_scale_config(
    *,
    effective_rows: int,
    chunk_rows: int,
    small_n: int,
    score_tolerance: float,
    runtime_budget_seconds: float,
    memory_budget_mb: float,
    backend_preference: str,
    random_seed_offset: int,
) -> _MassiveScaleConfig:
    if effective_rows <= 0:
        raise ValueError("effective_rows must be > 0")
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be > 0")
    if small_n < 1000:
        raise ValueError("small_n must be >= 1000")
    if score_tolerance <= 0:
        raise ValueError("score_tolerance must be > 0")
    if runtime_budget_seconds <= 0:
        raise ValueError("runtime_budget_seconds must be > 0")
    if memory_budget_mb <= 0:
        raise ValueError("memory_budget_mb must be > 0")
    if backend_preference not in {"numpy", "pyarrow_auto"}:
        raise ValueError("backend_preference must be one of: numpy, pyarrow_auto")
    return _MassiveScaleConfig(
        effective_rows=effective_rows,
        chunk_rows=chunk_rows,
        small_n=small_n,
        score_tolerance=score_tolerance,
        runtime_budget_seconds=runtime_budget_seconds,
        memory_budget_mb=memory_budget_mb,
        backend_preference=backend_preference,
        random_seed_offset=random_seed_offset,
    )


def _resolve_chunk_backend(
    backend_preference: str,
) -> tuple[str, Any | None, Any | None]:
    if backend_preference != "pyarrow_auto":
        return "numpy", None, None
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.compute as pc  # type: ignore

        return "pyarrow", pa, pc
    except Exception:
        return "numpy", None, None


def _arrow_chunk_hist(arr: np.ndarray, bins: np.ndarray, pa: Any, pc: Any) -> np.ndarray:
    out = np.zeros(max(len(bins) - 1, 1), dtype=float)
    arr_pa = pa.array(arr, type=pa.float64())
    for j in range(len(bins) - 1):
        left = float(bins[j])
        right = float(bins[j + 1])
        ge_left = pc.greater_equal(arr_pa, pa.scalar(left))
        if j == len(bins) - 2:
            lt_right = pc.less_equal(arr_pa, pa.scalar(right))
        else:
            lt_right = pc.less(arr_pa, pa.scalar(right))
        in_bin = pc.and_(ge_left, lt_right)
        count = pc.sum(in_bin).as_py()
        out[j] = float(count if count is not None else 0.0)
    return out


def _run_chunked_histograms(
    *,
    rng: np.random.Generator,
    effective_rows: int,
    chunk_rows: int,
    edges: np.ndarray,
    execution_backend: str,
    pa: Any | None,
    pc: Any | None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    n_chunks = int(np.ceil(effective_rows / chunk_rows))
    ref_counts = np.zeros(max(len(edges) - 1, 1), dtype=float)
    cur_counts = np.zeros(max(len(edges) - 1, 1), dtype=float)
    tracemalloc.start()
    started = time.perf_counter()
    for i in range(n_chunks):
        n_this = min(chunk_rows, effective_rows - i * chunk_rows)
        if n_this <= 0:
            break
        ref_chunk = rng.normal(0.0, 1.0, size=n_this)
        cur_chunk = rng.normal(0.35, 1.0, size=n_this)
        if execution_backend == "pyarrow":
            assert pa is not None and pc is not None
            ref_chunk_counts = _arrow_chunk_hist(ref_chunk, edges, pa, pc)
            cur_chunk_counts = _arrow_chunk_hist(cur_chunk, edges, pa, pc)
        else:
            ref_chunk_counts, _ = np.histogram(ref_chunk, bins=edges)
            cur_chunk_counts, _ = np.histogram(cur_chunk, bins=edges)
        ref_counts += ref_chunk_counts
        cur_counts += cur_chunk_counts
    elapsed = time.perf_counter() - started
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return ref_counts, cur_counts, elapsed, float(peak_bytes / (1024.0 * 1024.0))


def _psi_from_histograms(ref_counts: np.ndarray, cur_counts: np.ndarray) -> float:
    ref_perc = ref_counts / max(ref_counts.sum(), 1.0)
    cur_perc = cur_counts / max(cur_counts.sum(), 1.0)
    epsilon = 1e-6
    ref_perc = np.where(ref_perc == 0, epsilon, ref_perc)
    cur_perc = np.where(cur_perc == 0, epsilon, cur_perc)
    return float(np.sum((cur_perc - ref_perc) * np.log(cur_perc / ref_perc)))


class SyntheticDriftBenchmark:
    """Run repeatable detector benchmarks on synthetic drift scenarios."""

    def __init__(
        self,
        methods: list[str] | None = None,
        sample_size: int = 300,
        n_trials: int = 25,
        random_seed: int = 42,
        telemetry: DriftTelemetry | None = None,
    ) -> None:
        self.methods = methods or [
            "psi",
            "ks",
            "cvm",
            "js",
            "wasserstein",
            "mmd",
            "c2st",
            "energy",
            "chi2cat",
            "tvdcat",
        ]
        self.sample_size = sample_size
        self.n_trials = n_trials
        self.random_seed = random_seed
        self.telemetry = telemetry

    scenario_no_drift = staticmethod(_scenario_no_drift)
    scenario_mean_shift = staticmethod(_scenario_mean_shift)
    scenario_variance_shift = staticmethod(_scenario_variance_shift)
    scenario_tail_shift = staticmethod(_scenario_tail_shift)
    scenario_cat_no_drift = staticmethod(_scenario_cat_no_drift)
    scenario_cat_shift = staticmethod(_scenario_cat_shift)

    def _detector_for_method(self, method: str) -> UnifiedDriftDetector:
        return _detector_for_method(method)

    def run(
        self,
        scenarios: dict[str, ScenarioFn] | None = None,
    ) -> list[BenchmarkResult]:
        selected_scenarios = scenarios or _default_numeric_scenarios()
        rng = np.random.default_rng(self.random_seed)
        results: list[BenchmarkResult] = []
        with _telemetry_span(
            self.telemetry,
            "drift_control.benchmark.run",
            {"component": "benchmark", "n_trials": self.n_trials, "sample_size": self.sample_size},
        ):
            for method in self.methods:
                detector = self._detector_for_method(method)
                for scenario_name, scenario_fn in _selected_scenarios(
                    method, selected_scenarios
                ).items():
                    with _telemetry_span(
                        self.telemetry,
                        "drift_control.benchmark.scenario",
                        {"component": "benchmark", "method": method, "scenario": scenario_name},
                    ):
                        summary = _run_scenario_trials(
                            detector=detector,
                            scenario_fn=scenario_fn,
                            rng=rng,
                            sample_size=self.sample_size,
                            n_trials=self.n_trials,
                            telemetry=self.telemetry,
                            method=method,
                            scenario_name=scenario_name,
                        )
                        results.append(
                            BenchmarkResult(
                                method=method,
                                scenario=scenario_name,
                                n_trials=self.n_trials,
                                expected_drift=summary.expected_drift,
                                drift_rate=summary.drift_rate,
                                avg_score=summary.avg_score,
                                avg_latency_ms=summary.avg_latency_ms,
                            )
                        )
                        if self.telemetry is not None:
                            self.telemetry.record_drift_rate(
                                summary.drift_rate,
                                {
                                    "component": "benchmark",
                                    "method": method,
                                    "scenario": scenario_name,
                                },
                            )
        return results

    def run_large_scale_parity(
        self,
        methods: list[str] | None = None,
        small_n: int = 2_000,
        large_n: int = 20_000,
        tolerance: float = 0.12,
    ) -> list[LargeScaleParityResult]:
        """Compare small vs large sample scoring and assert parity within tolerance."""
        if small_n < 100 or large_n <= small_n:
            raise ValueError("small_n must be >= 100 and large_n must be > small_n")
        if tolerance <= 0:
            raise ValueError("tolerance must be > 0")
        selected = methods or ["psi", "ks", "js", "wasserstein"]
        rng = np.random.default_rng(self.random_seed)
        rows: list[LargeScaleParityResult] = []
        for method in selected:
            detector = self._detector_for_method(method)
            ref_small = rng.normal(0.0, 1.0, size=small_n)
            cur_small = rng.normal(0.35, 1.0, size=small_n)
            rep = int(np.ceil(large_n / small_n))
            ref_large = np.tile(ref_small, rep)[:large_n]
            cur_large = np.tile(cur_small, rep)[:large_n]
            small = detector.detect_drift(ref_small, cur_small)
            started = time.perf_counter()
            large = detector.detect_drift(ref_large, cur_large)
            large_latency_ms = (time.perf_counter() - started) * 1000.0
            rows.append(
                LargeScaleParityResult(
                    method=method,
                    small_n=small_n,
                    large_n=large_n,
                    small_score=float(small.score),
                    large_score=float(large.score),
                    abs_score_delta=float(abs(float(small.score) - float(large.score))),
                    same_drift_flag=bool(small.drift) == bool(large.drift),
                    large_latency_ms=float(large_latency_ms),
                )
            )
        _validate_large_scale_parity(rows, tolerance)
        return rows

    def run_streaming_smoke(
        self,
        n_batches: int = 120,
        batch_size: int = 64,
        assert_phase_behavior: bool = True,
    ) -> StreamingSmokeResult:
        """Run a deterministic streaming smoke scenario with schema changes and callbacks."""
        if n_batches < 10:
            raise ValueError("n_batches must be >= 10")
        if batch_size < 8:
            raise ValueError("batch_size must be >= 8")
        import pandas as pd

        rng = np.random.default_rng(self.random_seed)
        baseline = pd.DataFrame(
            {"x": rng.normal(0.0, 1.0, size=256), "y": rng.normal(0.0, 1.0, size=256)}
        )
        callback_events = 0

        def _on_drift(_result: dict[str, dict[str, object]]) -> None:
            nonlocal callback_events
            callback_events += 1

        monitor = StreamMonitor(
            detector=UnivariateDriftDetector(method="psi", threshold=0.2),
            on_drift=_on_drift,
            on_schema_change="drop",
            window_size=4,
            baseline_strategy="ewma",
            ewma_alpha=0.25,
            adaptive_threshold=True,
            threshold_history=64,
            min_threshold_samples=10,
        )
        monitor.set_baseline(baseline)

        async def _stream() -> AsyncIterable[pd.DataFrame]:
            for i in range(n_batches):
                shift = 0.0 if i < (n_batches // 3) else (1.0 if i < (2 * n_batches // 3) else 0.2)
                batch = pd.DataFrame(
                    {
                        "x": rng.normal(shift, 1.0, size=batch_size),
                        "y": rng.normal(0.0, 1.0, size=batch_size),
                    }
                )
                if i % 11 == 0:
                    batch["z"] = rng.normal(0.0, 1.0, size=batch_size)
                if i % 13 == 0:
                    batch = batch.drop(columns=["y"])
                yield batch

        async def _run() -> tuple[int, int, list[bool]]:
            batches_processed = 0
            drift_events = 0
            drift_flags: list[bool] = []
            async for result in monitor.monitor(_stream()):
                batches_processed += 1
                has_drift = any(bool(v.get("drift")) for v in result.values())
                drift_flags.append(has_drift)
                if has_drift:
                    drift_events += 1
            return batches_processed, drift_events, drift_flags

        processed, drift_events, drift_flags = asyncio.run(_run())
        if monitor.baseline is None:
            raise AssertionError("Streaming smoke failed: baseline was unexpectedly cleared.")
        if processed <= 0:
            raise AssertionError("Streaming smoke failed: no batches processed.")
        if callback_events <= 0:
            raise AssertionError("Streaming smoke failed: no drift callbacks fired.")
        if len(drift_flags) < 6:
            raise AssertionError("Streaming smoke failed: insufficient batches for phase checks.")
        early_rate, middle_rate, late_rate = _phase_drift_rates(drift_flags)
        if assert_phase_behavior:
            _validate_phase_behavior(early_rate, middle_rate, late_rate)
        return StreamingSmokeResult(
            batches_processed=processed,
            drift_events=drift_events,
            callback_events=callback_events,
            final_baseline_rows=int(len(monitor.baseline)),
            final_columns=list(monitor.baseline.columns),
            early_phase_drift_rate=early_rate,
            middle_phase_drift_rate=middle_rate,
            late_phase_drift_rate=late_rate,
        )

    def run_massive_scale_benchmark(
        self,
        effective_rows: int = 100_000_000,
        chunk_rows: int = 1_000_000,
        small_n: int = 50_000,
        score_tolerance: float = 0.2,
        runtime_budget_seconds: float = 240.0,
        memory_budget_mb: float = 2048.0,
        backend_preference: str = "numpy",
        random_seed_offset: int = 0,
    ) -> MassiveScaleBenchmarkResult:
        """Run a chunked large-row benchmark with parity and runtime assertions.

        This uses chunk synthesis so the benchmark can represent 100M effective
        rows without materializing all rows in memory at once.
        """
        config = _build_massive_scale_config(
            effective_rows=effective_rows,
            chunk_rows=chunk_rows,
            small_n=small_n,
            score_tolerance=score_tolerance,
            runtime_budget_seconds=runtime_budget_seconds,
            memory_budget_mb=memory_budget_mb,
            backend_preference=backend_preference,
            random_seed_offset=random_seed_offset,
        )
        method = "psi"
        rng = np.random.default_rng(self.random_seed + config.random_seed_offset)
        detector = self._detector_for_method(method)
        ref_small = rng.normal(0.0, 1.0, size=config.small_n)
        cur_small = rng.normal(0.35, 1.0, size=config.small_n)
        ref_res = detector.detect_drift(ref_small, cur_small)
        edges = bin_edges(np.asarray(ref_small, dtype=float), bins=detector.bins).copy()
        edges[0] = min(float(edges[0]), float(np.min(cur_small)))
        edges[-1] = max(float(edges[-1]), float(np.max(cur_small)))
        execution_backend, pa, pc = _resolve_chunk_backend(config.backend_preference)
        ref_counts, cur_counts, elapsed, peak_memory_mb = _run_chunked_histograms(
            rng=rng,
            effective_rows=config.effective_rows,
            chunk_rows=config.chunk_rows,
            edges=edges,
            execution_backend=execution_backend,
            pa=pa,
            pc=pc,
        )
        massive_score = _psi_from_histograms(ref_counts, cur_counts)
        massive_res_drift = bool(massive_score > detector.threshold)
        delta = float(abs(float(ref_res.score) - massive_score))
        throughput = float(config.effective_rows / max(elapsed, 1e-9))
        n_chunks = int(np.ceil(config.effective_rows / config.chunk_rows))
        within_tol = bool(delta <= config.score_tolerance) and (
            bool(ref_res.drift) == massive_res_drift
        )
        return MassiveScaleBenchmarkResult(
            method=method,
            effective_rows=config.effective_rows,
            chunk_rows=config.chunk_rows,
            n_chunks=n_chunks,
            reference_score=float(ref_res.score),
            massive_score=massive_score,
            abs_score_delta=delta,
            elapsed_seconds=float(elapsed),
            throughput_rows_per_second=throughput,
            peak_memory_mb=peak_memory_mb,
            execution_backend=execution_backend,
            within_tolerance=within_tol,
            within_runtime_budget=bool(elapsed <= config.runtime_budget_seconds),
            within_memory_budget=bool(peak_memory_mb <= config.memory_budget_mb),
        )
