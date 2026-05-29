from __future__ import annotations

import asyncio
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .telemetry import DriftTelemetry
from .unified_drift_detector import UnifiedDriftDetector
from .stream_monitor import StreamMonitor
from .psi_drift_detector import PSIDriftDetector


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

    @staticmethod
    def scenario_no_drift(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray, bool]:
        ref = rng.normal(loc=0.0, scale=1.0, size=n)
        cur = rng.normal(loc=0.0, scale=1.0, size=n)
        return ref, cur, False

    @staticmethod
    def scenario_mean_shift(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray, bool]:
        ref = rng.normal(loc=0.0, scale=1.0, size=n)
        cur = rng.normal(loc=0.8, scale=1.0, size=n)
        return ref, cur, True

    @staticmethod
    def scenario_variance_shift(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray, bool]:
        ref = rng.normal(loc=0.0, scale=1.0, size=n)
        cur = rng.normal(loc=0.0, scale=1.8, size=n)
        return ref, cur, True

    @staticmethod
    def scenario_tail_shift(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray, bool]:
        ref = rng.normal(loc=0.0, scale=1.0, size=n)
        cur = rng.standard_t(df=3, size=n)
        return ref, cur, True

    @staticmethod
    def scenario_cat_no_drift(
        rng: np.random.Generator, n: int
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        categories = np.array(["a", "b", "c"])
        ref = rng.choice(categories, size=n, p=[0.4, 0.4, 0.2])
        cur = rng.choice(categories, size=n, p=[0.4, 0.4, 0.2])
        return ref, cur, False

    @staticmethod
    def scenario_cat_shift(
        rng: np.random.Generator, n: int
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        categories = np.array(["a", "b", "c"])
        ref = rng.choice(categories, size=n, p=[0.5, 0.4, 0.1])
        cur = rng.choice(categories, size=n, p=[0.1, 0.3, 0.6])
        return ref, cur, True

    def _detector_for_method(self, method: str) -> UnifiedDriftDetector:
        if method in {"ks", "cvm"}:
            return UnifiedDriftDetector(method=method, alpha=0.05)
        if method in {"mmd", "wasserstein", "c2st", "energy"}:
            return UnifiedDriftDetector(method=method, alpha=0.05, n_permutations=80, random_state=7)
        if method == "chi2cat":
            return UnifiedDriftDetector(method=method, alpha=0.05)
        if method == "tvdcat":
            return UnifiedDriftDetector(method=method, threshold=0.1)
        return UnifiedDriftDetector(method=method, threshold=0.2 if method == "psi" else 0.1)

    def run(
        self,
        scenarios: dict[str, ScenarioFn] | None = None,
    ) -> list[BenchmarkResult]:
        scenarios = scenarios or {
            "no_drift": self.scenario_no_drift,
            "mean_shift": self.scenario_mean_shift,
            "variance_shift": self.scenario_variance_shift,
            "tail_shift": self.scenario_tail_shift,
        }
        categorical_scenarios = {
            "cat_no_drift": self.scenario_cat_no_drift,
            "cat_shift": self.scenario_cat_shift,
        }

        rng = np.random.default_rng(self.random_seed)
        out: list[BenchmarkResult] = []

        run_span = nullcontext()
        if self.telemetry is not None:
            start_span = getattr(self.telemetry, "start_span", None)
            if callable(start_span):
                run_span = start_span(
                    "drift_control.benchmark.run",
                    {"component": "benchmark", "n_trials": self.n_trials, "sample_size": self.sample_size},
                )
        with run_span:
            for method in self.methods:
                detector = self._detector_for_method(method)
                method_scenarios = (
                    categorical_scenarios
                    if method in {"chi2cat", "tvdcat"}
                    else scenarios
                )
                for scenario_name, scenario_fn in method_scenarios.items():
                    scenario_span = nullcontext()
                    if self.telemetry is not None:
                        start_span = getattr(self.telemetry, "start_span", None)
                        if callable(start_span):
                            scenario_span = start_span(
                                "drift_control.benchmark.scenario",
                                {"component": "benchmark", "method": method, "scenario": scenario_name},
                            )
                    with scenario_span:
                        drift_flags: list[bool] = []
                        scores: list[float] = []
                        latencies_ms: list[float] = []
                        expected: bool | None = None

                        for _ in range(self.n_trials):
                            ref, cur, exp = scenario_fn(rng, self.sample_size)
                            expected = exp if expected is None else expected
                            start = time.perf_counter()
                            try:
                                result = detector.detect_drift(ref, cur)
                            except Exception:
                                if self.telemetry is not None:
                                    self.telemetry.record_error(
                                        {
                                            "component": "benchmark",
                                            "method": method,
                                            "scenario": scenario_name,
                                        }
                                    )
                                raise
                            elapsed_ms = (time.perf_counter() - start) * 1000.0
                            drift_flags.append(bool(result.drift))
                            scores.append(float(result.score))
                            latencies_ms.append(float(elapsed_ms))
                            if self.telemetry is not None:
                                self.telemetry.record_latency(
                                    elapsed_ms,
                                    {
                                        "component": "benchmark",
                                        "method": method,
                                        "scenario": scenario_name,
                                    },
                                )

                        out.append(
                            BenchmarkResult(
                                method=method,
                                scenario=scenario_name,
                                n_trials=self.n_trials,
                                expected_drift=bool(expected),
                                drift_rate=float(np.mean(drift_flags)),
                                avg_score=float(np.mean(scores)),
                                avg_latency_ms=float(np.mean(latencies_ms)),
                            )
                        )
                        if self.telemetry is not None:
                            self.telemetry.record_drift_rate(
                                float(np.mean(drift_flags)),
                                {
                                    "component": "benchmark",
                                    "method": method,
                                    "scenario": scenario_name,
                                },
                            )

        return out

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
        out: list[LargeScaleParityResult] = []

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

            delta = float(abs(float(small.score) - float(large.score)))
            out.append(
                LargeScaleParityResult(
                    method=method,
                    small_n=small_n,
                    large_n=large_n,
                    small_score=float(small.score),
                    large_score=float(large.score),
                    abs_score_delta=delta,
                    same_drift_flag=bool(small.drift) == bool(large.drift),
                    large_latency_ms=float(large_latency_ms),
                )
            )

        for row in out:
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
        return out

    def run_streaming_smoke(
        self,
        n_batches: int = 120,
        batch_size: int = 64,
    ) -> StreamingSmokeResult:
        """Run a deterministic streaming smoke scenario with schema changes and callbacks."""
        if n_batches < 10:
            raise ValueError("n_batches must be >= 10")
        if batch_size < 8:
            raise ValueError("batch_size must be >= 8")

        import pandas as pd

        rng = np.random.default_rng(self.random_seed)
        baseline = pd.DataFrame(
            {
                "x": rng.normal(0.0, 1.0, size=256),
                "y": rng.normal(0.0, 1.0, size=256),
            }
        )
        callback_events = 0

        def _on_drift(_result: dict[str, dict[str, object]]) -> None:
            nonlocal callback_events
            callback_events += 1

        monitor = StreamMonitor(
            detector=PSIDriftDetector(strategy="kll", sketch_size=128, random_state=self.random_seed),
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

        async def _stream():
            for i in range(n_batches):
                shift = 0.0 if i < (n_batches // 3) else (1.0 if i < (2 * n_batches // 3) else 0.2)
                batch = pd.DataFrame(
                    {
                        "x": rng.normal(shift, 1.0, size=batch_size),
                        "y": rng.normal(0.0, 1.0, size=batch_size),
                    }
                )
                # Exercise schema-evolution path in drop mode.
                if i % 11 == 0:
                    batch["z"] = rng.normal(0.0, 1.0, size=batch_size)
                if i % 13 == 0:
                    batch = batch.drop(columns=["y"])
                yield batch

        async def _run() -> tuple[int, int]:
            batches_processed = 0
            drift_events = 0
            async for result in monitor.monitor(_stream()):
                batches_processed += 1
                if any(bool(v.get("drift")) for v in result.values()):
                    drift_events += 1
            return batches_processed, drift_events

        processed, drift_events = asyncio.run(_run())
        if monitor.baseline is None:
            raise AssertionError("Streaming smoke failed: baseline was unexpectedly cleared.")
        if processed <= 0:
            raise AssertionError("Streaming smoke failed: no batches processed.")
        if callback_events <= 0:
            raise AssertionError("Streaming smoke failed: no drift callbacks fired.")

        return StreamingSmokeResult(
            batches_processed=processed,
            drift_events=drift_events,
            callback_events=callback_events,
            final_baseline_rows=int(len(monitor.baseline)),
            final_columns=list(monitor.baseline.columns),
        )
