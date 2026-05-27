from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

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
        if method in {"mmd", "wasserstein", "c2st"}:
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

        for method in self.methods:
            detector = self._detector_for_method(method)
            method_scenarios = (
                categorical_scenarios
                if method in {"chi2cat", "tvdcat"}
                else scenarios
            )
            for scenario_name, scenario_fn in method_scenarios.items():
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
