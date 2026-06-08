"""Async streaming drift monitors.

``StreamMonitor`` is the core engine: hand it a baseline DataFrame and a
detector, then feed an async iterable of batches. ``KafkaStreamMonitor`` is
a thin adapter that decodes broker messages into DataFrames and delegates to
the same engine.
"""

import inspect
from collections import deque
from collections.abc import AsyncIterable, Callable
from io import StringIO
from typing import Any

import pandas as pd

from ._interop import score_pair
from .alert_sinks import AlertSink
from .psi_drift_detector import PSIDriftDetector


def _read_json_frame(payload: str) -> pd.DataFrame:
    """Parse a JSON-serialised DataFrame, wrapping the literal in StringIO.

    Passing a raw JSON string to ``pd.read_json`` is deprecated and will be
    removed in a future pandas release.
    """
    return pd.read_json(StringIO(payload))


class StreamMonitor:
    """Asynchronously monitor drift for streaming data."""

    def __init__(
        self,
        detector: Any | None = None,
        on_drift: Callable[[dict[str, dict[str, Any]]], Any] | None = None,
        alert_sinks: list[AlertSink] | None = None,
        on_schema_change: str = "strict",
        window_size: int = 1,
        baseline_strategy: str = "fixed",
        sliding_window_batches: int = 3,
        ewma_alpha: float = 0.2,
        random_state: int = 42,
        adaptive_threshold: bool = False,
        threshold_quantile: float = 0.95,
        threshold_history: int = 100,
        min_threshold_samples: int = 20,
    ) -> None:
        if on_schema_change not in {"strict", "ignore", "drop"}:
            raise ValueError("on_schema_change must be one of: strict, ignore, drop")
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        if baseline_strategy not in {"fixed", "sliding", "ewma"}:
            raise ValueError("baseline_strategy must be one of: fixed, sliding, ewma")
        if sliding_window_batches < 1:
            raise ValueError("sliding_window_batches must be >= 1")
        if not (0 < ewma_alpha <= 1):
            raise ValueError("ewma_alpha must be in (0, 1]")
        if not (0 < threshold_quantile < 1):
            raise ValueError("threshold_quantile must be in (0, 1)")
        if threshold_history < 5:
            raise ValueError("threshold_history must be >= 5")
        if min_threshold_samples < 5:
            raise ValueError("min_threshold_samples must be >= 5")
        self.detector = detector or PSIDriftDetector()
        self.baseline: pd.DataFrame | None = None
        self.on_drift = on_drift
        self.alert_sinks = list(alert_sinks) if alert_sinks is not None else []
        self.on_schema_change = on_schema_change
        self.window_size = window_size
        self.baseline_strategy = baseline_strategy
        self.sliding_window_batches = sliding_window_batches
        self.ewma_alpha = ewma_alpha
        self.random_state = random_state
        self.adaptive_threshold = adaptive_threshold
        self.threshold_quantile = threshold_quantile
        self.threshold_history = threshold_history
        self.min_threshold_samples = min_threshold_samples
        self._rng_counter = 0
        self._recent_batches: deque[pd.DataFrame] = deque(maxlen=sliding_window_batches)
        self._score_history: dict[str, deque[float]] = {}

    def set_baseline(self, data: pd.DataFrame) -> None:
        """Store the baseline used for drift comparison."""
        self.baseline = data.copy()

    async def compare(self, batch: pd.DataFrame) -> dict[str, dict[str, Any]]:
        """Score a single batch against the baseline."""
        if self.baseline is None:
            raise ValueError("Baseline not set")
        batch = self._normalize_batch(batch)
        results: dict[str, dict[str, Any]] = {}
        for col in batch.columns:
            drift, score = score_pair(self.detector, self.baseline[col], batch[col])
            results[col] = {"drift": drift, "score": score}
        return results

    def _normalize_batch(self, batch: pd.DataFrame) -> pd.DataFrame:
        if self.baseline is None:
            raise ValueError("Baseline not set")
        extra = [c for c in batch.columns if c not in self.baseline.columns]
        missing = [c for c in self.baseline.columns if c not in batch.columns]
        if self.on_schema_change == "strict":
            if extra:
                raise ValueError(f"Batch contains unknown columns: {extra}")
            if missing:
                raise ValueError(f"Batch is missing baseline columns: {missing}")
            return batch.loc[:, self.baseline.columns]
        if self.on_schema_change == "ignore":
            if missing:
                raise ValueError(f"Batch is missing baseline columns: {missing}")
            return batch.loc[:, self.baseline.columns]
        shared = [c for c in self.baseline.columns if c in batch.columns]
        if not shared:
            raise ValueError("Batch has no shared columns with baseline.")
        return batch.loc[:, shared]

    def _update_baseline(self, batch: pd.DataFrame) -> None:
        if self.baseline is None:
            raise ValueError("Baseline not set")
        if self.on_schema_change == "drop":
            shared = [c for c in self.baseline.columns if c in batch.columns]
            if not shared:
                raise ValueError("Batch has no shared columns with baseline.")
            self.baseline = self.baseline.loc[:, shared].copy()
            batch = batch.loc[:, shared]
        if self.baseline_strategy == "fixed":
            return
        if self.baseline_strategy == "sliding":
            self._recent_batches.append(batch.copy())
            self.baseline = pd.concat(list(self._recent_batches), ignore_index=True)
            return
        n_total = len(self.baseline)
        if n_total == 0:
            self.baseline = batch.copy()
            return
        n_new = int(round(n_total * self.ewma_alpha))
        n_new = max(1, min(n_total, n_new))
        n_old = n_total - n_new
        rs = self.random_state + self._rng_counter
        self._rng_counter += 1
        old_part = self.baseline.sample(n=n_old, replace=(n_old > len(self.baseline)), random_state=rs)
        new_part = batch.sample(n=n_new, replace=(n_new > len(batch)), random_state=rs + 1)
        self.baseline = pd.concat([old_part, new_part], ignore_index=True)

    def _update_adaptive_thresholds(self, result: dict[str, dict[str, Any]]) -> None:
        if not self.adaptive_threshold:
            return
        detector_threshold = getattr(self.detector, "threshold", None)
        if not isinstance(detector_threshold, (int, float)):
            return
        for col, col_result in result.items():
            if bool(col_result.get("drift")):
                continue
            score = col_result.get("score")
            if not isinstance(score, (int, float)):
                continue
            history = self._score_history.get(col)
            if history is None:
                history = deque(maxlen=self.threshold_history)
                self._score_history[col] = history
            history.append(float(score))
            if len(history) >= self.min_threshold_samples:
                new_threshold = float(pd.Series(history, dtype=float).quantile(self.threshold_quantile))
                self.detector.threshold = new_threshold

    async def _process_batch(self, batch: pd.DataFrame) -> dict[str, dict[str, Any]]:
        norm_batch = self._normalize_batch(batch)
        result = await self.compare(norm_batch)
        self._update_adaptive_thresholds(result)
        has_drift = any(bool(col_result.get("drift")) for col_result in result.values())
        if self.on_drift is not None and has_drift:
            callback_out = self.on_drift(result)
            if inspect.isawaitable(callback_out):
                await callback_out
        if has_drift:
            for sink in self.alert_sinks:
                sink_out = sink.send(result)
                if inspect.isawaitable(sink_out):
                    await sink_out
        self._update_baseline(norm_batch)
        return result

    # Backwards-compatible private alias.
    _compare = compare

    async def monitor(
        self, stream: AsyncIterable[pd.DataFrame]
    ) -> AsyncIterable[dict[str, dict[str, Any]]]:
        """Yield drift results as new batches arrive."""
        if self.baseline is None:
            raise ValueError("Baseline not set")
        window: list[pd.DataFrame] = []
        async for batch in stream:
            window.append(batch)
            if len(window) >= self.window_size:
                merged = pd.concat(window, ignore_index=True)
                window.clear()
                yield await self._process_batch(merged)
        if window:
            merged = pd.concat(window, ignore_index=True)
            yield await self._process_batch(merged)


class KafkaStreamMonitor:
    """Consume messages from a Kafka topic and score each as a batch.

    Each message body must be a JSON-serialised DataFrame matching the
    baseline schema.
    """

    def __init__(
        self,
        topic: str,
        bootstrap_servers: str = "localhost:9092",
        detector: Any | None = None,
        on_drift: Callable[[dict[str, dict[str, Any]]], Any] | None = None,
        alert_sinks: list[AlertSink] | None = None,
        on_schema_change: str = "strict",
        window_size: int = 1,
        baseline_strategy: str = "fixed",
        sliding_window_batches: int = 3,
        ewma_alpha: float = 0.2,
        random_state: int = 42,
        adaptive_threshold: bool = False,
        threshold_quantile: float = 0.95,
        threshold_history: int = 100,
        min_threshold_samples: int = 20,
    ) -> None:
        try:
            from aiokafka import AIOKafkaConsumer
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError("aiokafka is required for KafkaStreamMonitor") from exc
        self._monitor = StreamMonitor(
            detector,
            on_drift=on_drift,
            alert_sinks=alert_sinks,
            on_schema_change=on_schema_change,
            window_size=window_size,
            baseline_strategy=baseline_strategy,
            sliding_window_batches=sliding_window_batches,
            ewma_alpha=ewma_alpha,
            random_state=random_state,
            adaptive_threshold=adaptive_threshold,
            threshold_quantile=threshold_quantile,
            threshold_history=threshold_history,
            min_threshold_samples=min_threshold_samples,
        )
        self._consumer_factory = lambda: AIOKafkaConsumer(
            topic, bootstrap_servers=bootstrap_servers
        )
        self._consumer = None

    def set_baseline(self, data: pd.DataFrame) -> None:
        """Store the baseline used for drift comparison."""
        self._monitor.set_baseline(data)

    async def monitor(self) -> AsyncIterable[dict[str, dict[str, Any]]]:
        """Yield drift results for each Kafka message."""
        if self._consumer is None:
            self._consumer = self._consumer_factory()
        await self._consumer.start()
        window: list[pd.DataFrame] = []
        try:
            async for msg in self._consumer:
                batch = _read_json_frame(msg.value.decode())
                window.append(batch)
                if len(window) >= self._monitor.window_size:
                    merged = pd.concat(window, ignore_index=True)
                    window.clear()
                    yield await self._monitor._process_batch(merged)
            if window:
                merged = pd.concat(window, ignore_index=True)
                yield await self._monitor._process_batch(merged)
        finally:
            await self._consumer.stop()

