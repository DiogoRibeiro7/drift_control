"""Async streaming drift monitors.

``StreamMonitor`` is the core engine: hand it a baseline DataFrame and a
detector, then feed an async iterable of batches. The Kafka and RabbitMQ
classes are thin adapters that decode broker messages into DataFrames and
delegate to the same engine.
"""

from io import StringIO
import inspect
from typing import AsyncIterable, Dict, Any, Callable
import pandas as pd

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
        on_drift: Callable[[Dict[str, Dict[str, Any]]], Any] | None = None,
        on_schema_change: str = "strict",
    ) -> None:
        if on_schema_change not in {"strict", "ignore", "drop"}:
            raise ValueError("on_schema_change must be one of: strict, ignore, drop")
        self.detector = detector or PSIDriftDetector()
        self.baseline: pd.DataFrame | None = None
        self.on_drift = on_drift
        self.on_schema_change = on_schema_change

    def set_baseline(self, data: pd.DataFrame) -> None:
        """Store the baseline used for drift comparison."""
        self.baseline = data.copy()

    async def compare(self, batch: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Score a single batch against the baseline."""
        if self.baseline is None:
            raise ValueError("Baseline not set")
        extra = [c for c in batch.columns if c not in self.baseline.columns]
        if extra and self.on_schema_change == "strict":
            raise ValueError(f"Batch contains unknown columns: {extra}")
        missing = [c for c in self.baseline.columns if c not in batch.columns]
        if missing:
            raise ValueError(f"Batch is missing baseline columns: {missing}")
        if self.on_schema_change == "drop":
            batch = batch.loc[:, self.baseline.columns]
        results: Dict[str, Dict[str, Any]] = {}
        for col in self.baseline.columns:
            drift, score = self.detector.detect_drift(
                self.baseline[col], batch[col]
            )
            results[col] = {"drift": drift, "score": score}
        return results

    # Backwards-compatible private alias.
    _compare = compare

    async def monitor(
        self, stream: AsyncIterable[pd.DataFrame]
    ) -> AsyncIterable[Dict[str, Dict[str, Any]]]:
        """Yield drift results as new batches arrive."""
        if self.baseline is None:
            raise ValueError("Baseline not set")
        async for batch in stream:
            result = await self.compare(batch)
            if self.on_drift is not None and any(
                bool(col_result.get("drift")) for col_result in result.values()
            ):
                callback_out = self.on_drift(result)
                if inspect.isawaitable(callback_out):
                    await callback_out
            yield result


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
        on_drift: Callable[[Dict[str, Dict[str, Any]]], Any] | None = None,
        on_schema_change: str = "strict",
    ) -> None:
        try:
            from aiokafka import AIOKafkaConsumer
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError("aiokafka is required for KafkaStreamMonitor") from exc
        self._monitor = StreamMonitor(
            detector,
            on_drift=on_drift,
            on_schema_change=on_schema_change,
        )
        self._consumer_factory = lambda: AIOKafkaConsumer(
            topic, bootstrap_servers=bootstrap_servers
        )
        self._consumer = None

    def set_baseline(self, data: pd.DataFrame) -> None:
        """Store the baseline used for drift comparison."""
        self._monitor.set_baseline(data)

    async def monitor(self) -> AsyncIterable[Dict[str, Dict[str, Any]]]:
        """Yield drift results for each Kafka message."""
        if self._consumer is None:
            self._consumer = self._consumer_factory()
        await self._consumer.start()
        try:
            async for msg in self._consumer:
                batch = _read_json_frame(msg.value.decode())
                yield await self._monitor.compare(batch)
        finally:
            await self._consumer.stop()


class RabbitMQStreamMonitor:
    """Consume messages from a RabbitMQ queue and score each as a batch.

    Each message body must be a JSON-serialised DataFrame matching the
    baseline schema. The queue must already exist (passive declare).
    """

    def __init__(
        self,
        queue: str,
        url: str = "amqp://localhost/",
        detector: Any | None = None,
        on_drift: Callable[[Dict[str, Dict[str, Any]]], Any] | None = None,
        on_schema_change: str = "strict",
    ) -> None:
        try:
            import aio_pika
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError("aio_pika is required for RabbitMQStreamMonitor") from exc
        self._aio_pika = aio_pika
        self._monitor = StreamMonitor(
            detector,
            on_drift=on_drift,
            on_schema_change=on_schema_change,
        )
        self.queue_name = queue
        self.url = url

    def set_baseline(self, data: pd.DataFrame) -> None:
        """Store the baseline used for drift comparison."""
        self._monitor.set_baseline(data)

    async def monitor(self) -> AsyncIterable[Dict[str, Dict[str, Any]]]:
        """Yield drift results for each RabbitMQ message."""
        connection = await self._aio_pika.connect_robust(self.url)
        async with connection:
            channel = await connection.channel()
            queue = await channel.declare_queue(self.queue_name, passive=True)
            async with queue.iterator() as queue_iter:
                async for message in queue_iter:
                    async with message.process():
                        batch = _read_json_frame(message.body.decode())
                        yield await self._monitor.compare(batch)
