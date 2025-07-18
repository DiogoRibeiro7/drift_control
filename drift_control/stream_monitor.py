import asyncio
from typing import AsyncIterable, Dict, Any
import pandas as pd

from .psi_drift_detector import PSIDriftDetector


class StreamMonitor:
    """Asynchronously monitor drift for streaming data."""

    def __init__(self, detector: Any | None = None) -> None:
        self.detector = detector or PSIDriftDetector()
        self.baseline: pd.DataFrame | None = None

    def set_baseline(self, data: pd.DataFrame) -> None:
        """Store the baseline used for drift comparison."""
        self.baseline = data.copy()

    async def _compare(self, batch: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        if self.baseline is None:
            raise ValueError("Baseline not set")
        results: Dict[str, Dict[str, Any]] = {}
        for col in self.baseline.columns:
            drift, score = self.detector.detect_drift(self.baseline[col], batch[col])
            results[col] = {"drift": drift, "score": score}
        return results

    async def monitor(self, stream: AsyncIterable[pd.DataFrame]) -> AsyncIterable[Dict[str, Dict[str, Any]]]:
        """Yield drift results as new batches arrive."""
        if self.baseline is None:
            raise ValueError("Baseline not set")
        async for batch in stream:
            yield await self._compare(batch)


class KafkaStreamMonitor(StreamMonitor):
    """Consume messages from Kafka topics and monitor drift."""

    def __init__(self, topic: str, bootstrap_servers: str = "localhost:9092", detector: Any | None = None) -> None:
        super().__init__(detector)
        try:
            from aiokafka import AIOKafkaConsumer
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError("aiokafka is required for KafkaStreamMonitor") from exc
        self.consumer = AIOKafkaConsumer(topic, bootstrap_servers=bootstrap_servers)

    async def monitor(self) -> AsyncIterable[Dict[str, Dict[str, Any]]]:
        await self.consumer.start()
        try:
            async for msg in self.consumer:
                batch = pd.read_json(msg.value.decode())
                yield await self._compare(batch)
        finally:
            await self.consumer.stop()


class RabbitMQStreamMonitor(StreamMonitor):
    """Consume messages from RabbitMQ queues and monitor drift."""

    def __init__(self, queue: str, url: str = "amqp://localhost/", detector: Any | None = None) -> None:
        super().__init__(detector)
        try:
            import aio_pika
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError("aio_pika is required for RabbitMQStreamMonitor") from exc
        self.queue_name = queue
        self.url = url
        self._aio_pika = aio_pika

    async def monitor(self) -> AsyncIterable[Dict[str, Dict[str, Any]]]:
        connection = await self._aio_pika.connect_robust(self.url)
        async with connection:
            channel = await connection.channel()
            queue = await channel.declare_queue(self.queue_name, passive=True)
            async with queue.iterator() as queue_iter:
                async for message in queue_iter:
                    async with message.process():
                        batch = pd.read_json(message.body.decode())
                        yield await self._compare(batch)
