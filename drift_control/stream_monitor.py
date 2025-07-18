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

    async def monitor(self, stream: AsyncIterable[pd.DataFrame]) -> AsyncIterable[Dict[str, Dict[str, Any]]]:
        """Yield drift results as new batches arrive."""
        if self.baseline is None:
            raise ValueError("Baseline not set")
        async for batch in stream:
            results: Dict[str, Dict[str, Any]] = {}
            for col in self.baseline.columns:
                drift, score = self.detector.detect_drift(self.baseline[col], batch[col])
                results[col] = {"drift": drift, "score": score}
            yield results
