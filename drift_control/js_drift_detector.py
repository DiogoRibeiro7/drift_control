import numpy as np
from .result_schema import DriftResult


class JensenShannonDriftDetector:
    """Detect drift using Jensen-Shannon distance between binned distributions."""

    def __init__(self, threshold: float = 0.1, bins: int = 20, strategy: str = "quantile") -> None:
        if threshold < 0:
            raise ValueError("threshold must be non-negative")
        if bins < 2:
            raise ValueError("bins must be >= 2")
        if strategy not in {"quantile", "uniform"}:
            raise ValueError("strategy must be 'quantile' or 'uniform'")
        self.threshold = threshold
        self.bins = bins
        self.strategy = strategy

    def _bin_edges(self, ref: np.ndarray, cur: np.ndarray) -> np.ndarray:
        if self.strategy == "quantile":
            edges = np.quantile(ref, np.linspace(0, 1, self.bins + 1))
        else:
            edges = np.linspace(ref.min(), ref.max(), self.bins + 1)
        edges = np.unique(edges)
        if edges.size < 2:
            edges = np.array([ref.min(), ref.min() + 1.0])
        edges[0] = min(edges[0], cur.min())
        edges[-1] = max(edges[-1], cur.max())
        return edges

    def calculate_js_distance(self, reference, current) -> float:
        from scipy.spatial.distance import jensenshannon

        ref = np.asarray(reference, dtype=float).ravel()
        cur = np.asarray(current, dtype=float).ravel()
        if ref.size == 0 or cur.size == 0:
            raise ValueError("reference and current must be non-empty")

        edges = self._bin_edges(ref, cur)
        ref_counts, _ = np.histogram(ref, bins=edges)
        cur_counts, _ = np.histogram(cur, bins=edges)

        ref_p = ref_counts / max(ref_counts.sum(), 1)
        cur_p = cur_counts / max(cur_counts.sum(), 1)

        eps = 1e-12
        ref_p = np.where(ref_p == 0, eps, ref_p)
        cur_p = np.where(cur_p == 0, eps, cur_p)
        return float(jensenshannon(ref_p, cur_p))

    def detect_drift(self, reference, current):
        js_distance = self.calculate_js_distance(reference, current)
        return js_distance > self.threshold, js_distance

    def detect_drift_result(self, reference, current) -> DriftResult:
        drift, js_distance = self.detect_drift(reference, current)
        return DriftResult(
            method="js",
            drift=bool(drift),
            score=float(js_distance),
            p_value=None,
            threshold=float(self.threshold),
            comparator=">",
            metadata={},
        )
