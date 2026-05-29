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
        self.last_backend = "numpy"

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

    @staticmethod
    def _is_pyarrow_like(values) -> bool:
        mod = getattr(getattr(values, "__class__", None), "__module__", "")
        return isinstance(mod, str) and mod.startswith("pyarrow.")

    def _arrow_edges_uniform(self, ref_arr, cur_arr):
        import pyarrow as pa  # type: ignore
        import pyarrow.compute as pc  # type: ignore

        ref_mm = pc.min_max(ref_arr).as_py()
        cur_mm = pc.min_max(cur_arr).as_py()
        ref_min = float(ref_mm["min"])
        ref_max = float(ref_mm["max"])
        cur_min = float(cur_mm["min"])
        cur_max = float(cur_mm["max"])
        edges = np.linspace(ref_min, ref_max, self.bins + 1)
        edges = np.unique(edges)
        if edges.size < 2:
            edges = np.array([ref_min, ref_min + 1.0], dtype=float)
        edges[0] = min(edges[0], cur_min)
        edges[-1] = max(edges[-1], cur_max)
        if edges[-1] <= edges[0]:
            edges[-1] = edges[0] + 1.0
        return edges

    def _arrow_edges_quantile(self, ref_arr, cur_arr):
        import pyarrow.compute as pc  # type: ignore

        q = np.linspace(0, 1, self.bins + 1).tolist()
        q_vals = pc.quantile(ref_arr, q=q, interpolation="linear")
        edges = np.asarray(q_vals.to_pylist(), dtype=float)
        edges = np.unique(edges)
        if edges.size < 2:
            ref_mm = pc.min_max(ref_arr).as_py()
            ref_min = float(ref_mm["min"])
            edges = np.array([ref_min, ref_min + 1.0], dtype=float)
        ref_mm = pc.min_max(ref_arr).as_py()
        cur_mm = pc.min_max(cur_arr).as_py()
        lo = min(float(ref_mm["min"]), float(cur_mm["min"]))
        hi = max(float(ref_mm["max"]), float(cur_mm["max"]))
        edges[0] = lo
        edges[-1] = hi if hi > lo else lo + 1.0
        return edges

    @staticmethod
    def _arrow_hist_counts(arr, edges):
        import pyarrow as pa  # type: ignore
        import pyarrow.compute as pc  # type: ignore

        counts = np.zeros(max(len(edges) - 1, 1), dtype=float)
        for i in range(len(edges) - 1):
            left = float(edges[i])
            right = float(edges[i + 1])
            ge_left = pc.greater_equal(arr, pa.scalar(left))
            if i == len(edges) - 2:
                lt_right = pc.less_equal(arr, pa.scalar(right))
            else:
                lt_right = pc.less(arr, pa.scalar(right))
            in_bin = pc.and_(ge_left, lt_right)
            c = pc.sum(in_bin).as_py()
            counts[i] = float(c if c is not None else 0.0)
        return counts

    def calculate_js_distance(self, reference, current) -> float:
        from scipy.spatial.distance import jensenshannon

        if self._is_pyarrow_like(reference) or self._is_pyarrow_like(current):
            try:
                import pyarrow as pa  # type: ignore
                ref_arr = pa.array(reference, type=pa.float64())
                cur_arr = pa.array(current, type=pa.float64())
                if len(ref_arr) == 0 or len(cur_arr) == 0:
                    raise ValueError("reference and current must be non-empty")
                if self.strategy == "quantile":
                    edges = self._arrow_edges_quantile(ref_arr, cur_arr)
                else:
                    edges = self._arrow_edges_uniform(ref_arr, cur_arr)
                ref_counts = self._arrow_hist_counts(ref_arr, edges)
                cur_counts = self._arrow_hist_counts(cur_arr, edges)
                ref_p = ref_counts / max(ref_counts.sum(), 1.0)
                cur_p = cur_counts / max(cur_counts.sum(), 1.0)
                eps = 1e-12
                ref_p = np.where(ref_p == 0, eps, ref_p)
                cur_p = np.where(cur_p == 0, eps, cur_p)
                self.last_backend = "pyarrow"
                return float(jensenshannon(ref_p, cur_p))
            except RuntimeError:
                pass
            except ImportError:
                pass

        self.last_backend = "numpy"
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
