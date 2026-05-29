import numpy as np
from .result_schema import DriftResult
from .quantile_sketch import KLLSketch


class PSIDriftDetector:
    """Detect drift using the Population Stability Index (PSI)."""

    def __init__(
        self,
        threshold: float = 0.2,
        bins: int = 10,
        strategy: str = "quantile",
        sketch_size: int = 200,
        random_state: int = 42,
    ) -> None:
        """Create detector with a PSI threshold, bins and binning strategy.

        :param threshold: PSI value above which drift is flagged.
        :param bins: Number of bins to divide the data.
        :param strategy: ``"quantile"`` for quantile-based bins or ``"uniform"`` for
            equal-width bins.
        """

        if strategy not in {"quantile", "uniform", "kll"}:
            raise ValueError("strategy must be 'quantile', 'uniform' or 'kll'")

        self.threshold = threshold
        self.bins = bins
        self.strategy = strategy
        self.sketch_size = sketch_size
        self.random_state = random_state
        self._reference_sketch: KLLSketch | None = None

    def _bin_edges(self, ref: np.ndarray, cur: np.ndarray) -> np.ndarray:
        """Return bin edges fitted on the reference distribution only.

        Edges are widened to cover the current array so out-of-range
        observations are still binned rather than dropped.
        """
        if self.strategy == "quantile":
            edges = np.quantile(ref, np.linspace(0, 1, self.bins + 1))
        elif self.strategy == "kll":
            sketch = self._reference_sketch
            if sketch is None:
                sketch = KLLSketch(k=self.sketch_size, random_state=self.random_state)
                sketch.update(ref)
            edges = sketch.quantiles(np.linspace(0, 1, self.bins + 1))
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

    def _calculate_psi_pyarrow_uniform(self, reference, current) -> float:
        """Compute PSI with a PyArrow-native uniform-binning path."""
        try:
            import pyarrow as pa  # type: ignore
            import pyarrow.compute as pc  # type: ignore
        except Exception:
            # Optional dependency unavailable; caller should fallback.
            raise RuntimeError("pyarrow unavailable")

        ref_arr = pa.array(reference, type=pa.float64())
        cur_arr = pa.array(current, type=pa.float64())
        if len(ref_arr) == 0 or len(cur_arr) == 0:
            raise ValueError("reference and current must be non-empty")

        ref_mm = pc.min_max(ref_arr).as_py()
        cur_mm = pc.min_max(cur_arr).as_py()
        ref_min = float(ref_mm["min"])
        ref_max = float(ref_mm["max"])
        cur_min = float(cur_mm["min"])
        cur_max = float(cur_mm["max"])

        lo = min(ref_min, cur_min)
        hi = max(ref_max, cur_max)
        if hi <= lo:
            hi = lo + 1.0

        edges = np.linspace(ref_min, ref_max, self.bins + 1)
        edges = np.unique(edges)
        if edges.size < 2:
            edges = np.array([ref_min, ref_min + 1.0], dtype=float)
        edges[0] = lo
        edges[-1] = hi

        def _count_bins(arr):
            counts = np.zeros(max(edges.size - 1, 1), dtype=float)
            for i in range(edges.size - 1):
                left = float(edges[i])
                right = float(edges[i + 1])
                ge_left = pc.greater_equal(arr, pa.scalar(left))
                if i == edges.size - 2:
                    lt_right = pc.less_equal(arr, pa.scalar(right))
                else:
                    lt_right = pc.less(arr, pa.scalar(right))
                in_bin = pc.and_(ge_left, lt_right)
                c = pc.sum(in_bin).as_py()
                counts[i] = float(c if c is not None else 0.0)
            return counts

        ref_counts = _count_bins(ref_arr)
        cur_counts = _count_bins(cur_arr)
        ref_perc = ref_counts / max(ref_counts.sum(), 1.0)
        cur_perc = cur_counts / max(cur_counts.sum(), 1.0)
        epsilon = 1e-6
        ref_perc = np.where(ref_perc == 0, epsilon, ref_perc)
        cur_perc = np.where(cur_perc == 0, epsilon, cur_perc)
        psi = float(np.sum((cur_perc - ref_perc) * np.log(cur_perc / ref_perc)))
        return psi

    def fit_reference(self, reference) -> "PSIDriftDetector":
        """Fit a reusable reference sketch for online PSI binning."""
        ref = np.asarray(reference, dtype=float).ravel()
        if ref.size == 0:
            raise ValueError("reference must be non-empty")
        sketch = KLLSketch(k=self.sketch_size, random_state=self.random_state)
        sketch.update(ref)
        self._reference_sketch = sketch
        return self

    def update_reference(self, reference_chunk) -> "PSIDriftDetector":
        """Incrementally update the reference sketch using a new chunk."""
        ref = np.asarray(reference_chunk, dtype=float).ravel()
        if ref.size == 0:
            return self
        if self._reference_sketch is None:
            self._reference_sketch = KLLSketch(k=self.sketch_size, random_state=self.random_state)
        self._reference_sketch.update(ref)
        return self

    def calculate_psi(self, reference, current) -> float:
        """Compute PSI between reference and current arrays."""
        if self.strategy == "uniform" and (
            self._is_pyarrow_like(reference) or self._is_pyarrow_like(current)
        ):
            try:
                return self._calculate_psi_pyarrow_uniform(reference, current)
            except RuntimeError:
                pass
        ref = np.asarray(reference, dtype=float).ravel()
        cur = np.asarray(current, dtype=float).ravel()
        if ref.size == 0 or cur.size == 0:
            raise ValueError("reference and current must be non-empty")
        edges = self._bin_edges(ref, cur)
        ref_counts, _ = np.histogram(ref, bins=edges)
        cur_counts, _ = np.histogram(cur, bins=edges)
        ref_perc = ref_counts / max(ref_counts.sum(), 1)
        cur_perc = cur_counts / max(cur_counts.sum(), 1)
        epsilon = 1e-6
        ref_perc = np.where(ref_perc == 0, epsilon, ref_perc)
        cur_perc = np.where(cur_perc == 0, epsilon, cur_perc)
        psi = float(np.sum((cur_perc - ref_perc) * np.log(cur_perc / ref_perc)))
        return psi

    def detect_drift(self, reference, current):
        """Return whether drift is detected and the PSI value."""
        psi = self.calculate_psi(reference, current)
        return psi > self.threshold, psi

    def detect_drift_result(self, reference, current) -> DriftResult:
        drift, psi = self.detect_drift(reference, current)
        return DriftResult(
            method="psi",
            drift=bool(drift),
            score=float(psi),
            p_value=None,
            threshold=float(self.threshold),
            comparator=">",
            metadata={},
        )
