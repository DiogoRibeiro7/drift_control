import numpy as np


class PSIDriftDetector:
    """Detect drift using the Population Stability Index (PSI)."""

    def __init__(self, threshold: float = 0.2, bins: int = 10, strategy: str = "quantile") -> None:
        """Create detector with a PSI threshold, bins and binning strategy.

        :param threshold: PSI value above which drift is flagged.
        :param bins: Number of bins to divide the data.
        :param strategy: ``"quantile"`` for quantile-based bins or ``"uniform"`` for
            equal-width bins.
        """

        if strategy not in {"quantile", "uniform"}:
            raise ValueError("strategy must be 'quantile' or 'uniform'")

        self.threshold = threshold
        self.bins = bins
        self.strategy = strategy

    def _bin_edges(self, ref: np.ndarray, cur: np.ndarray) -> np.ndarray:
        """Return the bin edges according to the chosen strategy."""
        combined = np.concatenate([ref, cur])
        if self.strategy == "quantile":
            return np.quantile(combined, np.linspace(0, 1, self.bins + 1))
        return np.linspace(combined.min(), combined.max(), self.bins + 1)

    def calculate_psi(self, reference, current) -> float:
        """Compute PSI between reference and current arrays."""
        ref = np.asarray(reference, dtype=float)
        cur = np.asarray(current, dtype=float)
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
