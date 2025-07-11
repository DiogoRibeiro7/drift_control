import numpy as np

class PSIDriftDetector:
    """Detect drift using the Population Stability Index (PSI)."""

    def __init__(self, threshold: float = 0.2, bins: int = 10) -> None:
        """Create detector with a PSI threshold and number of bins."""
        self.threshold = threshold
        self.bins = bins

    def calculate_psi(self, reference, current) -> float:
        """Compute PSI between reference and current arrays."""
        ref = np.asarray(reference, dtype=float)
        cur = np.asarray(current, dtype=float)
        combined = np.concatenate([ref, cur])
        bin_edges = np.linspace(combined.min(), combined.max(), self.bins + 1)
        ref_counts, _ = np.histogram(ref, bins=bin_edges)
        cur_counts, _ = np.histogram(cur, bins=bin_edges)
        ref_perc = ref_counts / max(ref_counts.sum(), 1)
        cur_perc = cur_counts / max(cur_counts.sum(), 1)
        ref_perc = np.where(ref_perc == 0, 1e-6, ref_perc)
        cur_perc = np.where(cur_perc == 0, 1e-6, cur_perc)
        psi = np.sum((cur_perc - ref_perc) * np.log(cur_perc / ref_perc))
        return psi

    def detect_drift(self, reference, current):
        """Return whether drift is detected and the PSI value."""
        psi = self.calculate_psi(reference, current)
        return psi > self.threshold, psi
