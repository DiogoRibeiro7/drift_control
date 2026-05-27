from __future__ import annotations

import numpy as np


class TotalVariationDriftDetector:
    """Categorical drift detector using total variation distance."""

    def __init__(self, threshold: float = 0.1) -> None:
        if threshold < 0:
            raise ValueError("threshold must be non-negative")
        self.threshold = float(threshold)

    def calculate_tvd(self, reference, current) -> float:
        ref = np.asarray(reference, dtype=str).ravel()
        cur = np.asarray(current, dtype=str).ravel()
        if ref.size == 0 or cur.size == 0:
            raise ValueError("reference and current must be non-empty")

        ref_vals, ref_counts = np.unique(ref, return_counts=True)
        cur_vals, cur_counts = np.unique(cur, return_counts=True)
        categories = np.union1d(ref_vals, cur_vals)

        ref_map = {k: v for k, v in zip(ref_vals, ref_counts)}
        cur_map = {k: v for k, v in zip(cur_vals, cur_counts)}

        ref_probs = np.array([ref_map.get(c, 0) for c in categories], dtype=float)
        cur_probs = np.array([cur_map.get(c, 0) for c in categories], dtype=float)

        ref_probs = ref_probs / max(ref_probs.sum(), 1.0)
        cur_probs = cur_probs / max(cur_probs.sum(), 1.0)

        return float(0.5 * np.sum(np.abs(ref_probs - cur_probs)))

    def detect_drift(self, reference, current):
        score = self.calculate_tvd(reference, current)
        return score > self.threshold, score
