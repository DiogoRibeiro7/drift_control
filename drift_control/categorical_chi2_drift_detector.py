from __future__ import annotations

import numpy as np
from scipy.stats import chisquare
from .result_schema import DriftResult


class ChiSquareDriftDetector:
    """Categorical drift detector using chi-square goodness-of-fit p-value."""

    def __init__(self, alpha: float = 0.05) -> None:
        if not (0 < alpha < 1):
            raise ValueError("alpha must be between 0 and 1")
        self.alpha = float(alpha)

    @staticmethod
    def _is_pyarrow_like(values) -> bool:
        mod = getattr(getattr(values, "__class__", None), "__module__", "")
        return isinstance(mod, str) and mod.startswith("pyarrow.")

    @staticmethod
    def _arrow_counts(values) -> dict[str, int]:
        import pyarrow as pa  # type: ignore
        import pyarrow.compute as pc  # type: ignore

        arr = pa.array(values)
        vc = pc.value_counts(arr)
        out: dict[str, int] = {}
        for item in vc.to_pylist():
            out[str(item["values"])] = int(item["counts"])
        return out

    def calculate_pvalue(self, reference, current) -> float:
        if self._is_pyarrow_like(reference) or self._is_pyarrow_like(current):
            try:
                ref_map = self._arrow_counts(reference)
                cur_map = self._arrow_counts(current)
                categories = np.union1d(np.array(list(ref_map.keys())), np.array(list(cur_map.keys())))
                ref_aligned = np.array([ref_map.get(str(c), 0) for c in categories], dtype=float)
                cur_aligned = np.array([cur_map.get(str(c), 0) for c in categories], dtype=float)
                k = len(categories)
                ref_probs = (ref_aligned + 1.0) / (ref_aligned.sum() + k)
                expected = ref_probs * max(cur_aligned.sum(), 1.0)
                _, pvalue = chisquare(cur_aligned, expected)
                return float(pvalue)
            except ImportError:
                pass

        ref = np.asarray(reference, dtype=str).ravel()
        cur = np.asarray(current, dtype=str).ravel()
        if ref.size == 0 or cur.size == 0:
            raise ValueError("reference and current must be non-empty")

        ref_vals, ref_counts = np.unique(ref, return_counts=True)
        cur_vals, cur_counts = np.unique(cur, return_counts=True)
        categories = np.union1d(ref_vals, cur_vals)

        ref_map = {k: v for k, v in zip(ref_vals, ref_counts)}
        cur_map = {k: v for k, v in zip(cur_vals, cur_counts)}
        ref_aligned = np.array([ref_map.get(c, 0) for c in categories], dtype=float)
        cur_aligned = np.array([cur_map.get(c, 0) for c in categories], dtype=float)

        # Laplace smoothing to avoid zero expected counts.
        k = len(categories)
        ref_probs = (ref_aligned + 1.0) / (ref_aligned.sum() + k)
        expected = ref_probs * max(cur_aligned.sum(), 1.0)
        _, pvalue = chisquare(cur_aligned, expected)
        return float(pvalue)

    def detect_drift(self, reference, current):
        pvalue = self.calculate_pvalue(reference, current)
        return pvalue < self.alpha, pvalue

    def detect_drift_result(self, reference, current) -> DriftResult:
        drift, pvalue = self.detect_drift(reference, current)
        return DriftResult(
            method="chi2cat",
            drift=bool(drift),
            score=float(pvalue),
            p_value=float(pvalue),
            threshold=float(self.alpha),
            comparator="<",
            metadata={},
        )
