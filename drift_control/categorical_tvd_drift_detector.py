from __future__ import annotations

import numpy as np
from .result_schema import DriftResult


class TotalVariationDriftDetector:
    """Categorical drift detector using total variation distance."""

    def __init__(self, threshold: float = 0.1) -> None:
        if threshold < 0:
            raise ValueError("threshold must be non-negative")
        self.threshold = float(threshold)
        self.last_backend = "numpy"

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

    def calculate_tvd(self, reference, current) -> float:
        if self._is_pyarrow_like(reference) or self._is_pyarrow_like(current):
            try:
                ref_map = self._arrow_counts(reference)
                cur_map = self._arrow_counts(current)
                categories = np.union1d(np.array(list(ref_map.keys())), np.array(list(cur_map.keys())))
                ref_probs = np.array([ref_map.get(str(c), 0) for c in categories], dtype=float)
                cur_probs = np.array([cur_map.get(str(c), 0) for c in categories], dtype=float)
                ref_probs = ref_probs / max(ref_probs.sum(), 1.0)
                cur_probs = cur_probs / max(cur_probs.sum(), 1.0)
                self.last_backend = "pyarrow"
                return float(0.5 * np.sum(np.abs(ref_probs - cur_probs)))
            except ImportError:
                pass

        self.last_backend = "numpy"
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

    def detect_drift_result(self, reference, current) -> DriftResult:
        drift, score = self.detect_drift(reference, current)
        return DriftResult(
            method="tvdcat",
            drift=bool(drift),
            score=float(score),
            p_value=None,
            threshold=float(self.threshold),
            comparator=">",
            metadata={},
        )
