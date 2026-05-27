from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .result_schema import DriftResult
from .unified_drift_detector import UnifiedDriftDetector


@dataclass(frozen=True)
class SliceResult:
    slice_value: str
    n_reference: int
    n_current: int
    drift_result: DriftResult


class SliceDriftDetector:
    """Run a detector independently per cohort/slice and return slice-level results."""

    def __init__(
        self,
        detector: UnifiedDriftDetector,
        by: str,
        min_samples_per_slice: int = 20,
    ) -> None:
        if min_samples_per_slice < 2:
            raise ValueError("min_samples_per_slice must be >= 2")
        self.detector = detector
        self.by = by
        self.min_samples_per_slice = min_samples_per_slice

    def detect_drift(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        value_column: str,
    ) -> list[SliceResult]:
        if not isinstance(reference_df, pd.DataFrame) or not isinstance(current_df, pd.DataFrame):
            raise TypeError("reference_df and current_df must be pandas DataFrames")
        for df_name, df in (("reference_df", reference_df), ("current_df", current_df)):
            for col in (self.by, value_column):
                if col not in df.columns:
                    raise ValueError(f"{df_name} is missing required column '{col}'")

        ref = reference_df[[self.by, value_column]].copy()
        cur = current_df[[self.by, value_column]].copy()
        ref[self.by] = ref[self.by].astype(str)
        cur[self.by] = cur[self.by].astype(str)

        slices = sorted(set(ref[self.by].unique()) | set(cur[self.by].unique()))
        results: list[SliceResult] = []

        for slice_value in slices:
            ref_slice = ref.loc[ref[self.by] == slice_value, value_column]
            cur_slice = cur.loc[cur[self.by] == slice_value, value_column]
            n_ref = int(ref_slice.shape[0])
            n_cur = int(cur_slice.shape[0])

            if n_ref < self.min_samples_per_slice or n_cur < self.min_samples_per_slice:
                continue

            drift_result = self.detector.detect_drift(ref_slice.values, cur_slice.values)
            results.append(
                SliceResult(
                    slice_value=slice_value,
                    n_reference=n_ref,
                    n_current=n_cur,
                    drift_result=drift_result,
                )
            )

        return results

    @staticmethod
    def to_frame(results: list[SliceResult]) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for r in results:
            rows.append(
                {
                    "slice": r.slice_value,
                    "n_reference": r.n_reference,
                    "n_current": r.n_current,
                    "method": r.drift_result.method,
                    "drift": r.drift_result.drift,
                    "score": r.drift_result.score,
                    "p_value": r.drift_result.p_value,
                    "threshold": r.drift_result.threshold,
                    "comparator": r.drift_result.comparator,
                }
            )
        return pd.DataFrame(rows)
