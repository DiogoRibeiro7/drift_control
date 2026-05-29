from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .result_schema import DriftResult
from .unified_drift_detector import UnifiedDriftDetector


def compute_tabular_embeddings(
    data: pd.DataFrame | np.ndarray,
    embedding_columns: Sequence[str] | None = None,
) -> np.ndarray:
    """Return a 2D embedding matrix from tabular inputs.

    - If ``data`` is a DataFrame, ``embedding_columns`` must name numeric columns.
    - If ``data`` is an ndarray, it must already be 2D and ``embedding_columns`` is ignored.
    """
    if isinstance(data, np.ndarray):
        arr = np.asarray(data, dtype=float)
        if arr.ndim != 2:
            raise ValueError("tabular embedding array must be 2D")
        return arr

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame or numpy ndarray")
    if data.empty:
        raise ValueError("data must be non-empty")
    if embedding_columns is None or len(embedding_columns) == 0:
        raise ValueError("embedding_columns must be a non-empty sequence for DataFrame inputs")

    missing = [c for c in embedding_columns if c not in data.columns]
    if missing:
        raise ValueError(f"embedding_columns not found in data: {missing}")

    emb = data.loc[:, list(embedding_columns)].apply(pd.to_numeric, errors="coerce")
    if emb.isna().any().any():
        raise ValueError("embedding columns must be numeric and non-null")
    arr = emb.to_numpy(dtype=float)
    if arr.ndim != 2:
        raise ValueError("computed tabular embeddings must be 2D")
    return arr


def detect_tabular_embedding_drift(
    reference_data: pd.DataFrame | np.ndarray,
    current_data: pd.DataFrame | np.ndarray,
    embedding_columns: Sequence[str] | None = None,
    method: str = "energy",
    **detector_kwargs,
) -> DriftResult:
    """Run multivariate drift detection on precomputed tabular embeddings."""
    if method not in {"mmd", "energy", "c2st"}:
        raise ValueError("method must be one of: mmd, energy, c2st")
    ref_emb = compute_tabular_embeddings(reference_data, embedding_columns=embedding_columns)
    cur_emb = compute_tabular_embeddings(current_data, embedding_columns=embedding_columns)
    detector = UnifiedDriftDetector(method=method, **detector_kwargs)
    return detector.detect_drift(ref_emb, cur_emb)

