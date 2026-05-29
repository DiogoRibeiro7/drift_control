from __future__ import annotations

from typing import Sequence

import numpy as np

from .result_schema import DriftResult
from .unified_drift_detector import UnifiedDriftDetector


def compute_text_embeddings(
    texts: Sequence[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> np.ndarray:
    """Encode raw texts into embedding vectors using sentence-transformers."""
    if not texts:
        raise ValueError("texts must be a non-empty sequence")
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "sentence-transformers is required for text embedding helpers"
        ) from exc
    model = SentenceTransformer(model_name)
    vectors = model.encode(list(texts), convert_to_numpy=True)
    arr = np.asarray(vectors, dtype=float)
    if arr.ndim != 2:
        raise ValueError("embedding model returned unexpected shape; expected 2D array")
    return arr


def detect_text_embedding_drift(
    reference_texts: Sequence[str],
    current_texts: Sequence[str],
    method: str = "energy",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    **detector_kwargs,
) -> DriftResult:
    """Compute embeddings from text and run multivariate drift detection."""
    if method not in {"mmd", "energy", "c2st"}:
        raise ValueError("method must be one of: mmd, energy, c2st")
    ref_emb = compute_text_embeddings(reference_texts, model_name=model_name)
    cur_emb = compute_text_embeddings(current_texts, model_name=model_name)
    detector = UnifiedDriftDetector(method=method, **detector_kwargs)
    return detector.detect_drift(ref_emb, cur_emb)

