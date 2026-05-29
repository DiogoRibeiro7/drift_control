from __future__ import annotations

from typing import Sequence

import numpy as np

from .result_schema import DriftResult
from .unified_drift_detector import UnifiedDriftDetector


def compute_image_embeddings(
    images: Sequence[np.ndarray],
    embedder=None,
) -> np.ndarray:
    """Embed image arrays using a custom embedder or torchvision default."""
    if not images:
        raise ValueError("images must be a non-empty sequence")
    if embedder is None:
        try:
            import torch  # type: ignore
            from torchvision.models import resnet18  # type: ignore
            from torchvision.models.feature_extraction import create_feature_extractor  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "torch and torchvision are required when embedder is not provided"
            ) from exc

        model = resnet18(weights=None)
        model.eval()
        extractor = create_feature_extractor(model, return_nodes={"avgpool": "feat"})

        def embedder(batch_images: Sequence[np.ndarray]) -> np.ndarray:
            arr = np.asarray(batch_images, dtype=np.float32)
            if arr.ndim != 4:
                raise ValueError("images must be an array-like of shape (N, H, W, C)")
            x = np.transpose(arr, (0, 3, 1, 2))
            with torch.no_grad():
                out = extractor(torch.from_numpy(x))["feat"]
            return out.reshape(out.shape[0], -1).numpy()

    vecs = np.asarray(embedder(images), dtype=float)
    if vecs.ndim != 2:
        raise ValueError("image embedder returned unexpected shape; expected 2D array")
    return vecs


def detect_image_embedding_drift(
    reference_images: Sequence[np.ndarray],
    current_images: Sequence[np.ndarray],
    method: str = "energy",
    embedder=None,
    **detector_kwargs,
) -> DriftResult:
    """Compute image embeddings and run multivariate drift detection."""
    if method not in {"mmd", "energy", "c2st"}:
        raise ValueError("method must be one of: mmd, energy, c2st")
    ref_emb = compute_image_embeddings(reference_images, embedder=embedder)
    cur_emb = compute_image_embeddings(current_images, embedder=embedder)
    detector = UnifiedDriftDetector(method=method, **detector_kwargs)
    return detector.detect_drift(ref_emb, cur_emb)

