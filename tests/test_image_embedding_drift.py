import numpy as np
import pytest

from drift_control.image_embedding_drift import (
    compute_image_embeddings,
    detect_image_embedding_drift,
)


def _fake_embedder(images):
    arr = np.asarray(images, dtype=float)
    return arr.reshape(arr.shape[0], -1)[:, :8]


def test_compute_image_embeddings_requires_data():
    with pytest.raises(ValueError, match="non-empty"):
        compute_image_embeddings([])


def test_compute_image_embeddings_with_fake_embedder():
    imgs = [np.ones((8, 8, 3), dtype=np.float32), np.zeros((8, 8, 3), dtype=np.float32)]
    emb = compute_image_embeddings(imgs, embedder=_fake_embedder)
    assert emb.shape == (2, 8)


def test_compute_image_embeddings_rejects_bad_shape():
    def _bad_embedder(_images):
        return np.array([1.0, 2.0, 3.0])

    imgs = [np.ones((8, 8, 3), dtype=np.float32)]
    with pytest.raises(ValueError, match="expected 2D array"):
        compute_image_embeddings(imgs, embedder=_bad_embedder)


def test_detect_image_embedding_drift_with_fake_embedder():
    rng = np.random.default_rng(0)
    ref = [rng.normal(0, 1, size=(8, 8, 3)).astype(np.float32) for _ in range(160)]
    cur = [rng.normal(0.7, 1, size=(8, 8, 3)).astype(np.float32) for _ in range(160)]
    res = detect_image_embedding_drift(
        reference_images=ref,
        current_images=cur,
        method="energy",
        embedder=_fake_embedder,
        alpha=0.05,
        n_permutations=40,
        random_state=7,
    )
    assert res.method == "energy"
    assert isinstance(res.score, float)


def test_detect_image_embedding_drift_rejects_unknown_method():
    img = [np.zeros((8, 8, 3), dtype=np.float32)]
    with pytest.raises(ValueError, match="method must be one of"):
        detect_image_embedding_drift(img, img, method="psi", embedder=_fake_embedder)
