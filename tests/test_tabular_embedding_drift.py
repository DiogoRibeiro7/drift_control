import numpy as np
import pandas as pd
import pytest

from drift_control.tabular_embedding_drift import (
    compute_tabular_embeddings,
    detect_tabular_embedding_drift,
)


def test_compute_tabular_embeddings_from_dataframe():
    df = pd.DataFrame({"e1": [0.1, 0.2], "e2": [1.0, 2.0], "x": ["a", "b"]})
    emb = compute_tabular_embeddings(df, embedding_columns=["e1", "e2"])
    assert emb.shape == (2, 2)
    assert emb.dtype == float


def test_compute_tabular_embeddings_from_array_requires_2d():
    arr = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="must be 2D"):
        compute_tabular_embeddings(arr)


def test_compute_tabular_embeddings_dataframe_rejects_missing_columns():
    df = pd.DataFrame({"e1": [0.1], "e2": [0.2]})
    with pytest.raises(ValueError, match="not found"):
        compute_tabular_embeddings(df, embedding_columns=["e3"])


def test_detect_tabular_embedding_drift_dataframe_energy():
    rng = np.random.default_rng(0)
    ref = pd.DataFrame(
        {
            "emb_0": rng.normal(0.0, 1.0, size=160),
            "emb_1": rng.normal(0.0, 1.0, size=160),
            "feature": rng.integers(0, 3, size=160),
        }
    )
    cur = pd.DataFrame(
        {
            "emb_0": rng.normal(0.8, 1.0, size=160),
            "emb_1": rng.normal(0.8, 1.0, size=160),
            "feature": rng.integers(0, 3, size=160),
        }
    )
    res = detect_tabular_embedding_drift(
        reference_data=ref,
        current_data=cur,
        embedding_columns=["emb_0", "emb_1"],
        method="energy",
        alpha=0.05,
        n_permutations=40,
        random_state=7,
    )
    assert res.method == "energy"
    assert isinstance(res.score, float)


def test_detect_tabular_embedding_drift_array_input():
    rng = np.random.default_rng(0)
    ref = rng.normal(0.0, 1.0, size=(160, 4))
    cur = rng.normal(0.5, 1.0, size=(160, 4))
    res = detect_tabular_embedding_drift(ref, cur, method="mmd", alpha=0.05, n_permutations=50, random_state=7)
    assert res.method == "mmd"
    assert res.p_value is not None
