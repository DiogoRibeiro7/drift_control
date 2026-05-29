import sys
import types

import numpy as np
import pytest

from drift_control.text_embedding_drift import (
    compute_text_embeddings,
    detect_text_embedding_drift,
)


def test_compute_text_embeddings_missing_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(ImportError, match="sentence-transformers"):
        compute_text_embeddings(["hello"])


def test_compute_text_embeddings_with_fake_encoder(monkeypatch):
    class _FakeModel:
        def __init__(self, name):
            self.name = name

        def encode(self, texts, convert_to_numpy=True):
            assert convert_to_numpy is True
            return np.array([[float(len(t)), 1.0] for t in texts], dtype=float)

    fake_mod = types.SimpleNamespace(SentenceTransformer=_FakeModel)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_mod)
    emb = compute_text_embeddings(["a", "abcd"], model_name="fake")
    assert emb.shape == (2, 2)
    assert emb[0, 0] == 1.0
    assert emb[1, 0] == 4.0


def test_detect_text_embedding_drift_with_fake_encoder(monkeypatch):
    class _FakeModel:
        def __init__(self, _name):
            pass

        def encode(self, texts, convert_to_numpy=True):
            return np.array([[float(len(t)), 0.0] for t in texts], dtype=float)

    fake_mod = types.SimpleNamespace(SentenceTransformer=_FakeModel)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_mod)
    result = detect_text_embedding_drift(
        reference_texts=["a", "aa", "aaa", "aaaa"] * 40,
        current_texts=["bbbb", "bbbbb", "bbbbbb", "bbbbbbb"] * 40,
        method="energy",
        alpha=0.05,
        n_permutations=40,
        random_state=7,
    )
    assert result.method == "energy"
    assert isinstance(result.score, float)


def test_detect_text_embedding_drift_rejects_unknown_method():
    with pytest.raises(ValueError, match="method must be one of"):
        detect_text_embedding_drift(["a"], ["b"], method="psi")

