import pandas as pd
import pytest

from drift_control.ensemble_drift_detector import EnsembleDriftDetector


def test_ensemble_detects_strong_shift_with_majority_vote():
    df_prior = pd.DataFrame({"x": [0, 1, 2, 3, 4, 5]})
    df_post = pd.DataFrame({"x": [10, 11, 12, 13, 14, 15]})

    detector = EnsembleDriftDetector(methods=["psi", "ks", "cvm", "js"], vote_mode="majority")
    out = detector.detect_drift(df_prior, df_post)

    assert "x" in out
    assert out["x"].drift_detected is True
    assert out["x"].votes >= out["x"].required_votes


def test_ensemble_any_vote_mode_detects_when_one_method_fires():
    df_prior = pd.DataFrame({"x": [0, 1, 2, 3, 4, 5]})
    df_post = pd.DataFrame({"x": [0, 1, 2, 3, 4, 6]})

    detector = EnsembleDriftDetector(methods=["psi", "js"], vote_mode="any")
    out = detector.detect_drift(df_prior, df_post)
    assert out["x"].required_votes == 1


def test_ensemble_rejects_unknown_method():
    with pytest.raises(ValueError, match="unknown methods"):
        EnsembleDriftDetector(methods=["ks", "foo"])


def test_ensemble_rejects_bad_min_votes():
    detector = EnsembleDriftDetector(methods=["ks", "cvm"], min_votes=3)
    with pytest.raises(ValueError, match="min_votes"):
        detector.detect_drift(pd.DataFrame({"x": [1, 2]}), pd.DataFrame({"x": [2, 3]}))
