import matplotlib
import pandas as pd

matplotlib.use("Agg")
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

from drift_control.multivariate_drift_detector import CovariateShiftDetector


def test_covariate_shift_detector(tmp_path, monkeypatch):
    df1 = pd.DataFrame({'f1': [0, 0, 1, 1], 'f2': [0, 1, 0, 1]})
    df2 = pd.DataFrame({'f1': [1, 1, 0, 0], 'f2': [1, 0, 1, 0]})
    detector = CovariateShiftDetector(df1, df2)
    monkeypatch.chdir(tmp_path)
    result = detector.monitor_shift(test_size=0.5, random_state=0, n_permutations=20)
    assert "drift_detected" in result
    assert "roc_auc" in result
    assert list(tmp_path.glob("distribution_*.png")) == []


def test_covariate_shift_detector_does_not_mutate_inputs():
    df1 = pd.DataFrame({'f1': [0, 0, 1, 1], 'f2': [0, 1, 0, 1]})
    df2 = pd.DataFrame({'f1': [1, 1, 0, 0], 'f2': [1, 0, 1, 0]})
    df1_before = df1.copy(deep=True)
    df2_before = df2.copy(deep=True)

    CovariateShiftDetector(df1, df2)

    pd.testing.assert_frame_equal(df1, df1_before)
    pd.testing.assert_frame_equal(df2, df2_before)


def test_visualize_shift_can_save_when_requested(tmp_path):
    df1 = pd.DataFrame({'f1': [0, 0, 1, 1], 'f2': [0, 1, 0, 1]})
    df2 = pd.DataFrame({'f1': [1, 1, 0, 0], 'f2': [1, 0, 1, 0]})
    detector = CovariateShiftDetector(df1, df2)
    detector.visualize_shift(save_dir=str(tmp_path), show=False)
    assert len(list(tmp_path.glob("distribution_*.png"))) == 2


def test_covariate_shift_detector_non_numeric_requires_preprocessor():
    df1 = pd.DataFrame({'city': ['a', 'b', 'a', 'c'], 'f1': [0.0, 0.1, 0.2, 0.3]})
    df2 = pd.DataFrame({'city': ['c', 'c', 'b', 'a'], 'f1': [1.0, 1.1, 1.2, 1.3]})
    with pytest.raises(TypeError, match="requires numeric features"):
        CovariateShiftDetector(df1, df2)


def test_covariate_shift_detector_accepts_preprocessor_and_custom_estimator():
    df1 = pd.DataFrame({'city': ['a', 'b', 'a', 'c'], 'f1': [0.0, 0.1, 0.2, 0.3]})
    df2 = pd.DataFrame({'city': ['c', 'c', 'b', 'a'], 'f1': [1.0, 1.1, 1.2, 1.3]})
    pre = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), ["city"])],
        remainder="passthrough",
    )
    detector = CovariateShiftDetector(
        df1,
        df2,
        estimator=LogisticRegression(max_iter=1000),
        preprocessor=pre,
    )
    result = detector.monitor_shift(test_size=0.5, random_state=0, n_permutations=20)
    assert "drift_detected" in result
