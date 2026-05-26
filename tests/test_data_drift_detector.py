import pandas as pd
import pytest

from drift_control.drift_detector import DataDriftDetector


def _make_frames():
    df_prior = pd.DataFrame({
        "num": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "cat": ["a", "a", "b", "b", "c", "c", "a", "b"],
    })
    df_post = pd.DataFrame({
        "num": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
        "cat": ["a", "b", "b", "b", "c", "c", "c", "c"],
    })
    return df_prior, df_post


def test_calculate_drift_returns_both_groups():
    df_prior, df_post = _make_frames()
    result = DataDriftDetector(df_prior, df_post).calculate_drift()

    assert set(result) == {"categorical", "numerical"}
    assert set(result["categorical"]) == {"cat"}
    assert set(result["numerical"]) == {"num"}

    num = result["numerical"]["num"]
    assert num["ks_2sample_test_statistic"] == pytest.approx(1.0)
    assert num["ks_2sample_test_p_value"] < 0.05
    assert num["wasserstein_distance"] > 0

    cat = result["categorical"]["cat"]
    for key in (
        "chi_square_test_statistic",
        "chi_square_test_p_value",
        "kl_divergence_post_given_prior",
        "kl_divergence_prior_given_post",
        "jensen_shannon_distance",
        "wasserstein_distance",
    ):
        assert key in cat


def test_constructor_rejects_mismatched_columns():
    df_prior = pd.DataFrame({"a": [1, 2]})
    df_post = pd.DataFrame({"b": [1, 2]})
    with pytest.raises(ValueError, match="same column names"):
        DataDriftDetector(df_prior, df_post)


def test_calculate_drift_skips_constant_columns_safely():
    df_prior = pd.DataFrame({"num": [1.0, 1.0, 1.0, 1.0]})
    df_post = pd.DataFrame({"num": [1.0, 1.0, 2.0, 2.0]})
    result = DataDriftDetector(df_prior, df_post).calculate_drift()
    # JSD is NaN when prior is constant, but other stats are still produced.
    assert "num" in result["numerical"]
    assert result["numerical"]["num"]["jensen_shannon_distance"] != \
        result["numerical"]["num"]["jensen_shannon_distance"]  # NaN check
