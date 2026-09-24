import pytest

from drift_control.multiple_testing import adjust_pvalues


def test_adjust_pvalues_none_identity():
    p = [0.01, 0.02, 0.5]
    assert adjust_pvalues(p, method="none") == p


def test_adjust_pvalues_bonferroni_monotonic_increase():
    p = [0.01, 0.02]
    adj = adjust_pvalues(p, method="bonferroni")
    assert adj[0] == pytest.approx(0.02)
    assert adj[1] == pytest.approx(0.04)


def test_adjust_pvalues_bh_within_bounds():
    p = [0.01, 0.03, 0.2]
    adj = adjust_pvalues(p, method="bh")
    assert all(0.0 <= v <= 1.0 for v in adj)


def test_adjust_pvalues_empty_returns_empty():
    assert adjust_pvalues([], method="bh") == []


def test_adjust_pvalues_rejects_unknown_method():
    with pytest.raises(ValueError, match="correction must be one of"):
        adjust_pvalues([0.1], method="holm")
