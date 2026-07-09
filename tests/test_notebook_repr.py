from drift_control.ensemble_drift_detector import EnsembleColumnResult
from drift_control.result_schema import DriftResult


def test_drift_result_repr_html_contains_key_fields():
    result = DriftResult(
        method="ks",
        drift=True,
        score=0.01,
        p_value=0.01,
        threshold=0.05,
        comparator="<",
        metadata={"calibrated_threshold": 0.04},
    )
    html = result._repr_html_()
    assert "<table>" in html
    assert "Method" in html
    assert "ks" in html
    assert "calibrated_threshold" in html


def test_drift_result_repr_html_escapes_user_content():
    result = DriftResult(
        method="<script>",
        drift=False,
        score=0.2,
        comparator="<",
        metadata={"payload": "<b>unsafe</b>"},
    )
    html = result._repr_html_()
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "&lt;b&gt;unsafe&lt;/b&gt;" in html


def test_ensemble_column_result_repr_html_contains_summary_and_methods():
    result = EnsembleColumnResult(
        drift_detected=True,
        votes=3,
        required_votes=2,
        method_results={
            "ks": {"score": 0.01, "p_value": 0.01, "drift": True},
            "psi": {"score": 0.7, "drift": True},
        },
    )
    html = result._repr_html_()
    assert "<table>" in html
    assert "Required Votes" in html
    assert "ks" in html
    assert "psi" in html
