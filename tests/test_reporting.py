"""Phase 8: DriftReport aggregation, severity, rendering, alert payload."""

import json

import pytest

from drift_control.core.exceptions import ValidationError
from drift_control.core.result import DriftResult
from drift_control.monitoring import DriftReport, DriftReportItem


def _threshold_result(*, drift, score, threshold, method="psi"):
    return DriftResult(method=method, drift=drift, score=score, threshold=threshold, comparator=">")


def _pvalue_result(*, drift, p, alpha=0.05, method="ks"):
    return DriftResult(
        method=method,
        drift=drift,
        score=p,
        threshold=alpha,
        comparator="<",
        p_value=p,
        metadata={"feature": f"col_{method}"},
    )


# --- aggregation ------------------------------------------------------------


def test_from_results_summary():
    results = [
        _threshold_result(drift=False, score=0.05, threshold=0.2),
        _threshold_result(drift=True, score=0.6, threshold=0.2),
        _pvalue_result(drift=True, p=0.001),
    ]
    report = DriftReport.from_results(results)
    assert report.n_total == 3
    assert report.n_drifting == 2
    assert report.drift_rate == pytest.approx(2 / 3)
    assert report.any_drift is True


def test_empty_report():
    report = DriftReport.from_results([])
    assert report.n_total == 0
    assert report.max_severity == "none"
    assert report.any_drift is False
    assert report.to_dict()["items"] == []


def test_names_override_and_length_check():
    results = [_threshold_result(drift=True, score=0.6, threshold=0.2)]
    report = DriftReport.from_results(results, names=["my_feature"])
    assert report.items[0].name == "my_feature"
    with pytest.raises(ValidationError):
        DriftReport.from_results(results, names=["a", "b"])


def test_default_name_from_metadata():
    report = DriftReport.from_results([_pvalue_result(drift=True, p=0.01)])
    assert report.items[0].name == "col_ks"


def test_default_name_falls_back_to_check_index_when_missing():
    result = DriftResult(
        method="",
        drift=True,
        score=0.6,
        threshold=0.2,
        comparator=">",
        metadata={},
    )
    report = DriftReport.from_results([result])
    assert report.items[0].name == "check_0"


# --- severity scoring -------------------------------------------------------


def test_severity_none_for_no_drift():
    report = DriftReport.from_results([_threshold_result(drift=False, score=0.1, threshold=0.2)])
    assert report.items[0].severity == "none"


def test_severity_threshold_scaling():
    low = DriftReport.from_results([_threshold_result(drift=True, score=0.24, threshold=0.2)])
    med = DriftReport.from_results([_threshold_result(drift=True, score=0.4, threshold=0.2)])
    high = DriftReport.from_results([_threshold_result(drift=True, score=0.8, threshold=0.2)])
    assert low.items[0].severity == "low"
    assert med.items[0].severity == "medium"
    assert high.items[0].severity == "high"


def test_severity_pvalue_scaling():
    high = DriftReport.from_results([_pvalue_result(drift=True, p=0.001)])  # p/alpha=0.02
    med = DriftReport.from_results([_pvalue_result(drift=True, p=0.02)])  # p/alpha=0.4
    low = DriftReport.from_results([_pvalue_result(drift=True, p=0.04)])  # p/alpha=0.8
    assert high.items[0].severity == "high"
    assert med.items[0].severity == "medium"
    assert low.items[0].severity == "low"


def test_max_severity_is_worst():
    results = [
        _threshold_result(drift=True, score=0.24, threshold=0.2),  # low
        _threshold_result(drift=True, score=0.8, threshold=0.2),  # high
    ]
    assert DriftReport.from_results(results).max_severity == "high"


# --- rendering --------------------------------------------------------------


def test_to_json_is_valid_and_stable_schema():
    report = DriftReport.from_results([_threshold_result(drift=True, score=0.8, threshold=0.2)])
    payload = json.loads(report.to_json())
    assert payload["schema_version"] == "1.0"
    assert payload["summary"]["n_drifting"] == 1
    assert payload["items"][0]["severity"] == "high"


def test_to_markdown_has_table():
    report = DriftReport.from_results(
        [_threshold_result(drift=True, score=0.6, threshold=0.2)], names=["feat"]
    )
    md = report.to_markdown()
    assert "# Drift Report" in md
    assert "| Name | Method | Drift | Severity | Score | Threshold |" in md
    assert "| feat | psi | yes |" in md


# --- alert payload / notify -------------------------------------------------


def test_alert_payload_and_notify():
    report = DriftReport.from_results(
        [_threshold_result(drift=True, score=0.8, threshold=0.2)], names=["x"]
    )
    payload = report.to_alert_payload()
    assert payload["x"]["drift"] is True
    assert payload["x"]["severity"] == "high"

    captured = {}

    class _Sink:
        def send(self, p):
            captured.update(p)

    report.notify(_Sink())
    assert captured["x"]["drift"] is True


def test_works_with_real_alert_sink():
    from drift_control.alert_sinks import CompositeAlertSink, LogAlertSink

    report = DriftReport.from_results([_threshold_result(drift=True, score=0.6, threshold=0.2)])
    # should not raise
    report.notify(CompositeAlertSink([LogAlertSink()]))


# --- item ------------------------------------------------------------------


def test_report_item_to_dict():
    item = DriftReportItem(
        name="f",
        method="ks",
        drift=True,
        severity="high",
        score=0.01,
        threshold=0.05,
        comparator="<",
        p_value=0.01,
    )
    d = item.to_dict()
    assert d["name"] == "f" and d["severity"] == "high"


def test_not_exposed_at_root_to_avoid_legacy_collision():
    import drift_control

    # legacy DriftReport stays at the package root; the new one lives in monitoring
    assert drift_control.DriftReport is not DriftReport
