from pathlib import Path

from drift_control.drift_report import DriftReport


def test_drift_report_renders_html_file(tmp_path: Path):
    report = DriftReport(
        method="ks",
        correction="none",
        columns={
            "x": {"score": 0.01, "p_value": 0.01, "drift": True},
            "y": {"score": 0.9, "drift": False},
        },
    )
    out = tmp_path / "drift_report.html"
    report.render(out)
    html = out.read_text(encoding="utf-8")
    assert "<html" in html
    assert "Drift Report" in html
    assert "ks" in html
    assert "x" in html
    assert "y" in html
