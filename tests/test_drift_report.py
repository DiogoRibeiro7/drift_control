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


def test_drift_report_markdown_top_drifting_sorted_and_limited(tmp_path: Path):
    report = DriftReport(
        method="ks",
        correction="none",
        columns={
            "a": {"score": 0.2, "p_value": 0.2, "drift": False},
            "b": {"score": 0.9, "p_value": 0.01, "drift": True},
            "c": {"score": 0.4, "p_value": 0.04, "drift": True},
        },
    )
    md = report.top_drifting_markdown(limit=2)
    lines = md.splitlines()
    assert len(lines) == 4  # header + separator + 2 rows
    assert "| b |" in lines[2]
    assert "| c |" in lines[3]

    out = tmp_path / "top.md"
    report.render_markdown(out, limit=2)
    assert out.exists()
