"""File errors retain their cause internally and remain readable at the CLI."""

from __future__ import annotations

import pandas as pd
import pytest
from click.testing import CliRunner
from dataexcept import DataLoadingError, FileWriteError

from drift_control._cli_app import read_csv_file, write_output_file
from drift_control.cli import check, report_command
from drift_control.drift_report import HtmlDriftReport


def test_csv_reader_identifies_bad_input(tmp_path):
    source = tmp_path / "current.csv"
    source.write_text('value\n"unfinished', encoding="utf-8")

    with pytest.raises(DataLoadingError, match="current.csv") as error:
        read_csv_file(str(source))

    assert isinstance(error.value.__cause__, pd.errors.ParserError)


def test_cli_reports_csv_failure_without_traceback(tmp_path):
    baseline = tmp_path / "baseline.csv"
    current = tmp_path / "current.csv"
    baseline.write_text("value\n1\n2\n", encoding="utf-8")
    current.write_text('value\n"unfinished', encoding="utf-8")

    for command in (check, report_command):
        result = CliRunner().invoke(
            command,
            ["--baseline", str(baseline), "--current", str(current)],
        )
        assert result.exit_code != 0
        assert "current.csv" in result.output
        assert "Traceback" not in result.output


def test_output_file_preserves_failed_destination(tmp_path):
    blocked = tmp_path / "blocked"
    blocked.write_text("ordinary file", encoding="utf-8")
    target = blocked / "report.json"

    with pytest.raises(FileWriteError, match="report.json") as error:
        write_output_file(str(target), "{}")

    assert isinstance(error.value.__cause__, OSError)


def test_cli_reports_blocked_output_without_traceback(tmp_path):
    baseline = tmp_path / "baseline.csv"
    current = tmp_path / "current.csv"
    blocked = tmp_path / "blocked"
    baseline.write_text("value\n1\n2\n3\n", encoding="utf-8")
    current.write_text("value\n4\n5\n6\n", encoding="utf-8")
    blocked.write_text("ordinary file", encoding="utf-8")

    result = CliRunner().invoke(
        report_command,
        [
            "--baseline",
            str(baseline),
            "--current",
            str(current),
            "--output",
            str(blocked / "report.json"),
        ],
    )

    assert result.exit_code != 0
    assert "report.json" in result.output
    assert "Traceback" not in result.output


def test_report_renderers_preserve_failed_destination(tmp_path):
    blocked = tmp_path / "blocked"
    blocked.write_text("ordinary file", encoding="utf-8")
    report = HtmlDriftReport(method="psi", columns={})
    history = [{"age": {"score": 0.1}}]

    for render in (
        lambda: report.render(blocked / "report.html"),
        lambda: report.render_markdown(blocked / "report.md"),
        lambda: report.render_time_series_heatmap(blocked / "heatmap.html", history),
    ):
        with pytest.raises(FileWriteError) as error:
            render()
        assert "blocked" in str(error.value)
        assert isinstance(error.value.__cause__, OSError)
