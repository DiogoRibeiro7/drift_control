"""drift-control-report: structured DriftReport over two CSVs."""

import json

import numpy as np
import pandas as pd
from click.testing import CliRunner

from drift_control.cli import report_command

RNG = np.random.default_rng(0)


def _write(tmp_path, name, frame):
    path = tmp_path / name
    frame.to_csv(path, index=False)
    return str(path)


def _baseline(n=400):
    return pd.DataFrame(
        {
            "age": RNG.normal(40, 5, n),
            "country": RNG.choice(["US", "UK", "DE"], n, p=[0.5, 0.3, 0.2]),
        }
    )


def _drifted(n=400):
    return pd.DataFrame(
        {
            "age": RNG.normal(47, 5, n),
            "country": RNG.choice(["US", "UK", "DE"], n, p=[0.2, 0.3, 0.5]),
        }
    )


def test_report_json_detects_mixed_drift(tmp_path):
    base = _write(tmp_path, "b.csv", _baseline())
    cur = _write(tmp_path, "c.csv", _drifted())
    result = CliRunner().invoke(report_command, ["--baseline", base, "--current", cur])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["schema_version"] == "1.0"
    assert payload["summary"]["n_drifting"] == 2
    methods = {it["name"]: it["method"] for it in payload["items"]}
    assert methods["age"] == "ks" and methods["country"] == "chi2"


def test_report_no_drift(tmp_path):
    base = _write(tmp_path, "b.csv", _baseline())
    cur = _write(tmp_path, "c.csv", _baseline())
    result = CliRunner().invoke(report_command, ["--baseline", base, "--current", cur])
    payload = json.loads(result.output)
    assert payload["summary"]["n_drifting"] == 0


def test_report_markdown(tmp_path):
    base = _write(tmp_path, "b.csv", _baseline())
    cur = _write(tmp_path, "c.csv", _drifted())
    result = CliRunner().invoke(
        report_command, ["--baseline", base, "--current", cur, "--format", "markdown"]
    )
    assert result.exit_code == 0
    assert "# Drift Report" in result.output
    assert "| Name | Method |" in result.output


def test_report_fail_on_drift_exit_code(tmp_path):
    base = _write(tmp_path, "b.csv", _baseline())
    cur = _write(tmp_path, "c.csv", _drifted())
    result = CliRunner().invoke(
        report_command, ["--baseline", base, "--current", cur, "--fail-on-drift"]
    )
    assert result.exit_code == 1  # drift present
    no_drift = CliRunner().invoke(
        report_command,
        ["--baseline", base, "--current", base, "--fail-on-drift"],
    )
    assert no_drift.exit_code == 0


def test_report_writes_output_file(tmp_path):
    base = _write(tmp_path, "b.csv", _baseline())
    cur = _write(tmp_path, "c.csv", _drifted())
    out = tmp_path / "report.json"
    result = CliRunner().invoke(
        report_command,
        ["--baseline", base, "--current", cur, "--output", str(out)],
    )
    assert result.exit_code == 0
    assert json.loads(out.read_text())["summary"]["n_drifting"] == 2


def test_report_psi_numeric_method(tmp_path):
    base = _write(tmp_path, "b.csv", _baseline())
    cur = _write(tmp_path, "c.csv", _drifted())
    result = CliRunner().invoke(
        report_command,
        ["--baseline", base, "--current", cur, "--numeric-method", "psi"],
    )
    assert result.exit_code == 0
    methods = {it["name"]: it["method"] for it in json.loads(result.output)["items"]}
    assert methods["age"] == "psi"


def test_report_bad_column_mismatch_errors(tmp_path):
    base = _write(tmp_path, "b.csv", _baseline())
    cur = _write(tmp_path, "c.csv", pd.DataFrame({"age": [1.0, 2.0], "x": [3.0, 4.0]}))
    result = CliRunner().invoke(report_command, ["--baseline", base, "--current", cur])
    assert result.exit_code != 0
    assert "columns" in result.output
