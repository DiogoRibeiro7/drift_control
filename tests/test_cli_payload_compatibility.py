import json

from click.testing import CliRunner
import pandas as pd

from drift_control.cli import benchmark_report, check


SUPPORTED_SCHEMAS = {"1.0"}


def _write_numeric_csv(path, values):
    pd.DataFrame({"x": values}).to_csv(path, index=False)


def test_cli_schema_version_is_supported_and_shape_is_stable_for_v1(tmp_path):
    baseline = tmp_path / "b.csv"
    current = tmp_path / "c.csv"
    _write_numeric_csv(baseline, [0, 1, 2, 3, 4])
    _write_numeric_csv(current, [10, 11, 12, 13, 14])

    result = CliRunner().invoke(
        check,
        ["--baseline", str(baseline), "--current", str(current), "--output-json"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)

    assert payload["schema_version"] in SUPPORTED_SCHEMAS
    if payload["schema_version"] == "1.0":
        assert set(payload.keys()) == {
            "schema_version",
            "method",
            "threshold",
            "correction",
            "columns",
        }
        col_payload = payload["columns"]["x"]
        assert {"score", "drift"} <= set(col_payload.keys())


def test_cli_ensemble_schema_v1_shape_is_stable(tmp_path):
    baseline = tmp_path / "b.csv"
    current = tmp_path / "c.csv"
    _write_numeric_csv(baseline, [0, 1, 2, 3, 4])
    _write_numeric_csv(current, [10, 11, 12, 13, 14])

    result = CliRunner().invoke(
        check,
        [
            "--baseline",
            str(baseline),
            "--current",
            str(current),
            "--method",
            "ensemble",
            "--output-json",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)

    assert payload["schema_version"] in SUPPORTED_SCHEMAS
    if payload["schema_version"] == "1.0":
        assert set(payload.keys()) == {
            "schema_version",
            "method",
            "threshold",
            "correction",
            "columns",
            "ensemble",
        }
        assert {"methods", "vote_mode", "min_votes"} == set(payload["ensemble"].keys())
        col_payload = payload["columns"]["x"]
        assert {"score", "drift", "votes", "required_votes"} == set(col_payload.keys())


def test_benchmark_schema_version_and_v1_shape_are_stable():
    result = CliRunner().invoke(
        benchmark_report,
        [
            "--methods",
            "ks",
            "--sample-size",
            "40",
            "--n-trials",
            "2",
            "--output-format",
            "json",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)

    assert payload["schema_version"] in SUPPORTED_SCHEMAS
    if payload["schema_version"] == "1.0":
        assert set(payload.keys()) == {
            "schema_version",
            "benchmark_type",
            "methods",
            "sample_size",
            "n_trials",
            "random_seed",
            "results",
        }
