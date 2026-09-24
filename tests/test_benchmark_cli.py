import json

from click.testing import CliRunner

from drift_control.cli import benchmark_report


def test_benchmark_cli_json_output():
    runner = CliRunner()
    result = runner.invoke(
        benchmark_report,
        [
            "--methods",
            "ks,psi",
            "--sample-size",
            "60",
            "--n-trials",
            "3",
            "--random-seed",
            "0",
            "--output-format",
            "json",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["schema_version"] == "1.0"
    assert payload["benchmark_type"] == "synthetic_drift"
    assert payload["methods"] == ["ks", "psi"]
    assert isinstance(payload["results"], list)
    assert len(payload["results"]) == 8  # 2 methods x 4 scenarios


def test_benchmark_cli_csv_output_and_file_write(tmp_path):
    out_path = tmp_path / "bench.csv"
    runner = CliRunner()
    result = runner.invoke(
        benchmark_report,
        [
            "--methods",
            "ks",
            "--sample-size",
            "50",
            "--n-trials",
            "2",
            "--output-format",
            "csv",
            "--output-path",
            str(out_path),
        ],
    )
    assert result.exit_code == 0
    assert (
        "method,scenario,n_trials,expected_drift,drift_rate,avg_score,avg_latency_ms"
        in result.output
    )
    assert out_path.exists()
    text = out_path.read_text(encoding="utf-8")
    assert "method,scenario,n_trials,expected_drift,drift_rate,avg_score,avg_latency_ms" in text
