import json

import pandas as pd
from click.testing import CliRunner

import drift_control.cli as cli_mod
from drift_control.baseline_manager import BaselineManager
from drift_control.cli import check


def test_cli_runs(tmp_path):
    baseline = tmp_path / "b.csv"
    current = tmp_path / "c.csv"
    pd.DataFrame({"x": [0, 1, 2]}).to_csv(baseline, index=False)
    pd.DataFrame({"x": [3, 4, 5]}).to_csv(current, index=False)
    runner = CliRunner()
    result = runner.invoke(check, ["--baseline", str(baseline), "--current", str(current)])
    assert result.exit_code == 0
    assert "x:" in result.output


def test_cli_schema_mismatch_fails(tmp_path):
    baseline = tmp_path / "b.csv"
    current = tmp_path / "c.csv"
    pd.DataFrame({"x": [0, 1, 2]}).to_csv(baseline, index=False)
    pd.DataFrame({"y": [3, 4, 5]}).to_csv(current, index=False)
    runner = CliRunner()
    result = runner.invoke(check, ["--baseline", str(baseline), "--current", str(current)])
    assert result.exit_code != 0
    assert "Schema mismatch" in result.output


def test_cli_non_numeric_column_fails(tmp_path):
    baseline = tmp_path / "b.csv"
    current = tmp_path / "c.csv"
    pd.DataFrame({"x": ["a", "b", "c"]}).to_csv(baseline, index=False)
    pd.DataFrame({"x": ["d", "e", "f"]}).to_csv(current, index=False)
    runner = CliRunner()
    result = runner.invoke(check, ["--baseline", str(baseline), "--current", str(current)])
    assert result.exit_code != 0
    assert "must be numeric" in result.output


def test_cli_output_json_emits_structured_payload(tmp_path):
    baseline = tmp_path / "b.csv"
    current = tmp_path / "c.csv"
    pd.DataFrame({"x": [0, 1, 2, 3, 4]}).to_csv(baseline, index=False)
    pd.DataFrame({"x": [10, 11, 12, 13, 14]}).to_csv(current, index=False)
    runner = CliRunner()
    result = runner.invoke(
        check,
        ["--baseline", str(baseline), "--current", str(current), "--output-json"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["schema_version"] == "1.0"
    assert payload["method"] == "psi"
    assert payload["threshold"] == 0.2
    assert "x" in payload["columns"]
    assert payload["columns"]["x"]["drift"] is True
    assert isinstance(payload["columns"]["x"]["score"], float)


def test_cli_threshold_override_changes_drift_flag(tmp_path):
    baseline = tmp_path / "b.csv"
    current = tmp_path / "c.csv"
    pd.DataFrame({"x": [0, 1, 2, 3, 4]}).to_csv(baseline, index=False)
    pd.DataFrame({"x": [10, 11, 12, 13, 14]}).to_csv(current, index=False)
    runner = CliRunner()
    # Same data, but a huge threshold should suppress the drift flag.
    result = runner.invoke(
        check,
        [
            "--baseline",
            str(baseline),
            "--current",
            str(current),
            "--threshold",
            "1000.0",
            "--output-json",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["schema_version"] == "1.0"
    assert payload["threshold"] == 1000.0
    assert payload["columns"]["x"]["drift"] is False


def test_cli_psi_kll_strategy_output_json(tmp_path):
    baseline = tmp_path / "b.csv"
    current = tmp_path / "c.csv"
    pd.DataFrame({"x": [float(i) for i in range(200)]}).to_csv(baseline, index=False)
    pd.DataFrame({"x": [float(i) + 40.0 for i in range(200)]}).to_csv(current, index=False)
    runner = CliRunner()
    result = runner.invoke(
        check,
        [
            "--baseline",
            str(baseline),
            "--current",
            str(current),
            "--method",
            "psi",
            "--psi-strategy",
            "kll",
            "--psi-sketch-size",
            "80",
            "--psi-random-state",
            "7",
            "--output-json",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["method"] == "psi"
    assert "x" in payload["columns"]
    assert isinstance(payload["columns"]["x"]["score"], float)


def test_cli_threshold_override_for_ks_method(tmp_path):
    baseline = tmp_path / "b.csv"
    current = tmp_path / "c.csv"
    pd.DataFrame({"x": [0, 1, 2, 3, 4]}).to_csv(baseline, index=False)
    pd.DataFrame({"x": [10, 11, 12, 13, 14]}).to_csv(current, index=False)
    runner = CliRunner()
    result = runner.invoke(
        check,
        [
            "--baseline",
            str(baseline),
            "--current",
            str(current),
            "--method",
            "ks",
            "--threshold",
            "0.01",
            "--output-json",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["schema_version"] == "1.0"
    assert payload["method"] == "ks"
    assert payload["threshold"] == 0.01


def test_cli_mmd_output_json(tmp_path):
    baseline = tmp_path / "b.csv"
    current = tmp_path / "c.csv"
    pd.DataFrame({"x1": [0, 1, 2, 3, 4], "x2": [0, 0, 1, 1, 2]}).to_csv(baseline, index=False)
    pd.DataFrame({"x1": [10, 11, 12, 13, 14], "x2": [4, 4, 5, 5, 6]}).to_csv(current, index=False)
    runner = CliRunner()
    result = runner.invoke(
        check,
        [
            "--baseline",
            str(baseline),
            "--current",
            str(current),
            "--method",
            "mmd",
            "--output-json",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["schema_version"] == "1.0"
    assert payload["method"] == "mmd"
    assert "dataset" in payload["columns"]
    assert "p_value" in payload["columns"]["dataset"]


def test_cli_cvm_output_json(tmp_path):
    baseline = tmp_path / "b.csv"
    current = tmp_path / "c.csv"
    pd.DataFrame({"x": [0, 1, 2, 3, 4]}).to_csv(baseline, index=False)
    pd.DataFrame({"x": [10, 11, 12, 13, 14]}).to_csv(current, index=False)
    runner = CliRunner()
    result = runner.invoke(
        check,
        [
            "--baseline",
            str(baseline),
            "--current",
            str(current),
            "--method",
            "cvm",
            "--output-json",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["schema_version"] == "1.0"
    assert payload["method"] == "cvm"
    assert "x" in payload["columns"]
    assert isinstance(payload["columns"]["x"]["score"], float)


def test_cli_js_output_json(tmp_path):
    baseline = tmp_path / "b.csv"
    current = tmp_path / "c.csv"
    pd.DataFrame({"x": [0, 1, 2, 3, 4]}).to_csv(baseline, index=False)
    pd.DataFrame({"x": [10, 11, 12, 13, 14]}).to_csv(current, index=False)
    runner = CliRunner()
    result = runner.invoke(
        check,
        [
            "--baseline",
            str(baseline),
            "--current",
            str(current),
            "--method",
            "js",
            "--output-json",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["schema_version"] == "1.0"
    assert payload["method"] == "js"
    assert "x" in payload["columns"]
    assert isinstance(payload["columns"]["x"]["score"], float)


def test_cli_wasserstein_output_json(tmp_path):
    baseline = tmp_path / "b.csv"
    current = tmp_path / "c.csv"
    pd.DataFrame({"x": [0, 1, 2, 3, 4]}).to_csv(baseline, index=False)
    pd.DataFrame({"x": [10, 11, 12, 13, 14]}).to_csv(current, index=False)
    runner = CliRunner()
    result = runner.invoke(
        check,
        [
            "--baseline",
            str(baseline),
            "--current",
            str(current),
            "--method",
            "wasserstein",
            "--output-json",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["schema_version"] == "1.0"
    assert payload["method"] == "wasserstein"
    assert "x" in payload["columns"]
    assert "p_value" in payload["columns"]["x"]


def test_cli_ensemble_output_json(tmp_path):
    baseline = tmp_path / "b.csv"
    current = tmp_path / "c.csv"
    pd.DataFrame({"x": [0, 1, 2, 3, 4]}).to_csv(baseline, index=False)
    pd.DataFrame({"x": [10, 11, 12, 13, 14]}).to_csv(current, index=False)
    runner = CliRunner()
    result = runner.invoke(
        check,
        [
            "--baseline",
            str(baseline),
            "--current",
            str(current),
            "--method",
            "ensemble",
            "--output-json",
            "--ensemble-methods",
            "psi,ks,cvm,js",
            "--vote-mode",
            "majority",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["schema_version"] == "1.0"
    assert payload["method"] == "ensemble"
    assert payload["threshold"] is None
    assert payload["ensemble"]["methods"] == ["psi", "ks", "cvm", "js"]
    assert payload["ensemble"]["vote_mode"] == "majority"
    assert payload["ensemble"]["stack_threshold"] == 0.5
    assert "x" in payload["columns"]
    assert "votes" in payload["columns"]["x"]
    assert "required_votes" in payload["columns"]["x"]


def test_cli_ensemble_invalid_method_fails(tmp_path):
    baseline = tmp_path / "b.csv"
    current = tmp_path / "c.csv"
    pd.DataFrame({"x": [0, 1, 2, 3, 4]}).to_csv(baseline, index=False)
    pd.DataFrame({"x": [10, 11, 12, 13, 14]}).to_csv(current, index=False)
    runner = CliRunner()
    result = runner.invoke(
        check,
        [
            "--baseline",
            str(baseline),
            "--current",
            str(current),
            "--method",
            "ensemble",
            "--ensemble-methods",
            "psi,notreal",
        ],
    )
    assert result.exit_code != 0
    assert "unknown methods" in result.output


def test_cli_ensemble_stacking_mode_output_json(tmp_path):
    baseline = tmp_path / "b.csv"
    current = tmp_path / "c.csv"
    pd.DataFrame({"x": [0, 1, 2, 3, 4]}).to_csv(baseline, index=False)
    pd.DataFrame({"x": [10, 11, 12, 13, 14]}).to_csv(current, index=False)
    runner = CliRunner()
    result = runner.invoke(
        check,
        [
            "--baseline",
            str(baseline),
            "--current",
            str(current),
            "--method",
            "ensemble",
            "--output-json",
            "--vote-mode",
            "stacking",
            "--stack-threshold",
            "0.25",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["ensemble"]["vote_mode"] == "stacking"
    assert payload["ensemble"]["stack_threshold"] == 0.25


def test_cli_json_payload_top_level_keys_compatibility(tmp_path):
    baseline = tmp_path / "b.csv"
    current = tmp_path / "c.csv"
    pd.DataFrame({"x": [0, 1, 2]}).to_csv(baseline, index=False)
    pd.DataFrame({"x": [10, 11, 12]}).to_csv(current, index=False)
    runner = CliRunner()
    result = runner.invoke(
        check,
        ["--baseline", str(baseline), "--current", str(current), "--output-json"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert set(payload.keys()) == {"schema_version", "method", "threshold", "correction", "columns"}


def test_cli_json_payload_top_level_keys_compatibility_ensemble(tmp_path):
    baseline = tmp_path / "b.csv"
    current = tmp_path / "c.csv"
    pd.DataFrame({"x": [0, 1, 2]}).to_csv(baseline, index=False)
    pd.DataFrame({"x": [10, 11, 12]}).to_csv(current, index=False)
    runner = CliRunner()
    result = runner.invoke(
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
    assert set(payload.keys()) == {
        "schema_version",
        "method",
        "threshold",
        "correction",
        "columns",
        "ensemble",
    }
    assert "stack_threshold" in payload["ensemble"]


def test_cli_correction_bonferroni_adjusts_pvalues(tmp_path):
    baseline = tmp_path / "b.csv"
    current = tmp_path / "c.csv"
    # Two columns with mild shift to exercise correction path.
    pd.DataFrame(
        {
            "x1": [0, 1, 2, 3, 4, 5, 6, 7],
            "x2": [0, 1, 2, 3, 4, 5, 6, 7],
        }
    ).to_csv(baseline, index=False)
    pd.DataFrame(
        {
            "x1": [0, 1, 2, 3, 4, 5, 6, 20],
            "x2": [0, 1, 2, 3, 4, 5, 6, 20],
        }
    ).to_csv(current, index=False)
    runner = CliRunner()
    raw_res = runner.invoke(
        check,
        [
            "--baseline",
            str(baseline),
            "--current",
            str(current),
            "--method",
            "ks",
            "--output-json",
            "--correction",
            "none",
        ],
    )
    adj_res = runner.invoke(
        check,
        [
            "--baseline",
            str(baseline),
            "--current",
            str(current),
            "--method",
            "ks",
            "--output-json",
            "--correction",
            "bonferroni",
        ],
    )
    assert raw_res.exit_code == 0
    assert adj_res.exit_code == 0
    raw = json.loads(raw_res.output)
    adj = json.loads(adj_res.output)
    assert raw["correction"] == "none"
    assert adj["correction"] == "bonferroni"
    for col in ["x1", "x2"]:
        assert adj["columns"][col]["p_value"] >= raw["columns"][col]["p_value"]


def test_cli_c2st_output_json(tmp_path):
    baseline = tmp_path / "b.csv"
    current = tmp_path / "c.csv"
    pd.DataFrame({"x1": [0, 1, 2, 3, 4], "x2": [0, 0, 1, 1, 2]}).to_csv(baseline, index=False)
    pd.DataFrame({"x1": [10, 11, 12, 13, 14], "x2": [4, 4, 5, 5, 6]}).to_csv(current, index=False)
    runner = CliRunner()
    result = runner.invoke(
        check,
        [
            "--baseline",
            str(baseline),
            "--current",
            str(current),
            "--method",
            "c2st",
            "--output-json",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["schema_version"] == "1.0"
    assert payload["method"] == "c2st"
    assert "dataset" in payload["columns"]
    assert "p_value" in payload["columns"]["dataset"]


def test_cli_energy_output_json(tmp_path):
    baseline = tmp_path / "b.csv"
    current = tmp_path / "c.csv"
    pd.DataFrame({"x1": [0, 1, 2, 3, 4], "x2": [0, 0, 1, 1, 2]}).to_csv(baseline, index=False)
    pd.DataFrame({"x1": [10, 11, 12, 13, 14], "x2": [4, 4, 5, 5, 6]}).to_csv(current, index=False)
    runner = CliRunner()
    result = runner.invoke(
        check,
        [
            "--baseline",
            str(baseline),
            "--current",
            str(current),
            "--method",
            "energy",
            "--output-json",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["schema_version"] == "1.0"
    assert payload["method"] == "energy"
    assert "dataset" in payload["columns"]
    assert "p_value" in payload["columns"]["dataset"]


def test_cli_columns_subset(tmp_path):
    baseline = tmp_path / "b.csv"
    current = tmp_path / "c.csv"
    pd.DataFrame({"x": [0, 1, 2], "y": [0, 1, 2]}).to_csv(baseline, index=False)
    pd.DataFrame({"x": [10, 11, 12], "y": [0, 1, 2]}).to_csv(current, index=False)
    runner = CliRunner()
    result = runner.invoke(
        check,
        ["--baseline", str(baseline), "--current", str(current), "--columns", "x", "--output-json"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert set(payload["columns"].keys()) == {"x"}


def test_cli_fail_on_drift_returns_nonzero(tmp_path):
    baseline = tmp_path / "b.csv"
    current = tmp_path / "c.csv"
    pd.DataFrame({"x": [0, 1, 2, 3, 4]}).to_csv(baseline, index=False)
    pd.DataFrame({"x": [10, 11, 12, 13, 14]}).to_csv(current, index=False)
    runner = CliRunner()
    result = runner.invoke(
        check,
        ["--baseline", str(baseline), "--current", str(current), "--fail-on-drift"],
    )
    assert result.exit_code != 0
    assert "Drift detected" in result.output


def test_cli_config_file_json(tmp_path):
    baseline = tmp_path / "b.csv"
    current = tmp_path / "c.csv"
    cfg = tmp_path / "drift.json"

    pd.DataFrame({"x": [0, 1, 2, 3, 4]}).to_csv(baseline, index=False)
    pd.DataFrame({"x": [10, 11, 12, 13, 14]}).to_csv(current, index=False)
    cfg.write_text(json.dumps({"method": "ks", "threshold": 0.01}), encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        check,
        [
            "--baseline",
            str(baseline),
            "--current",
            str(current),
            "--config",
            str(cfg),
            "--output-json",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["method"] == "ks"
    assert payload["threshold"] == 0.01


def test_cli_malformed_config_is_reported_without_traceback(tmp_path):
    current = tmp_path / "current.csv"
    config = tmp_path / "invalid.json"
    current.write_text("x\n1\n", encoding="utf-8")
    config.write_text("{", encoding="utf-8")

    result = CliRunner().invoke(
        check,
        ["--current", str(current), "--config", str(config)],
    )

    assert result.exit_code != 0
    assert "invalid.json" in result.output
    assert "Traceback" not in result.output


def test_cli_chi2cat_output_json(tmp_path):
    baseline = tmp_path / "b.csv"
    current = tmp_path / "c.csv"
    pd.DataFrame({"x": ["a", "a", "b", "c"]}).to_csv(baseline, index=False)
    pd.DataFrame({"x": ["c", "c", "c", "b"]}).to_csv(current, index=False)
    runner = CliRunner()
    result = runner.invoke(
        check,
        [
            "--baseline",
            str(baseline),
            "--current",
            str(current),
            "--method",
            "chi2cat",
            "--output-json",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["method"] == "chi2cat"
    assert "p_value" in payload["columns"]["x"]


def test_cli_tvdcat_output_json(tmp_path):
    baseline = tmp_path / "b.csv"
    current = tmp_path / "c.csv"
    pd.DataFrame({"x": ["a", "a", "b", "c"]}).to_csv(baseline, index=False)
    pd.DataFrame({"x": ["c", "c", "c", "b"]}).to_csv(current, index=False)
    runner = CliRunner()
    result = runner.invoke(
        check,
        [
            "--baseline",
            str(baseline),
            "--current",
            str(current),
            "--method",
            "tvdcat",
            "--output-json",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["method"] == "tvdcat"
    assert "p_value" not in payload["columns"]["x"]


def test_cli_datetime_output_json(tmp_path):
    baseline = tmp_path / "b.csv"
    current = tmp_path / "c.csv"
    pd.DataFrame({"ts": pd.date_range("2026-01-01", periods=120, freq="h").astype(str)}).to_csv(
        baseline, index=False
    )
    pd.DataFrame(
        {"ts": pd.date_range("2026-01-05 10:00:00", periods=120, freq="h").astype(str)}
    ).to_csv(current, index=False)
    runner = CliRunner()
    result = runner.invoke(
        check,
        [
            "--baseline",
            str(baseline),
            "--current",
            str(current),
            "--method",
            "datetime",
            "--output-json",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["method"] == "datetime"
    assert "ts" in payload["columns"]


def test_cli_html_report_output(tmp_path):
    baseline = tmp_path / "b.csv"
    current = tmp_path / "c.csv"
    report = tmp_path / "report.html"
    pd.DataFrame({"x": [0, 1, 2, 3, 4]}).to_csv(baseline, index=False)
    pd.DataFrame({"x": [10, 11, 12, 13, 14]}).to_csv(current, index=False)
    runner = CliRunner()
    result = runner.invoke(
        check,
        [
            "--baseline",
            str(baseline),
            "--current",
            str(current),
            "--method",
            "ks",
            "--html-report",
            str(report),
        ],
    )
    assert result.exit_code == 0
    assert report.exists()
    html = report.read_text(encoding="utf-8")
    assert "Drift Report" in html


def test_cli_markdown_report_output(tmp_path):
    baseline = tmp_path / "b.csv"
    current = tmp_path / "c.csv"
    report = tmp_path / "report.md"
    pd.DataFrame({"x": [0, 1, 2, 3, 4]}).to_csv(baseline, index=False)
    pd.DataFrame({"x": [10, 11, 12, 13, 14]}).to_csv(current, index=False)
    runner = CliRunner()
    result = runner.invoke(
        check,
        [
            "--baseline",
            str(baseline),
            "--current",
            str(current),
            "--method",
            "ks",
            "--markdown-report",
            str(report),
            "--report-top-n",
            "1",
        ],
    )
    assert result.exit_code == 0
    assert report.exists()
    md = report.read_text(encoding="utf-8")
    assert md.startswith("| Column | Score |")


def test_cli_baseline_version_loads_from_manager(tmp_path):
    baseline_dir = tmp_path / "baselines"
    manager = BaselineManager(directory=str(baseline_dir))
    manager.save_baseline(pd.DataFrame({"x": [0, 1, 2, 3, 4]}), name="mydata", version="1")
    current = tmp_path / "c.csv"
    pd.DataFrame({"x": [10, 11, 12, 13, 14]}).to_csv(current, index=False)

    runner = CliRunner()
    result = runner.invoke(
        check,
        [
            "--baseline-version",
            "mydata@1",
            "--baseline-dir",
            str(baseline_dir),
            "--current",
            str(current),
            "--output-json",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["method"] == "psi"


def test_cli_baseline_and_baseline_version_are_mutually_exclusive(tmp_path):
    baseline = tmp_path / "b.csv"
    current = tmp_path / "c.csv"
    pd.DataFrame({"x": [0, 1, 2]}).to_csv(baseline, index=False)
    pd.DataFrame({"x": [3, 4, 5]}).to_csv(current, index=False)
    runner = CliRunner()
    result = runner.invoke(
        check,
        [
            "--baseline",
            str(baseline),
            "--baseline-version",
            "mydata@1",
            "--current",
            str(current),
        ],
    )
    assert result.exit_code != 0
    assert "either --baseline or --baseline-version" in result.output


def test_cli_baseline_version_s3_store_loads(tmp_path, monkeypatch):
    current = tmp_path / "c.csv"
    pd.DataFrame({"x": [10, 11, 12, 13, 14]}).to_csv(current, index=False)

    class _FakeS3Store:
        def __init__(self, bucket: str, prefix: str):
            self.bucket = bucket
            self.prefix = prefix

        def load(self, name: str, version: str, verify_integrity: bool = True) -> pd.DataFrame:
            return pd.DataFrame({"x": [0, 1, 2, 3, 4]})

    monkeypatch.setattr(cli_mod, "S3BaselineStore", _FakeS3Store)
    runner = CliRunner()
    result = runner.invoke(
        check,
        [
            "--baseline-version",
            "mydata@1",
            "--baseline-store",
            "s3",
            "--baseline-bucket",
            "my-bucket",
            "--baseline-prefix",
            "my-prefix",
            "--current",
            str(current),
            "--output-json",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["method"] == "psi"


def test_cli_baseline_version_s3_requires_bucket(tmp_path):
    current = tmp_path / "c.csv"
    pd.DataFrame({"x": [10, 11, 12]}).to_csv(current, index=False)
    runner = CliRunner()
    result = runner.invoke(
        check,
        [
            "--baseline-version",
            "mydata@1",
            "--baseline-store",
            "s3",
            "--current",
            str(current),
        ],
    )
    assert result.exit_code != 0
    assert "--baseline-bucket is required" in result.output


def test_cli_invokes_telemetry_span(tmp_path, monkeypatch):
    baseline = tmp_path / "b.csv"
    current = tmp_path / "c.csv"
    pd.DataFrame({"x": [0, 1, 2]}).to_csv(baseline, index=False)
    pd.DataFrame({"x": [3, 4, 5]}).to_csv(current, index=False)
    calls = {"n": 0}

    class _SpanCtx:
        def __enter__(self):
            calls["n"] += 1
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _FakeTelemetry:
        def __init__(self, namespace="drift_control.cli"):
            self.namespace = namespace

        def start_span(self, name, attributes=None):
            return _SpanCtx()

        def record_error(self, attributes=None):
            return None

        def record_drift_rate(self, value, attributes=None):
            return None

        def record_latency(self, value_ms, attributes=None):
            return None

    monkeypatch.setattr(cli_mod, "DriftTelemetry", _FakeTelemetry)
    runner = CliRunner()
    result = runner.invoke(
        check,
        ["--baseline", str(baseline), "--current", str(current), "--output-json"],
    )
    assert result.exit_code == 0
    assert calls["n"] == 1


def test_cli_jobs_invalid_fails(tmp_path):
    baseline = tmp_path / "b.csv"
    current = tmp_path / "c.csv"
    pd.DataFrame({"x": [0, 1, 2]}).to_csv(baseline, index=False)
    pd.DataFrame({"x": [3, 4, 5]}).to_csv(current, index=False)
    runner = CliRunner()
    result = runner.invoke(
        check,
        ["--baseline", str(baseline), "--current", str(current), "--jobs", "0"],
    )
    assert result.exit_code != 0
    assert "--jobs must be >=" in result.output


def test_cli_jobs_parallel_matches_single(tmp_path):
    baseline = tmp_path / "b.csv"
    current = tmp_path / "c.csv"
    pd.DataFrame(
        {
            "x": [0, 1, 2, 3, 4, 5],
            "y": [10, 11, 12, 13, 14, 15],
            "z": [20, 21, 22, 23, 24, 25],
        }
    ).to_csv(baseline, index=False)
    pd.DataFrame(
        {
            "x": [5, 6, 7, 8, 9, 10],
            "y": [10, 11, 12, 13, 14, 15],
            "z": [19, 20, 21, 22, 23, 24],
        }
    ).to_csv(current, index=False)
    runner = CliRunner()
    single = runner.invoke(
        check,
        [
            "--baseline",
            str(baseline),
            "--current",
            str(current),
            "--method",
            "psi",
            "--jobs",
            "1",
            "--output-json",
        ],
    )
    parallel = runner.invoke(
        check,
        [
            "--baseline",
            str(baseline),
            "--current",
            str(current),
            "--method",
            "psi",
            "--jobs",
            "3",
            "--output-json",
        ],
    )
    assert single.exit_code == 0
    assert parallel.exit_code == 0
    p1 = json.loads(single.output)
    p2 = json.loads(parallel.output)
    assert p1["columns"] == p2["columns"]
