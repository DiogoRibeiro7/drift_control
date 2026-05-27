import json

from click.testing import CliRunner
import pandas as pd
from drift_control.cli import check


def test_cli_runs(tmp_path):
    baseline = tmp_path / 'b.csv'
    current = tmp_path / 'c.csv'
    pd.DataFrame({'x': [0, 1, 2]}).to_csv(baseline, index=False)
    pd.DataFrame({'x': [3, 4, 5]}).to_csv(current, index=False)
    runner = CliRunner()
    result = runner.invoke(check, ['--baseline', str(baseline), '--current', str(current)])
    assert result.exit_code == 0
    assert 'x:' in result.output


def test_cli_schema_mismatch_fails(tmp_path):
    baseline = tmp_path / 'b.csv'
    current = tmp_path / 'c.csv'
    pd.DataFrame({'x': [0, 1, 2]}).to_csv(baseline, index=False)
    pd.DataFrame({'y': [3, 4, 5]}).to_csv(current, index=False)
    runner = CliRunner()
    result = runner.invoke(check, ['--baseline', str(baseline), '--current', str(current)])
    assert result.exit_code != 0
    assert 'Schema mismatch' in result.output


def test_cli_non_numeric_column_fails(tmp_path):
    baseline = tmp_path / 'b.csv'
    current = tmp_path / 'c.csv'
    pd.DataFrame({'x': ['a', 'b', 'c']}).to_csv(baseline, index=False)
    pd.DataFrame({'x': ['d', 'e', 'f']}).to_csv(current, index=False)
    runner = CliRunner()
    result = runner.invoke(check, ['--baseline', str(baseline), '--current', str(current)])
    assert result.exit_code != 0
    assert "must be numeric" in result.output


def test_cli_output_json_emits_structured_payload(tmp_path):
    baseline = tmp_path / 'b.csv'
    current = tmp_path / 'c.csv'
    pd.DataFrame({'x': [0, 1, 2, 3, 4]}).to_csv(baseline, index=False)
    pd.DataFrame({'x': [10, 11, 12, 13, 14]}).to_csv(current, index=False)
    runner = CliRunner()
    result = runner.invoke(
        check,
        ['--baseline', str(baseline), '--current', str(current), '--output-json'],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload['schema_version'] == '1.0'
    assert payload['method'] == 'psi'
    assert payload['threshold'] == 0.2
    assert 'x' in payload['columns']
    assert payload['columns']['x']['drift'] is True
    assert isinstance(payload['columns']['x']['score'], float)


def test_cli_threshold_override_changes_drift_flag(tmp_path):
    baseline = tmp_path / 'b.csv'
    current = tmp_path / 'c.csv'
    pd.DataFrame({'x': [0, 1, 2, 3, 4]}).to_csv(baseline, index=False)
    pd.DataFrame({'x': [10, 11, 12, 13, 14]}).to_csv(current, index=False)
    runner = CliRunner()
    # Same data, but a huge threshold should suppress the drift flag.
    result = runner.invoke(
        check,
        [
            '--baseline', str(baseline), '--current', str(current),
            '--threshold', '1000.0', '--output-json',
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload['schema_version'] == '1.0'
    assert payload['threshold'] == 1000.0
    assert payload['columns']['x']['drift'] is False


def test_cli_threshold_override_for_ks_method(tmp_path):
    baseline = tmp_path / 'b.csv'
    current = tmp_path / 'c.csv'
    pd.DataFrame({'x': [0, 1, 2, 3, 4]}).to_csv(baseline, index=False)
    pd.DataFrame({'x': [10, 11, 12, 13, 14]}).to_csv(current, index=False)
    runner = CliRunner()
    result = runner.invoke(
        check,
        [
            '--baseline', str(baseline), '--current', str(current),
            '--method', 'ks', '--threshold', '0.01', '--output-json',
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload['schema_version'] == '1.0'
    assert payload['method'] == 'ks'
    assert payload['threshold'] == 0.01


def test_cli_mmd_output_json(tmp_path):
    baseline = tmp_path / 'b.csv'
    current = tmp_path / 'c.csv'
    pd.DataFrame({'x1': [0, 1, 2, 3, 4], 'x2': [0, 0, 1, 1, 2]}).to_csv(baseline, index=False)
    pd.DataFrame({'x1': [10, 11, 12, 13, 14], 'x2': [4, 4, 5, 5, 6]}).to_csv(current, index=False)
    runner = CliRunner()
    result = runner.invoke(
        check,
        [
            '--baseline', str(baseline), '--current', str(current),
            '--method', 'mmd', '--output-json',
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload['schema_version'] == '1.0'
    assert payload['method'] == 'mmd'
    assert 'dataset' in payload['columns']
    assert 'p_value' in payload['columns']['dataset']


def test_cli_cvm_output_json(tmp_path):
    baseline = tmp_path / 'b.csv'
    current = tmp_path / 'c.csv'
    pd.DataFrame({'x': [0, 1, 2, 3, 4]}).to_csv(baseline, index=False)
    pd.DataFrame({'x': [10, 11, 12, 13, 14]}).to_csv(current, index=False)
    runner = CliRunner()
    result = runner.invoke(
        check,
        [
            '--baseline', str(baseline), '--current', str(current),
            '--method', 'cvm', '--output-json',
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload['schema_version'] == '1.0'
    assert payload['method'] == 'cvm'
    assert 'x' in payload['columns']
    assert isinstance(payload['columns']['x']['score'], float)


def test_cli_js_output_json(tmp_path):
    baseline = tmp_path / 'b.csv'
    current = tmp_path / 'c.csv'
    pd.DataFrame({'x': [0, 1, 2, 3, 4]}).to_csv(baseline, index=False)
    pd.DataFrame({'x': [10, 11, 12, 13, 14]}).to_csv(current, index=False)
    runner = CliRunner()
    result = runner.invoke(
        check,
        [
            '--baseline', str(baseline), '--current', str(current),
            '--method', 'js', '--output-json',
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload['schema_version'] == '1.0'
    assert payload['method'] == 'js'
    assert 'x' in payload['columns']
    assert isinstance(payload['columns']['x']['score'], float)


def test_cli_wasserstein_output_json(tmp_path):
    baseline = tmp_path / 'b.csv'
    current = tmp_path / 'c.csv'
    pd.DataFrame({'x': [0, 1, 2, 3, 4]}).to_csv(baseline, index=False)
    pd.DataFrame({'x': [10, 11, 12, 13, 14]}).to_csv(current, index=False)
    runner = CliRunner()
    result = runner.invoke(
        check,
        [
            '--baseline', str(baseline), '--current', str(current),
            '--method', 'wasserstein', '--output-json',
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload['schema_version'] == '1.0'
    assert payload['method'] == 'wasserstein'
    assert 'x' in payload['columns']
    assert 'p_value' in payload['columns']['x']


def test_cli_ensemble_output_json(tmp_path):
    baseline = tmp_path / 'b.csv'
    current = tmp_path / 'c.csv'
    pd.DataFrame({'x': [0, 1, 2, 3, 4]}).to_csv(baseline, index=False)
    pd.DataFrame({'x': [10, 11, 12, 13, 14]}).to_csv(current, index=False)
    runner = CliRunner()
    result = runner.invoke(
        check,
        [
            '--baseline', str(baseline), '--current', str(current),
            '--method', 'ensemble', '--output-json',
            '--ensemble-methods', 'psi,ks,cvm,js',
            '--vote-mode', 'majority',
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload['schema_version'] == '1.0'
    assert payload['method'] == 'ensemble'
    assert payload['threshold'] is None
    assert payload['ensemble']['methods'] == ['psi', 'ks', 'cvm', 'js']
    assert payload['ensemble']['vote_mode'] == 'majority'
    assert 'x' in payload['columns']
    assert 'votes' in payload['columns']['x']
    assert 'required_votes' in payload['columns']['x']


def test_cli_ensemble_invalid_method_fails(tmp_path):
    baseline = tmp_path / 'b.csv'
    current = tmp_path / 'c.csv'
    pd.DataFrame({'x': [0, 1, 2, 3, 4]}).to_csv(baseline, index=False)
    pd.DataFrame({'x': [10, 11, 12, 13, 14]}).to_csv(current, index=False)
    runner = CliRunner()
    result = runner.invoke(
        check,
        [
            '--baseline', str(baseline), '--current', str(current),
            '--method', 'ensemble',
            '--ensemble-methods', 'psi,notreal',
        ],
    )
    assert result.exit_code != 0
    assert 'unknown methods' in result.output


def test_cli_json_payload_top_level_keys_compatibility(tmp_path):
    baseline = tmp_path / 'b.csv'
    current = tmp_path / 'c.csv'
    pd.DataFrame({'x': [0, 1, 2]}).to_csv(baseline, index=False)
    pd.DataFrame({'x': [10, 11, 12]}).to_csv(current, index=False)
    runner = CliRunner()
    result = runner.invoke(
        check,
        ['--baseline', str(baseline), '--current', str(current), '--output-json'],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert set(payload.keys()) == {'schema_version', 'method', 'threshold', 'columns'}


def test_cli_json_payload_top_level_keys_compatibility_ensemble(tmp_path):
    baseline = tmp_path / 'b.csv'
    current = tmp_path / 'c.csv'
    pd.DataFrame({'x': [0, 1, 2]}).to_csv(baseline, index=False)
    pd.DataFrame({'x': [10, 11, 12]}).to_csv(current, index=False)
    runner = CliRunner()
    result = runner.invoke(
        check,
        [
            '--baseline', str(baseline), '--current', str(current),
            '--method', 'ensemble', '--output-json',
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert set(payload.keys()) == {'schema_version', 'method', 'threshold', 'columns', 'ensemble'}


def test_cli_c2st_output_json(tmp_path):
    baseline = tmp_path / 'b.csv'
    current = tmp_path / 'c.csv'
    pd.DataFrame({'x1': [0, 1, 2, 3, 4], 'x2': [0, 0, 1, 1, 2]}).to_csv(baseline, index=False)
    pd.DataFrame({'x1': [10, 11, 12, 13, 14], 'x2': [4, 4, 5, 5, 6]}).to_csv(current, index=False)
    runner = CliRunner()
    result = runner.invoke(
        check,
        [
            '--baseline', str(baseline), '--current', str(current),
            '--method', 'c2st', '--output-json',
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload['schema_version'] == '1.0'
    assert payload['method'] == 'c2st'
    assert 'dataset' in payload['columns']
    assert 'p_value' in payload['columns']['dataset']
