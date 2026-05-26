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
