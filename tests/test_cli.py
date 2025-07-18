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
