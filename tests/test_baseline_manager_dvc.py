import pandas as pd
import subprocess
import shutil
from drift_control.baseline_manager import BaselineManager


def test_save_with_dvc(monkeypatch, tmp_path):
    data = pd.DataFrame({'a': [1]})
    manager = BaselineManager(directory=tmp_path)
    called = {}
    monkeypatch.setattr(shutil, 'which', lambda x: 'dvc')
    def fake_run(cmd, check):
        called['cmd'] = cmd
    monkeypatch.setattr(subprocess, 'run', fake_run)
    path = manager.save_with_dvc(data, 'x', '1')
    assert 'dvc' in called['cmd'][0]
    assert path.endswith('x_v1.csv')
