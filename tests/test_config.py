import json

import pytest

from drift_control.config import DriftCheckConfig, EnsembleConfig


def test_ensemble_config_rejects_unknown_method():
    with pytest.raises(ValueError, match='unknown methods'):
        EnsembleConfig(methods=['psi', 'bad'])


def test_ensemble_config_rejects_invalid_vote_mode():
    with pytest.raises(ValueError, match='vote_mode'):
        EnsembleConfig(methods=['psi'], vote_mode='weird')


def test_drift_check_config_from_cli_defaults_methods():
    cfg = DriftCheckConfig.from_cli(
        method='ensemble',
        threshold=None,
        correction='none',
        ensemble_methods='',
        vote_mode='majority',
        min_votes=None,
    )
    assert cfg.ensemble.methods == ['psi', 'ks', 'cvm', 'js']


def test_drift_check_config_rejects_unknown_method():
    with pytest.raises(ValueError, match='unsupported method'):
        DriftCheckConfig(method='bad')
import json


def test_drift_check_config_from_file_json(tmp_path):
    cfg_path = tmp_path / 'drift.json'
    cfg_path.write_text(json.dumps({
        'method': 'ensemble',
        'threshold': None,
        'ensemble': {
            'methods': ['psi', 'ks'],
            'vote_mode': 'any',
            'min_votes': 1,
        },
    }), encoding='utf-8')

    cfg = DriftCheckConfig.from_file(cfg_path)
    assert cfg.method == 'ensemble'
    assert cfg.ensemble.methods == ['psi', 'ks']
    assert cfg.ensemble.vote_mode == 'any'
    assert cfg.ensemble.min_votes == 1


def test_drift_check_config_from_env(monkeypatch):
    monkeypatch.setenv('DRIFT_CONTROL_METHOD', 'ks')
    monkeypatch.setenv('DRIFT_CONTROL_THRESHOLD', '0.01')
    cfg = DriftCheckConfig.from_env()
    assert cfg.method == 'ks'
    assert cfg.threshold == 0.01


def test_config_accepts_categorical_methods():
    cfg = DriftCheckConfig.from_cli(
        method='chi2cat',
        threshold=0.05,
        correction='none',
        ensemble_methods='chi2cat,tvdcat',
        vote_mode='any',
        min_votes=1,
    )
    assert cfg.method == 'chi2cat'
    assert cfg.ensemble.methods == ['chi2cat', 'tvdcat']
