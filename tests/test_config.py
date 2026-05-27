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
        ensemble_methods='',
        vote_mode='majority',
        min_votes=None,
    )
    assert cfg.ensemble.methods == ['psi', 'ks', 'cvm', 'js']


def test_drift_check_config_rejects_unknown_method():
    with pytest.raises(ValueError, match='unsupported method'):
        DriftCheckConfig(method='bad')
