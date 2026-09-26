import json

import pytest
from dataexcept import DataLoadingError

from drift_control.config import DriftCheckConfig, EnsembleConfig


@pytest.mark.parametrize(
    ("suffix", "content", "cause"),
    [
        (".json", "{", json.JSONDecodeError),
        (".toml", "method = [", ValueError),
        (".yaml", "method: [", Exception),
    ],
)
def test_malformed_config_reports_source(tmp_path, suffix, content, cause):
    if suffix == ".yaml":
        yaml = pytest.importorskip("yaml")
        cause = yaml.YAMLError
    path = tmp_path / f"config{suffix}"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(DataLoadingError, match=f"config{suffix}") as error:
        DriftCheckConfig.from_file(path)

    assert isinstance(error.value.__cause__, cause)


def test_missing_config_reports_source(tmp_path):
    path = tmp_path / "missing.json"
    with pytest.raises(DataLoadingError, match="missing.json") as error:
        DriftCheckConfig.from_file(path)
    assert isinstance(error.value.__cause__, FileNotFoundError)


def test_config_validation_remains_value_error(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text('{"method": "unknown"}', encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported method"):
        DriftCheckConfig.from_file(path)


def test_ensemble_config_rejects_unknown_method():
    with pytest.raises(ValueError, match="unknown methods"):
        EnsembleConfig(methods=["psi", "bad"])


def test_ensemble_config_rejects_invalid_vote_mode():
    with pytest.raises(ValueError, match="vote_mode"):
        EnsembleConfig(methods=["psi"], vote_mode="weird")


def test_drift_check_config_from_cli_defaults_methods():
    cfg = DriftCheckConfig.from_cli(
        method="ensemble",
        threshold=None,
        correction="none",
        ensemble_methods="",
        vote_mode="majority",
        min_votes=None,
    )
    assert cfg.ensemble.methods == ["psi", "ks", "cvm", "js"]


def test_drift_check_config_rejects_unknown_method():
    with pytest.raises(ValueError, match="unsupported method"):
        DriftCheckConfig(method="bad")


def test_drift_check_config_from_file_json(tmp_path):
    cfg_path = tmp_path / "drift.json"
    cfg_path.write_text(
        json.dumps(
            {
                "method": "ensemble",
                "threshold": None,
                "ensemble": {
                    "methods": ["psi", "ks"],
                    "vote_mode": "any",
                    "min_votes": 1,
                },
            }
        ),
        encoding="utf-8",
    )

    cfg = DriftCheckConfig.from_file(cfg_path)
    assert cfg.method == "ensemble"
    assert cfg.ensemble.methods == ["psi", "ks"]
    assert cfg.ensemble.vote_mode == "any"
    assert cfg.ensemble.min_votes == 1


def test_drift_check_config_from_env(monkeypatch):
    monkeypatch.setenv("DRIFT_CONTROL_METHOD", "ks")
    monkeypatch.setenv("DRIFT_CONTROL_THRESHOLD", "0.01")
    cfg = DriftCheckConfig.from_env()
    assert cfg.method == "ks"
    assert cfg.threshold == 0.01


def test_config_accepts_categorical_methods():
    cfg = DriftCheckConfig.from_cli(
        method="chi2cat",
        threshold=0.05,
        correction="none",
        ensemble_methods="chi2cat,tvdcat",
        vote_mode="any",
        min_votes=1,
    )
    assert cfg.method == "chi2cat"
    assert cfg.ensemble.methods == ["chi2cat", "tvdcat"]
