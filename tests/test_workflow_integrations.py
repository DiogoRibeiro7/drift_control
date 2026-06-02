import sys
import types

import pytest

from drift_control.integrations.airflow import create_airflow_drift_task


def test_create_airflow_drift_task_missing_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "airflow", None)
    monkeypatch.setitem(sys.modules, "airflow.operators", None)
    monkeypatch.setitem(sys.modules, "airflow.operators.python", None)
    with pytest.raises(ImportError, match="apache-airflow"):
        create_airflow_drift_task("drift_check", args=["--help"])


def test_create_airflow_drift_task_with_fake_module(monkeypatch):
    class _FakePythonOperator:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_python_mod = types.SimpleNamespace(PythonOperator=_FakePythonOperator)
    fake_operators_mod = types.SimpleNamespace(python=fake_python_mod)
    fake_airflow = types.SimpleNamespace(operators=fake_operators_mod)
    monkeypatch.setitem(sys.modules, "airflow", fake_airflow)
    monkeypatch.setitem(sys.modules, "airflow.operators", fake_operators_mod)
    monkeypatch.setitem(sys.modules, "airflow.operators.python", fake_python_mod)

    op = create_airflow_drift_task("drift_check", args=["--help"])
    assert op.kwargs["task_id"] == "drift_check"
    assert op.kwargs["op_kwargs"]["args"] == ["--help"]
