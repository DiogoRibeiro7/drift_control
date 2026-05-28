import sys
import types

import pytest

from drift_control.integrations.airflow import create_airflow_drift_task
from drift_control.integrations.prefect import create_prefect_drift_task


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


def test_create_prefect_drift_task_missing_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "prefect", None)
    with pytest.raises(ImportError, match="prefect"):
        create_prefect_drift_task(args=["--help"])


def test_create_prefect_drift_task_with_fake_module(monkeypatch):
    def _fake_task(name=None):
        def _decorator(fn):
            fn._task_name = name
            return fn

        return _decorator

    fake_prefect = types.SimpleNamespace(task=_fake_task)
    monkeypatch.setitem(sys.modules, "prefect", fake_prefect)

    task_fn = create_prefect_drift_task(args=["--help"], name="dc-task")
    assert getattr(task_fn, "_task_name") == "dc-task"
