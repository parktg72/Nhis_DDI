"""
test_ddi_train_dag.py — _run_training 시그니처 버그 회귀 테스트

run_training(df, config) 대신 run_training(partition=str, ...) 로 호출되는지 확인.
"""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Airflow Mock (미설치 환경 대응)
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_airflow_mock():
    if "airflow" in sys.modules:
        return

    airflow_mod = types.ModuleType("airflow")

    class MockDAG:
        def __init__(self, dag_id, **kwargs):
            self.dag_id = dag_id
            self.tasks = []
            self.schedule_interval = kwargs.get("schedule_interval")
            self.tags = kwargs.get("tags", [])

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    airflow_mod.DAG = MockDAG

    ops = types.ModuleType("airflow.operators")
    python_mod = types.ModuleType("airflow.operators.python")
    empty_mod = types.ModuleType("airflow.operators.empty")

    class MockOperator:
        def __init__(self, task_id, python_callable=None, **kwargs):
            self.task_id = task_id
            self.python_callable = python_callable

        def __rshift__(self, other):
            return other

        def __lshift__(self, other):
            return other

    class BranchPythonOperator(MockOperator):
        pass

    python_mod.PythonOperator = MockOperator
    python_mod.BranchPythonOperator = BranchPythonOperator
    empty_mod.EmptyOperator = MockOperator

    sensors_mod = types.ModuleType("airflow.sensors")
    ext_task_mod = types.ModuleType("airflow.sensors.external_task")

    class MockSensor(MockOperator):
        pass

    ext_task_mod.ExternalTaskSensor = MockSensor

    from datetime import datetime
    utils_mod = types.ModuleType("airflow.utils")
    dates_mod = types.ModuleType("airflow.utils.dates")
    dates_mod.days_ago = lambda n: datetime(2026, 1, 1)
    utils_mod.dates = dates_mod

    for name, mod in [
        ("airflow", airflow_mod),
        ("airflow.operators", ops),
        ("airflow.operators.python", python_mod),
        ("airflow.operators.empty", empty_mod),
        ("airflow.sensors", sensors_mod),
        ("airflow.sensors.external_task", ext_task_mod),
        ("airflow.utils", utils_mod),
        ("airflow.utils.dates", dates_mod),
    ]:
        sys.modules[name] = mod


_ensure_airflow_mock()


# ─────────────────────────────────────────────────────────────────────────────
# 테스트
# ─────────────────────────────────────────────────────────────────────────────

def test_run_training_calls_with_partition_str(monkeypatch):
    """_run_training must call run_training(partition=str, ...) not run_training(df, config)."""
    captured = {}

    def fake_run_training(partition, **kwargs):
        captured["partition"] = partition
        captured["kwargs"] = kwargs
        # Return a minimal TrainResult
        from scripts.train.pipeline import TrainResult
        from scripts.train.evaluator import EvalResult
        r = TrainResult(partition=partition, model_type="xgboost")
        r.model_path = "/tmp/model.pkl"
        r.eval_results = {"val": EvalResult("val", recall=0.95, auc_roc=0.90)}
        r.passed = True
        return r

    monkeypatch.setattr("scripts.train.pipeline.run_training", fake_run_training)

    class FakeTI:
        def xcom_pull(self, key, task_ids):
            return None
        def xcom_push(self, key, value):
            pass

    import sys
    # Force reimport after monkeypatch
    if "dags.ddi_train_dag" in sys.modules:
        del sys.modules["dags.ddi_train_dag"]
    if "ddi_train_dag" in sys.modules:
        del sys.modules["ddi_train_dag"]

    from dags.ddi_train_dag import _run_training
    _run_training(ti=FakeTI())

    assert isinstance(captured.get("partition"), str), (
        f"run_training first arg must be str, got: {type(captured.get('partition'))}"
    )
    assert captured["partition"] == "staging"
