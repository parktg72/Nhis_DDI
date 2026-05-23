from __future__ import annotations

import importlib

from tests.test_dags.test_dags import _make_airflow_mock


def test_all_dags_import_with_airflow_mock() -> None:
    _make_airflow_mock()
    for module_name in [
        "dags.ddi_etl_dag",
        "dags.ddi_feature_dag",
        "dags.ddi_train_dag",
        "dags.ddi_batch_predict_dag",
    ]:
        mod = importlib.import_module(module_name)
        assert mod is not None
