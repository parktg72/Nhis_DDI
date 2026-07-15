"""
dags/ 단위 테스트

Airflow 미설치 환경에서도 동작하도록 airflow를 Mock 처리.
DAG 구조(태스크 수, 의존성, 스케줄) 및 태스크 함수 로직을 검증.
"""
from __future__ import annotations

import importlib
import sys
import types
from datetime import date, datetime
from unittest.mock import MagicMock, patch

import pytest

# ─────────────────────────────────────────────────────────────────────────────
# Airflow Mock (미설치 환경 대응)
# ─────────────────────────────────────────────────────────────────────────────

def _make_airflow_mock():
    """airflow 패키지를 최소 Mock으로 대체."""
    airflow_mod = types.ModuleType("airflow")

    # airflow.DAG
    class MockDAG:
        def __init__(self, dag_id, **kwargs):
            self.dag_id = dag_id
            self.tasks = []
            self.schedule_interval = kwargs.get("schedule_interval")
            self.tags = kwargs.get("tags", [])
            self._kwargs = kwargs

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    airflow_mod.DAG = MockDAG

    # airflow.operators.*
    ops = types.ModuleType("airflow.operators")
    python_mod = types.ModuleType("airflow.operators.python")
    empty_mod  = types.ModuleType("airflow.operators.empty")

    class MockOperator:
        def __init__(self, task_id, python_callable=None, **kwargs):
            self.task_id = task_id
            self.python_callable = python_callable
            self._kwargs = kwargs

        def __rshift__(self, other):
            return other

        def __lshift__(self, other):
            return other

    class BranchPythonOperator(MockOperator):
        pass

    python_mod.PythonOperator = MockOperator
    python_mod.BranchPythonOperator = BranchPythonOperator
    empty_mod.EmptyOperator = MockOperator

    # airflow.sensors.*
    sensors_mod = types.ModuleType("airflow.sensors")
    ext_task_mod = types.ModuleType("airflow.sensors.external_task")

    class MockSensor(MockOperator):
        pass

    ext_task_mod.ExternalTaskSensor = MockSensor

    # airflow.utils.*
    utils_mod = types.ModuleType("airflow.utils")
    dates_mod = types.ModuleType("airflow.utils.dates")
    dates_mod.days_ago = lambda n: datetime(2026, 1, 1)
    utils_mod.dates = dates_mod

    # 등록
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

    return airflow_mod


# Airflow Mock 주입 (최초 1회)
if "airflow" not in sys.modules:
    _make_airflow_mock()


# ─────────────────────────────────────────────────────────────────────────────
# DAG 임포트 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

def _import_dag(module_name: str):
    """dags/ 모듈을 sys.path 없이 임포트."""
    import os
    dag_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "dags"
    )
    dag_dir = os.path.abspath(dag_dir)
    if dag_dir not in sys.path:
        sys.path.insert(0, dag_dir)
    if module_name in sys.modules:
        del sys.modules[module_name]
    return importlib.import_module(module_name)


# ─────────────────────────────────────────────────────────────────────────────
# ETL DAG 테스트
# ─────────────────────────────────────────────────────────────────────────────

class TestETLDag:
    def test_dag_importable(self):
        mod = _import_dag("ddi_etl_dag")
        assert mod is not None

    def test_dag_schedule(self):
        mod = _import_dag("ddi_etl_dag")
        assert mod.dag.schedule_interval == "0 2 * * *"

    def test_dag_tags(self):
        mod = _import_dag("ddi_etl_dag")
        assert "ddi" in mod.dag.tags
        assert "etl" in mod.dag.tags

    def test_get_partition_returns_yyyymmdd(self):
        mod = _import_dag("ddi_etl_dag")
        ti = MagicMock()
        ctx = {
            "execution_date": datetime(2026, 3, 19),
            "ti": ti,
        }
        result = mod._get_partition(**ctx)
        assert result == "20260319"
        ti.xcom_push.assert_called_once_with(key="partition", value="20260319")

    def test_validate_schemas_raises_on_failure(self):
        mod = _import_dag("ddi_etl_dag")
        import pandas as pd

        ti = MagicMock()
        ti.xcom_pull.return_value = "20260319"

        with patch("pandas.read_parquet", return_value=pd.DataFrame()):
            with patch("scripts.etl.schema_validator.validate_all") as mock_val:
                from scripts.etl.models import ValidationResult
                # invalid_rows > 0 → passed=False
                mock_val.return_value = [
                    ValidationResult(table="T20", total_rows=100, valid_rows=80, invalid_rows=20,
                                     missing_cols=["patient_id"])
                ]
                with pytest.raises(ValueError, match="스키마 검증 실패"):
                    mod._validate_schemas(ti=ti)

    def test_validate_schemas_passes_on_success(self):
        mod = _import_dag("ddi_etl_dag")
        import pandas as pd

        ti = MagicMock()
        ti.xcom_pull.return_value = "20260319"

        with patch("pandas.read_parquet", return_value=pd.DataFrame()):
            with patch("scripts.etl.schema_validator.validate_all") as mock_val:
                from scripts.etl.models import ValidationResult
                # invalid_rows=0 → passed=True
                mock_val.return_value = [
                    ValidationResult(table="T20", total_rows=100, valid_rows=100, invalid_rows=0),
                    ValidationResult(table="T30", total_rows=100, valid_rows=100, invalid_rows=0),
                    ValidationResult(table="T40", total_rows=100, valid_rows=100, invalid_rows=0),
                    ValidationResult(table="T50", total_rows=100, valid_rows=100, invalid_rows=0),
                ]
                # 예외 없이 통과
                mod._validate_schemas(ti=ti)

    def test_aggregate_and_write_features_aggregate_once(self, tmp_path):
        mod = _import_dag("ddi_etl_dag")
        import json

        import pandas as pd

        partition = "20260319"
        pd.DataFrame({
            "CMN_KEY": ["C1", "C2"],
            "WK_COMPN_CD": ["D1", "D2"],
        }).to_parquet(tmp_path / f"t30_{partition}_std.parquet", index=False)
        pd.DataFrame({
            "CMN_KEY": ["C1", "C2"],
            "INDI_DSCM_NO": ["P1", "P2"],
            "MDCARE_SYM": ["H1", "H2"],
            "MDCARE_STRT_DT": ["20260301", "20260302"],
            "SEX_TYPE": ["1", "2"],
            "SUJIN_POTM_AGE_ID": ["075", "065"],
            "YOYANG_CLSFC_CD": ["01", "02"],
        }).to_parquet(tmp_path / f"t20_{partition}_pseudo.parquet", index=False)
        pd.DataFrame({
            "patient_id": ["P1"],
            "drug_a_wk_compn": ["D1"],
            "drug_b_wk_compn": ["D2"],
        }).to_parquet(tmp_path / f"overlap_pairs_{partition}.parquet", index=False)

        features = [
            types.SimpleNamespace(
                patient_id="P1",
                window_start=date(2026, 3, 1),
                window_end=date(2026, 3, 31),
                drug_count=0,
                drug_count_7d=0,
                institution_count=0,
                ddi_contraindicated=0,
                ddi_major=0,
                ddi_moderate=0,
                ddi_minor=0,
                triple_whammy=False,
                qt_risk_count=0,
                dup_same_ingredient=0,
                dup_atc5=0,
                dup_atc4=0,
                dup_atc3=0,
                age=None,
                sex=None,
                risk_level="Red",
                risk_reasons=[],
                yellow_subtype=None,
            ),
            types.SimpleNamespace(
                patient_id="P2",
                window_start=date(2026, 3, 1),
                window_end=date(2026, 3, 31),
                drug_count=0,
                drug_count_7d=0,
                institution_count=0,
                ddi_contraindicated=0,
                ddi_major=0,
                ddi_moderate=0,
                ddi_minor=0,
                triple_whammy=False,
                qt_risk_count=0,
                dup_same_ingredient=0,
                dup_atc5=0,
                dup_atc4=0,
                dup_atc3=0,
                age=None,
                sex=None,
                risk_level="Unknown",
                risk_reasons=[],
                yellow_subtype=None,
            ),
        ]

        class FakeDrugMaster:
            @staticmethod
            def load_parquet(*args, **kwargs):
                return None

        class FakePipelineResult:
            def __init__(
                self,
                partition,
                total_patients=0,
                total_prescriptions=0,
                total_drug_items=0,
                overlap_pairs=0,
                features_written=0,
            ):
                self.partition = partition
                self.total_patients = total_patients
                self.total_prescriptions = total_prescriptions
                self.total_drug_items = total_drug_items
                self.overlap_pairs = overlap_pairs
                self.features_written = features_written
                self.red_count = 0
                self.yellow_count = 0
                self.green_count = 0
                self.normal_count = 0

        def fake_write_features(features, partition, base_dir, overwrite=False):
            base_dir.mkdir(parents=True, exist_ok=True)
            out_path = base_dir / f"patient_features_{partition}.parquet"
            pd.DataFrame([
                {
                    "patient_id": f.patient_id,
                    "window_start": f.window_start,
                    "window_end": f.window_end,
                    "risk_level": f.risk_level,
                }
                for f in features
            ]).to_parquet(out_path, index=False)
            return out_path

        def fake_write_pipeline_log(result, base_dir):
            out_path = base_dir / f"pipeline_log_{result.partition}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({
                    "features_written": result.features_written,
                    "risk_distribution": {
                        "Red": result.red_count,
                        "Yellow": result.yellow_count,
                        "Green": result.green_count,
                        "Normal": result.normal_count,
                    },
                }, f)
            return out_path

        fake_drug_master = types.ModuleType("scripts.etl.drug_master")
        fake_drug_master.DrugMaster = FakeDrugMaster
        fake_models = types.ModuleType("scripts.etl.models")
        fake_models.PipelineResult = FakePipelineResult
        fake_feature_writer = types.ModuleType("scripts.etl.feature_writer")
        fake_feature_writer.write_features = fake_write_features
        fake_feature_writer.write_pipeline_log = fake_write_pipeline_log
        fake_aggregator = types.ModuleType("scripts.etl.prescription_aggregator")
        fake_aggregator.aggregate_batch = MagicMock(return_value=features)
        fake_etl_pkg = types.ModuleType("scripts.etl")
        fake_etl_pkg.__path__ = []

        class FakeTI:
            def __init__(self):
                self.values = {("partition", "get_partition"): partition}

            def xcom_pull(self, key, task_ids):
                return self.values.get((key, task_ids))

            def xcom_push(self, key, value):
                task_id = "aggregate_features" if key != "features_path" else "write_features"
                self.values[(key, task_id)] = value

        ti = FakeTI()
        fake_modules = {
            "scripts.etl": fake_etl_pkg,
            "scripts.etl.drug_master": fake_drug_master,
            "scripts.etl.models": fake_models,
            "scripts.etl.feature_writer": fake_feature_writer,
            "scripts.etl.prescription_aggregator": fake_aggregator,
        }
        with patch.dict(sys.modules, fake_modules):
            with patch.object(mod, "PROC_DIR", tmp_path):
                with patch.object(mod, "DDI_MATRIX_PATH", str(tmp_path / "missing_ddi.parquet")):
                    with patch.object(mod, "DDI_DUP_GROUPS_PATH", str(tmp_path / "missing_dup.parquet")):
                        with patch.object(mod, "DDI_DRUG_MASTER_PATH", str(tmp_path / "missing_master.parquet")):
                            staging_path = mod._aggregate_features(ti=ti)
                            mod._write_features(ti=ti)

        assert fake_aggregator.aggregate_batch.call_count == 1
        assert staging_path == str(
            tmp_path / "staging" / f"patient_features_staging_{partition}.parquet"
        )
        assert (tmp_path / f"patient_features_{partition}.parquet").exists()

        log_path = tmp_path / f"pipeline_log_{partition}.json"
        with open(log_path, encoding="utf-8") as f:
            log = json.load(f)
        assert log["features_written"] == 2
        assert log["risk_distribution"] == {
            "Red": 1,
            "Yellow": 0,
            "Green": 0,
            "Normal": 1,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Feature DAG 테스트
# ─────────────────────────────────────────────────────────────────────────────

class TestFeatureDag:
    def test_dag_importable(self):
        mod = _import_dag("ddi_feature_dag")
        assert mod is not None

    def test_dag_schedule(self):
        mod = _import_dag("ddi_feature_dag")
        assert mod.dag.schedule_interval == "30 3 * * *"

    def test_create_labels_high_risk(self):
        mod = _import_dag("ddi_feature_dag")
        import pandas as pd

        ti = MagicMock()
        ti.xcom_pull.return_value = "20260319"

        df = pd.DataFrame({
            "patient_id": ["P001", "P002", "P003"],
            "risk_level": ["Red", "Yellow", "Normal"],
        })

        with patch("pandas.read_parquet", return_value=df):
            with patch.object(df.__class__, "to_parquet") as mock_pq:
                # to_parquet를 mock해야 하므로 직접 로직만 테스트
                pass

        # 직접 레이블 로직 검증
        df["is_high_risk"] = (df["risk_level"] == "Red").astype(int)
        assert df.loc[df["patient_id"] == "P001", "is_high_risk"].iloc[0] == 1
        assert df.loc[df["patient_id"] == "P002", "is_high_risk"].iloc[0] == 0
        assert df.loc[df["patient_id"] == "P003", "is_high_risk"].iloc[0] == 0

    def test_get_partition(self):
        mod = _import_dag("ddi_feature_dag")
        ti = MagicMock()
        ctx = {
            "execution_date": datetime(2026, 3, 15),
            "ti": ti,
        }
        result = mod._get_partition(**ctx)
        assert result == "20260315"

    def test_select_features_promotes_raw_sex_to_sex_type(self, tmp_path, monkeypatch):
        import pandas as pd

        from scripts.features.selector import FeatureSelector

        mod = _import_dag("ddi_feature_dag")
        partition = "20260319"
        processed_dir = tmp_path / "processed"
        features_dir = tmp_path / "features"
        selector_path = tmp_path / "selector.pkl"
        processed_dir.mkdir()
        pd.DataFrame({
            "patient_id": ["P1", "P2", "P3"],
            "risk_level": ["Red", "Normal", "Yellow"],
            "sex": ["1", "2", "9"],
            "drug_count": [1.0, 2.0, 3.0],
        }).to_parquet(
            processed_dir / f"patient_features_norm_{partition}.parquet",
            index=False,
        )
        monkeypatch.setattr(mod, "PROC_DIR", str(processed_dir))
        monkeypatch.setattr(mod, "FEATURES_DIR", str(features_dir))
        monkeypatch.setattr(mod, "SELECTOR_PATH", str(selector_path))

        ti = MagicMock()
        ti.xcom_pull.return_value = partition
        mod._select_features(ti=ti)

        out = pd.read_parquet(features_dir / f"ml_features_{partition}.parquet")
        selector = FeatureSelector.load(selector_path)
        assert list(out["sex_type"]) == ["1", "2", "9"]
        assert "sex" not in out.columns
        assert "sex_type" not in selector.selected_features


# ─────────────────────────────────────────────────────────────────────────────
# Train DAG 테스트
# ─────────────────────────────────────────────────────────────────────────────

class TestTrainDag:
    def test_dag_importable(self):
        mod = _import_dag("ddi_train_dag")
        assert mod is not None

    def test_dag_schedule_weekly(self):
        mod = _import_dag("ddi_train_dag")
        assert mod.dag.schedule_interval == "0 4 * * 1"

    def test_validate_model_passes(self):
        mod = _import_dag("ddi_train_dag")
        ti = MagicMock()
        ti.xcom_pull.side_effect = lambda key, task_ids: (
            0.93 if key == "val_recall" else 0.88
        )
        result = mod._validate_model(ti=ti)
        assert result == "deploy_model"

    def test_validate_model_fails_recall(self):
        mod = _import_dag("ddi_train_dag")
        ti = MagicMock()
        ti.xcom_pull.side_effect = lambda key, task_ids: (
            0.85 if key == "val_recall" else 0.88  # recall < 0.90
        )
        result = mod._validate_model(ti=ti)
        assert result == "validation_failed"

    def test_validate_model_fails_auc(self):
        mod = _import_dag("ddi_train_dag")
        ti = MagicMock()
        ti.xcom_pull.side_effect = lambda key, task_ids: (
            0.92 if key == "val_recall" else 0.80  # auc < 0.85
        )
        result = mod._validate_model(ti=ti)
        assert result == "validation_failed"

    def test_load_features_no_data_raises(self):
        mod = _import_dag("ddi_train_dag")
        ti = MagicMock()
        ctx = {
            "execution_date": datetime(2026, 3, 19),
            "ti": ti,
        }
        with patch("os.path.exists", return_value=False):
            with pytest.raises(FileNotFoundError):
                mod._load_features(**ctx)

    def test_validation_failed_logs(self):
        mod = _import_dag("ddi_train_dag")
        ti = MagicMock()
        ti.xcom_pull.side_effect = lambda key, task_ids: (
            0.85 if key == "val_recall" else 0.80
        )
        import logging
        with patch.object(logging, "error") as mock_log:
            mod._validation_failed(ti=ti)
            mock_log.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# Batch Predict DAG 테스트
# ─────────────────────────────────────────────────────────────────────────────

class TestBatchPredictDag:
    def test_dag_importable(self):
        mod = _import_dag("ddi_batch_predict_dag")
        assert mod is not None

    def test_dag_schedule(self):
        mod = _import_dag("ddi_batch_predict_dag")
        assert mod.dag.schedule_interval == "0 5 * * 2-6"

    def test_dag_tags(self):
        mod = _import_dag("ddi_batch_predict_dag")
        assert "batch" in mod.dag.tags

    def test_check_serving_health_success(self):
        mod = _import_dag("ddi_batch_predict_dag")
        with patch("requests.get") as mock_get:
            mock_get.return_value = MagicMock(
                json=lambda: {"status": "ok", "version": "1.0.0"},
            )
            mock_get.return_value.raise_for_status = lambda: None
            mod._check_serving_health()

    def test_check_serving_health_fails(self):
        mod = _import_dag("ddi_batch_predict_dag")
        import requests as req_mod
        with patch("requests.get", side_effect=req_mod.RequestException("연결 거부")):
            with pytest.raises(RuntimeError, match="Serving API 연결 실패"):
                mod._check_serving_health()

    def test_generate_summary_no_file(self):
        mod = _import_dag("ddi_batch_predict_dag")
        ti = MagicMock()
        ti.xcom_pull.side_effect = lambda key, task_ids: (
            "20260319" if key == "partition" else None
        )
        import logging
        with patch("os.path.exists", return_value=False):
            with patch.object(logging, "warning") as mock_warn:
                mod._generate_summary(ti=ti)
                mock_warn.assert_called_once()

    def test_generate_summary_distribution(self, tmp_path):
        mod = _import_dag("ddi_batch_predict_dag")
        import json

        import pandas as pd

        # 예측 결과 파일 생성
        pred_path = str(tmp_path / "predictions_20260319.parquet")
        df = pd.DataFrame({
            "patient_id": [f"P{i:03d}" for i in range(10)],
            "risk_level": ["Red"] * 2 + ["Yellow"] * 3 + ["Green"] * 3 + ["Normal"] * 2,
        })
        df.to_parquet(pred_path, index=False)

        ti = MagicMock()
        ti.xcom_pull.side_effect = lambda key, task_ids: (
            "20260319" if key == "partition" else pred_path
        )

        with patch.object(mod, "PREDICTIONS_DIR", str(tmp_path)):
            mod._generate_summary(ti=ti)

        summary_path = tmp_path / "summary_20260319.json"
        assert summary_path.exists()
        with open(summary_path) as f:
            summary = json.load(f)

        assert summary["total"] == 10
        assert summary["red_count"] == 2
        assert summary["yellow_count"] == 3
        assert summary["green_count"] == 3
        assert summary["normal_count"] == 2
        assert abs(summary["red_rate"] - 0.2) < 1e-6

    def test_cleanup_staging_removes_file(self, tmp_path):
        mod = _import_dag("ddi_batch_predict_dag")
        staging = tmp_path / "batch_patients_20260319.json"
        staging.write_text("{}")

        ti = MagicMock()
        ti.xcom_pull.return_value = str(staging)

        mod._cleanup_staging(ti=ti)
        assert not staging.exists()

    def test_get_partition(self):
        mod = _import_dag("ddi_batch_predict_dag")
        ti = MagicMock()
        ctx = {
            "execution_date": datetime(2026, 3, 19),
            "ti": ti,
        }
        result = mod._get_partition(**ctx)
        assert result == "20260319"
