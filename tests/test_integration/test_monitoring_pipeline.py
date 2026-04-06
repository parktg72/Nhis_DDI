"""모니터링 파이프라인 통합 테스트."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# ─────────────────────────────────────────────────────────────────────────────
# Airflow Mock (미설치 환경 대응)
# ─────────────────────────────────────────────────────────────────────────────

import sys, types

def _ensure_airflow_mock():
    if "airflow" in sys.modules:
        return
    airflow_mod = types.ModuleType("airflow")
    class MockDAG:
        def __init__(self, dag_id, **kwargs): self.dag_id = dag_id
        def __enter__(self): return self
        def __exit__(self, *a): pass
    airflow_mod.DAG = MockDAG
    ops = types.ModuleType("airflow.operators")
    python_mod = types.ModuleType("airflow.operators.python")
    empty_mod = types.ModuleType("airflow.operators.empty")
    class MockOperator:
        def __init__(self, task_id, python_callable=None, **kwargs):
            self.task_id = task_id
            self.python_callable = python_callable
        def __rshift__(self, other): return other
        def __lshift__(self, other): return other
    python_mod.PythonOperator = MockOperator
    empty_mod.EmptyOperator = MockOperator
    sensors = types.ModuleType("airflow.sensors")
    ext_mod = types.ModuleType("airflow.sensors.external_task")
    ext_mod.ExternalTaskSensor = MockOperator
    utils = types.ModuleType("airflow.utils")
    dates = types.ModuleType("airflow.utils.dates")
    from datetime import datetime as _dt
    dates.days_ago = lambda n: _dt(2026, 1, 1)
    utils.dates = dates
    for name, mod in [
        ("airflow", airflow_mod), ("airflow.operators", ops),
        ("airflow.operators.python", python_mod), ("airflow.operators.empty", empty_mod),
        ("airflow.sensors", sensors), ("airflow.sensors.external_task", ext_mod),
        ("airflow.utils", utils), ("airflow.utils.dates", dates),
    ]:
        sys.modules[name] = mod

_ensure_airflow_mock()

# ─────────────────────────────────────────────────────────────────────────────
# predict → MetricsWriter 기록 검증
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictMetricsWiring:
    @pytest.fixture
    def client_with_writer(self, tmp_path, monkeypatch):
        """MetricsWriter를 tmp_path에 연결한 TestClient."""
        jsonl_path = tmp_path / "metrics.jsonl"
        monkeypatch.setenv("DDI_METRICS_JSONL_PATH", str(jsonl_path))
        monkeypatch.setenv("ADMIN_API_KEY", "test-admin-key")

        # predictor mock
        from datetime import date
        from unittest.mock import MagicMock, patch
        from serving.schemas import PredictResponse, RiskLevel
        mock_pred = MagicMock()
        mock_pred.predict.return_value = PredictResponse(
            patient_id="P001",
            risk_level=RiskLevel.RED,
            rule_level=RiskLevel.RED,
            ml_level=RiskLevel.YELLOW,
            drug_count=2,
            ddi_alerts=[],
            risk_reasons=[],
            intervention="즉각 개입 (당일 약사 면담 필요)",
            reference_date=date.today(),
        )

        import importlib
        import config.settings as s
        importlib.reload(s)

        from monitoring.metrics_writer import init_metrics_writer
        init_metrics_writer(path=jsonl_path)

        with patch("serving.routers.predict.get_predictor", return_value=mock_pred):
            from serving.main import app
            client = TestClient(app, raise_server_exceptions=False)
            yield client, jsonl_path

    def test_predict_writes_to_jsonl(self, client_with_writer):
        client, jsonl_path = client_with_writer
        resp = client.post("/predict", json={
            "patient_id": "P001",
            "drugs": [
                {"edi_code": "A001", "total_days": 30},
                {"edi_code": "B002", "total_days": 30},
            ],
        })
        # 예측 성공 여부와 무관하게 jsonl 기록 확인
        assert jsonl_path.exists(), "predict() should write metrics to jsonl"
        lines = jsonl_path.read_text().strip().splitlines()
        assert len(lines) >= 1
        record = json.loads(lines[0])
        assert "patient_id" in record
        assert "risk_level" in record
        assert "latency_ms" in record

    def test_predict_metrics_writer_failure_does_not_break_response(self, tmp_path, monkeypatch):
        """MetricsWriter.append() 예외 → 정상 응답 반환 검증."""
        from unittest.mock import MagicMock, patch
        from serving.schemas import PredictResponse, RiskLevel

        from datetime import date
        mock_pred = MagicMock()
        mock_pred.predict.return_value = PredictResponse(
            patient_id="P001",
            risk_level=RiskLevel.GREEN,
            rule_level=RiskLevel.GREEN,
            ml_level=None,
            drug_count=1,
            ddi_alerts=[],
            risk_reasons=[],
            intervention="분기 1회 복약 상담",
            reference_date=date.today(),
        )

        mock_writer = MagicMock()
        mock_writer.append.side_effect = RuntimeError("disk full")

        import monitoring.metrics_writer as mw
        monkeypatch.setattr(mw, "_writer", mock_writer)

        with patch("serving.routers.predict.get_predictor", return_value=mock_pred):
            from serving.main import app
            client = TestClient(app)
            resp = client.post("/predict", json={
                "patient_id": "P001",
                "drugs": [{"edi_code": "A001", "total_days": 30}],
            })
        # MetricsWriter 실패해도 예측 응답 정상
        assert resp.status_code == 200
        data = resp.json()
        assert "risk_level" in data


class TestMetricsEndpoint:
    def test_get_metrics_without_admin_key_returns_error(self, tmp_path, monkeypatch):
        import serving.routers.health as health_router
        # _ADMIN_KEY는 모듈 로드 시 캐시 → monkeypatch.setenv 무효; 직접 패치
        monkeypatch.setattr(health_router, "_ADMIN_KEY", "secret-key")
        from monitoring.metrics_writer import init_metrics_writer
        init_metrics_writer(path=tmp_path / "metrics.jsonl")
        from serving.main import app
        client = TestClient(app)
        resp = client.get("/metrics")  # X-Admin-Key 헤더 없음
        assert resp.status_code in (401, 422, 503)

    def test_get_metrics_with_correct_admin_key(self, tmp_path, monkeypatch):
        import serving.routers.health as health_router
        monkeypatch.setattr(health_router, "_ADMIN_KEY", "secret-key")
        jsonl_path = tmp_path / "metrics.jsonl"
        from monitoring.metrics_writer import init_metrics_writer
        init_metrics_writer(path=jsonl_path)
        from serving.main import app
        client = TestClient(app)
        resp = client.get("/metrics", headers={"X-Admin-Key": "secret-key"})
        assert resp.status_code == 200
        data = resp.json()
        assert "records" in data
        assert "count" in data


class TestPipelineDriftReference:
    def test_save_drift_reference_creates_pkl(self, tmp_path):
        """_save_drift_reference() → drift_reference.pkl 생성 및 로드 가능."""
        import numpy as np
        import pandas as pd
        train_df = pd.DataFrame({
            "drug_count": np.random.randint(1, 20, 100),
            "ddi_count": np.random.randint(0, 5, 100),
            "label": np.random.randint(0, 2, 100),
        })
        drift_ref_path = tmp_path / "drift_reference.pkl"

        from scripts.train.pipeline import TrainPipeline
        pipeline = TrainPipeline.__new__(TrainPipeline)
        # optional 파라미터로 경로 직접 전달 (settings 패치 불필요)
        pipeline._save_drift_reference(train_df, drift_reference_path=drift_ref_path)

        assert drift_ref_path.exists()
        from monitoring.drift_detector import DriftDetector
        loaded = DriftDetector.load(str(drift_ref_path))
        assert loaded._fitted
        assert "drug_count" in loaded._reference
        assert "ddi_count" in loaded._reference
        assert set(loaded._reference.keys()) == {"drug_count", "ddi_count"}
        # label 컬럼은 제외됨
        assert "label" not in loaded._reference

    def test_save_drift_reference_skips_when_no_features(self, tmp_path):
        """label/patient_id/split 외 피처가 없으면 파일 미생성."""
        import pandas as pd
        # DataFrame with ONLY excluded columns
        train_df = pd.DataFrame({
            "label": [0, 1, 0],
            "patient_id": ["P001", "P002", "P003"],
            "split": ["train", "train", "train"],
        })
        drift_ref_path = tmp_path / "drift_reference.pkl"

        from scripts.train.pipeline import TrainPipeline
        pipeline = TrainPipeline.__new__(TrainPipeline)
        pipeline._save_drift_reference(train_df, drift_reference_path=drift_ref_path)

        # No file should be created when no feature columns remain
        assert not drift_ref_path.exists()


class TestDAGDriftAndAlerts:
    @pytest.fixture
    def setup_drift_env(self, tmp_path):
        """_detect_drift, _generate_alerts 테스트를 위한 환경 셋업."""
        import numpy as np
        import pandas as pd
        from monitoring.drift_detector import DriftDetector
        from monitoring.metrics_writer import MetricsWriter
        from datetime import datetime, timezone

        # drift_reference.pkl 생성
        ref_df = pd.DataFrame({
            "drug_count": np.random.randint(1, 20, 200),
            "ddi_count": np.random.randint(0, 5, 200),
        })
        detector = DriftDetector()
        detector.fit(ref_df)
        drift_ref_path = tmp_path / "drift_reference.pkl"
        detector.save(str(drift_ref_path))

        # predictions_{partition}.parquet 생성 (drug_count, ddi_count 포함)
        # 기존 DAG는 YYYYMMDD 포맷 사용
        partition = "20260406"
        pred_path = tmp_path / f"predictions_{partition}.parquet"
        pred_df = pd.DataFrame({
            "patient_id": [f"P{i:03d}" for i in range(50)],
            "risk_level": ["Red"] * 10 + ["Yellow"] * 20 + ["Green"] * 20,
            "drug_count": np.random.randint(1, 20, 50),
            "ddi_count": np.random.randint(0, 5, 50),
            "rule_triggered": [True] * 10 + [False] * 40,
        })
        pred_df.to_parquet(pred_path, index=False)

        # metrics_live.jsonl 생성 (Rule/ML 불일치 20%)
        metrics_path = tmp_path / "metrics_live.jsonl"
        writer = MetricsWriter(path=metrics_path)
        for i in range(10):
            writer.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "partition": partition,
                "patient_id": f"P{i:03d}",
                "risk_level": "Red",
                "rule_level": "Red",
                "ml_level": "Yellow" if i < 2 else "Red",
                "disagree": i < 2,
                "latency_ms": 10.0,
                "source": "batch",
            })

        return {
            "tmp_path": tmp_path,
            "partition": partition,
            "drift_ref_path": drift_ref_path,
            "pred_path": pred_path,
            "metrics_path": metrics_path,
            "monitoring_dir": tmp_path,
        }

    def test_detect_drift_creates_drift_json(self, setup_drift_env, monkeypatch):
        env = setup_drift_env
        import config.settings as _s
        monkeypatch.setattr(_s, "DRIFT_REFERENCE_PATH", env["drift_ref_path"])
        monkeypatch.setattr(_s, "PREDICTIONS_DIR", env["tmp_path"])
        monkeypatch.setattr(_s, "MONITORING_DIR", env["monitoring_dir"])

        from dags.ddi_batch_predict_dag import _detect_drift
        _detect_drift(partition=env["partition"])

        drift_json = env["tmp_path"] / f"drift_{env['partition']}.json"
        assert drift_json.exists()
        import json
        data = json.loads(drift_json.read_text())
        assert "partition" in data
        assert data["partition"] == env["partition"]

    def test_generate_alerts_creates_alert_json(self, setup_drift_env, monkeypatch):
        env = setup_drift_env
        partition = env["partition"]
        import config.settings as _s
        monkeypatch.setattr(_s, "MONITORING_DIR", env["monitoring_dir"])
        monkeypatch.setattr(_s, "METRICS_JSONL_PATH", env["metrics_path"])

        # 먼저 drift JSON을 수동 생성 (partition은 YYYYMMDD 포맷)
        import json
        from pathlib import Path
        drift_json = env["tmp_path"] / f"drift_{partition}.json"
        drift_json.write_text(json.dumps({
            "partition": partition,
            "generated_at": "2026-04-06T00:00:00",
            "n_drifted": 0,
            "trigger_retrain": False,
            "summary": {"total_features": 2, "stable": 2, "warning": 0, "drift": 0},
            "features": [
                {"feature": "drug_count", "psi": 0.05, "status": "stable"},
                {"feature": "ddi_count", "psi": 0.03, "status": "stable"},
            ],
        }))
        alert_json = env["tmp_path"] / f"alerts_{partition}.json"

        from dags.ddi_batch_predict_dag import _generate_alerts
        _generate_alerts(partition=partition)

        assert alert_json.exists()
