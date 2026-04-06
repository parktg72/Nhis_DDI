"""모니터링 파이프라인 통합 테스트."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


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
        # label 컬럼은 제외됨
        assert "label" not in loaded._reference
