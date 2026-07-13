"""/health and /model/info 가 계층 모델 단독 로드 상태를 정확히 반영하는지 검증.

Codex review 2026-05-06 회귀: HIERARCHICAL_MODEL_DIR 로 계층 모델만 로드된 구성에서
/health 가 model_loaded=False, /model/info 가 model_type="none" 으로 보였음 — 운영
모니터링이 정상 모델을 미로드로 오판하던 문제. 본 테스트는 그 회귀 가드.
"""
from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from serving.predictor import HybridPredictor, RequestFeatureBuilder


@pytest.fixture
def hierarchical_only_predictor():
    """계층 모델만 로드된 HybridPredictor — 단일 ML 은 미로드."""
    pred = HybridPredictor.__new__(HybridPredictor)
    pred._start_time = 0.0
    pred._ml = MagicMock()
    pred._ml.loaded = False
    pred._ml._model_type = None
    pred._ml._partition = None
    pred._ml._feature_names = None
    pred._ml._threshold = None
    pred._ddi_matrix = None
    pred._cyp = None
    pred._std = None
    pred._builder = RequestFeatureBuilder(
        ddi_matrix=None, cyp_extractor=None, code_standardizer=None
    )
    pred._safety_net = None
    pred._dup_detector = None
    pred._ml_lock = threading.Lock()
    pred._hier_lock = threading.RLock()

    hier = MagicMock()
    hier.loaded = True
    hier.feature_cols = [f"f{i}" for i in range(42)]
    hier._thresholds = {"tau_red": 0.7, "tau_review": 0.3}
    pred._hierarchical = hier
    return pred


@pytest.fixture
def app_client_hierarchical(hierarchical_only_predictor):
    import serving.predictor as pred_module
    from serving.main import app

    # lifespan startup 이 _predictor 를 덮어쓰므로, TestClient 진입 직후
    # mock 으로 교체한다 (mock_predictor 와 동일 패턴이지만 시점이 늦음).
    with TestClient(app, raise_server_exceptions=False) as client:
        pred_module._predictor = hierarchical_only_predictor
        yield client


def test_health_reports_loaded_when_only_hierarchical(app_client_hierarchical):
    """계층 모델만 로드되어 있어도 model_loaded=True 여야 함."""
    resp = app_client_hierarchical.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_loaded"] is True, (
        f"계층 모델 로드 상태가 model_loaded 에 미반영: {body}"
    )
    assert body["model_mode"] == "hierarchical"
    assert body["hierarchical_loaded"] is True


def test_model_info_reports_hierarchical_type(app_client_hierarchical):
    """계층 모델만 로드되어 있을 때 model_type='hierarchical'."""
    resp = app_client_hierarchical.get("/model/info")
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_type"] == "hierarchical", (
        f"계층 모델인데 model_type='none' 으로 보임: {body}"
    )
    assert body["n_features"] == 42
    assert body["threshold"] == pytest.approx(0.7)
