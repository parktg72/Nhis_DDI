"""/health 와 /model/info 의 schema_drift 노출 회귀 가드 — Codex 2026-05-07 #1.

배경: serving/predictor.py 의 _validate_feature_schema 가 FEATURE_SCHEMA_LENIENT=1
환경에서 unknown feature 모델을 lenient 로 통과시키면 silent 0.0 fallback 으로
prediction 이 조용히 잘못된 확률 반환. 이 escape hatch 가 모니터링에서 보이지
않으면 우회가 장기 고착.

본 테스트는 Codex 합의 그대로:
  - strict + 정상 모델: status="ok", schema_drift=[], feature_schema_lenient=False
  - lenient + unknown feature: status="degraded", schema_drift non-empty,
    feature_schema_lenient=True, degraded_reasons 사유 박힘
  - lenient env 켜져있지만 실제 drift 없음: status="ok" 유지
    ("우회 가능 상태"가 아닌 "실제 drift 모델 로드" 기준)
  - /model/info 도 schema_drift 노출
"""
from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from serving.predictor import HybridPredictor, RequestFeatureBuilder


def _make_pred(*, ml_loaded: bool, schema_drift: list[str]) -> HybridPredictor:
    """단일 ML mock — schema_drift trail 만 컨트롤."""
    pred = HybridPredictor.__new__(HybridPredictor)
    pred._start_time = 0.0
    pred._ml = MagicMock()
    pred._ml.loaded = ml_loaded
    pred._ml._model_type = "XGBoostTrainer" if ml_loaded else None
    pred._ml._partition = None
    pred._ml._feature_names = ["drug_count", "age"] if ml_loaded else None
    pred._ml._threshold = 0.5 if ml_loaded else None
    pred._ml._schema_drift = list(schema_drift)
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
    pred._hierarchical = None
    return pred


@pytest.fixture
def app_client_factory():
    """factory — 각 테스트가 자체 pred 를 주입해 TestClient 사용."""
    from serving.main import app
    import serving.predictor as pred_module

    def _build(pred: HybridPredictor) -> TestClient:
        client = TestClient(app, raise_server_exceptions=False)
        client.__enter__()
        pred_module._predictor = pred
        return client

    return _build


def test_strict_normal_model_health_ok(app_client_factory, monkeypatch):
    """strict 모드 + 정상 모델 → status='ok', drift=[], lenient=False."""
    monkeypatch.delenv("FEATURE_SCHEMA_LENIENT", raising=False)
    pred = _make_pred(ml_loaded=True, schema_drift=[])
    client = app_client_factory(pred)
    try:
        body = client.get("/health").json()
    finally:
        client.__exit__(None, None, None)
    assert body["status"] == "ok"
    assert body["schema_drift"] == []
    assert body["feature_schema_lenient"] is False
    assert body["degraded_reasons"] == []


def test_lenient_with_unknown_feature_health_degraded(
    app_client_factory, monkeypatch
):
    """lenient + unknown feature → status='degraded' + drift trail + 사유."""
    monkeypatch.setenv("FEATURE_SCHEMA_LENIENT", "1")
    pred = _make_pred(ml_loaded=True, schema_drift=["fake_xyz"])
    client = app_client_factory(pred)
    try:
        body = client.get("/health").json()
    finally:
        client.__exit__(None, None, None)
    assert body["status"] == "degraded", (
        f"schema_drift non-empty 면 degraded 자동 전환되어야 함: {body}"
    )
    assert body["schema_drift"] == ["fake_xyz"]
    assert body["feature_schema_lenient"] is True
    assert any(
        "feature_schema_drift" in r for r in body["degraded_reasons"]
    ), f"degraded_reasons 에 schema drift 사유 박혀야 함: {body['degraded_reasons']}"


def test_lenient_env_active_but_no_drift_health_ok(
    app_client_factory, monkeypatch
):
    """lenient env 켜졌지만 실제 drift 없음 → status='ok' 유지.

    Codex 합의: "우회 가능 상태"가 아닌 "실제 drift 모델 로드" 기준으로 degraded.
    """
    monkeypatch.setenv("FEATURE_SCHEMA_LENIENT", "1")
    pred = _make_pred(ml_loaded=True, schema_drift=[])
    client = app_client_factory(pred)
    try:
        body = client.get("/health").json()
    finally:
        client.__exit__(None, None, None)
    assert body["status"] == "ok"
    assert body["schema_drift"] == []
    assert body["feature_schema_lenient"] is True, (
        "lenient env 활성 trail 은 유지되어야 함 (운영 가시성)"
    )
    assert body["degraded_reasons"] == []


def test_model_info_exposes_schema_drift(app_client_factory, monkeypatch):
    """/model/info 도 schema_drift 노출 (디버깅/감사용)."""
    monkeypatch.setenv("FEATURE_SCHEMA_LENIENT", "1")
    pred = _make_pred(ml_loaded=True, schema_drift=["legacy_feat_a", "legacy_feat_b"])
    client = app_client_factory(pred)
    try:
        body = client.get("/model/info").json()
    finally:
        client.__exit__(None, None, None)
    assert body["schema_drift"] == ["legacy_feat_a", "legacy_feat_b"], (
        f"/model/info 가 schema_drift 미노출: {body}"
    )


def test_lenient_env_blocked_by_sunset_health_degraded(
    app_client_factory, monkeypatch
):
    """env=1 + sunset 통과 → /health 가 'env 켜졌지만 차단됨' 명시 (Codex #6-followup).

    feature_schema_lenient_allowed=False + sunset_date 노출 + degraded_reasons 사유.
    """
    monkeypatch.setenv("FEATURE_SCHEMA_LENIENT", "1")
    monkeypatch.setenv("FEATURE_SCHEMA_LENIENT_SUNSET_DATE", "2020-01-01")  # 과거
    pred = _make_pred(ml_loaded=True, schema_drift=[])
    client = app_client_factory(pred)
    try:
        body = client.get("/health").json()
    finally:
        client.__exit__(None, None, None)
    # env trail 은 그대로 (env 활성)
    assert body["feature_schema_lenient"] is True
    # 실제 효력은 차단
    assert body["feature_schema_lenient_allowed"] is False, (
        "sunset 통과 후엔 env=1 이어도 lenient 효력 차단되어야 함"
    )
    # sunset date 노출 — 운영자가 차단 시점 인지
    assert body["feature_schema_lenient_sunset_date"] == "2020-01-01"
    # status degraded + 사유
    assert body["status"] == "degraded"
    assert any(
        "lenient_blocked_by_sunset" in r for r in body["degraded_reasons"]
    ), f"sunset 차단 사유 박혀야 함: {body['degraded_reasons']}"


def test_lenient_env_within_sunset_allowed(app_client_factory, monkeypatch):
    """env=1 + sunset 안 → lenient_allowed=True (실제 효력)."""
    monkeypatch.setenv("FEATURE_SCHEMA_LENIENT", "1")
    monkeypatch.setenv("FEATURE_SCHEMA_LENIENT_SUNSET_DATE", "2099-12-31")  # 미래
    pred = _make_pred(ml_loaded=True, schema_drift=[])
    client = app_client_factory(pred)
    try:
        body = client.get("/health").json()
    finally:
        client.__exit__(None, None, None)
    assert body["feature_schema_lenient"] is True
    assert body["feature_schema_lenient_allowed"] is True, (
        "sunset 안 + env=1 → 실제 lenient 효력"
    )
    assert body["feature_schema_lenient_sunset_date"] == "2099-12-31"


def test_default_sunset_date_exposed(app_client_factory, monkeypatch):
    """env 미설정 시 코드 default sunset 노출 (운영자가 default 인지)."""
    monkeypatch.delenv("FEATURE_SCHEMA_LENIENT", raising=False)
    monkeypatch.delenv("FEATURE_SCHEMA_LENIENT_SUNSET_DATE", raising=False)
    pred = _make_pred(ml_loaded=True, schema_drift=[])
    client = app_client_factory(pred)
    try:
        body = client.get("/health").json()
    finally:
        client.__exit__(None, None, None)
    # 코드 default 2026-08-01 (Codex 합의)
    assert body["feature_schema_lenient_sunset_date"] == "2026-08-01"


def test_multiple_drift_columns_truncated_in_reason(
    app_client_factory, monkeypatch
):
    """6개 이상 drift 시 degraded_reasons 사유는 처음 5개만 + '...' (가독성)."""
    monkeypatch.setenv("FEATURE_SCHEMA_LENIENT", "1")
    pred = _make_pred(
        ml_loaded=True,
        schema_drift=["a", "b", "c", "d", "e", "f", "g"],
    )
    client = app_client_factory(pred)
    try:
        body = client.get("/health").json()
    finally:
        client.__exit__(None, None, None)
    # schema_drift 자체는 전체 list
    assert body["schema_drift"] == ["a", "b", "c", "d", "e", "f", "g"]
    # 사유는 truncated
    reason = body["degraded_reasons"][0]
    assert "7 unknown columns" in reason
    assert "..." in reason  # 6번째 이후 truncated 표시
