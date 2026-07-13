"""BatchPredictResponse requested/success/failed 분리 회귀 가드 — Codex 2026-05-07 #6.

직전까지 `total` 이 성공 건수였는데 클라이언트가 입력 건수로 오해 가능. 명시적
카운트 분리 + `total` 은 success_count alias 로 backward compat 유지.

Codex 합의 검증:
  - failed_count == requested_count - success_count
  - failed_count == len(warnings)  (현재 구조: 실패당 warnings 1건 push)
  - total == success_count
"""
from __future__ import annotations

import threading
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from serving.predictor import HybridPredictor, RequestFeatureBuilder
from serving.schemas import (
    BatchPredictResponse,
    RiskLevel,
)

# ─── Schema 단위 — 0 건 / 정합성 ─────────────────────────────────────────────

def test_schema_zero_counts_valid():
    """빈 results — 0/0/0 카운트 schema 가 거부 안 함 (라우터 계층 0건 케이스 보장)."""
    resp = BatchPredictResponse(
        results=[],
        requested_count=0,
        success_count=0,
        failed_count=0,
        total=0,
        red_count=0, yellow_count=0, green_count=0, normal_count=0,
        elapsed_ms=0.0,
    )
    assert resp.success_count == 0
    assert resp.failed_count == 0
    assert resp.total == 0


def test_schema_total_alias_to_success():
    """total 은 backward compat 용 success_count alias (Codex 2026-05-07 #6)."""
    resp = BatchPredictResponse(
        results=[], requested_count=5, success_count=3, failed_count=2, total=3,
        red_count=0, yellow_count=0, green_count=0, normal_count=0, elapsed_ms=1.0,
    )
    assert resp.total == resp.success_count == 3


# ─── 라우터 통합 — TestClient 로 /predict/batch 호출 ─────────────────────────

@pytest.fixture
def app_client_for_batch():
    """직전 test_serving 의 mock_predictor 패턴 재현."""
    import serving.predictor as pred_module
    from serving.main import app

    pred = HybridPredictor.__new__(HybridPredictor)
    pred._start_time = 0.0
    from unittest.mock import MagicMock
    pred._ml = MagicMock()
    pred._ml.loaded = False
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

    with TestClient(app, raise_server_exceptions=False) as client:
        pred_module._predictor = pred
        yield client


def _payload(n: int) -> dict:
    return {
        "requests": [
            {
                "patient_id": f"P{i:04d}",
                "drugs": [{"edi_code": "B001", "total_days": 30}],
            }
            for i in range(n)
        ]
    }


def test_all_success_counts_aligned(app_client_for_batch):
    """전건 성공 → success_count == requested_count, failed=0, warnings=[]."""
    with patch("serving.predictor._run_safety_net") as mock_sn, \
         patch("serving.predictor._run_duplicate_detector") as mock_dup:
        mock_sn.return_value = (RiskLevel.NORMAL, [], [])
        mock_dup.return_value = (0, [])
        resp = app_client_for_batch.post("/predict/batch", json=_payload(3))

    assert resp.status_code == 200
    body = resp.json()
    assert body["requested_count"] == 3
    assert body["success_count"] == 3
    assert body["failed_count"] == 0
    assert body["total"] == 3, "total 은 success_count alias (backward compat)"
    assert body["warnings"] == []
    # Codex 합의 검증
    assert body["failed_count"] == body["requested_count"] - body["success_count"]
    assert body["failed_count"] == len(body["warnings"])
    assert body["total"] == body["success_count"]


def test_all_fail_counts_aligned(app_client_for_batch):
    """전건 실패 → success=0, failed=requested, warnings 길이 일치."""
    def _boom(*a, **kw):
        raise RuntimeError("simulated failure")

    with patch("serving.predictor._run_safety_net", side_effect=_boom):
        resp = app_client_for_batch.post("/predict/batch", json=_payload(4))

    assert resp.status_code == 200
    body = resp.json()
    assert body["requested_count"] == 4
    assert body["success_count"] == 0
    assert body["failed_count"] == 4
    assert body["total"] == 0
    assert len(body["warnings"]) == 4
    assert body["failed_count"] == body["requested_count"] - body["success_count"]
    assert body["failed_count"] == len(body["warnings"])
    assert body["total"] == body["success_count"]


def test_partial_fail_counts_aligned(app_client_for_batch):
    """부분 실패 — patient_id 별로 mock 행동 분기. 합산 검증."""
    call_count = {"n": 0}

    def _maybe_boom(*a, **kw):
        call_count["n"] += 1
        # 5건 중 2번째와 4번째만 실패
        if call_count["n"] in (2, 4):
            raise RuntimeError("boom")
        return (RiskLevel.NORMAL, [], [])

    with patch("serving.predictor._run_safety_net", side_effect=_maybe_boom), \
         patch("serving.predictor._run_duplicate_detector",
               return_value=(0, [])):
        resp = app_client_for_batch.post("/predict/batch", json=_payload(5))

    assert resp.status_code == 200
    body = resp.json()
    assert body["requested_count"] == 5
    assert body["success_count"] == 3
    assert body["failed_count"] == 2
    assert body["total"] == 3
    assert len(body["warnings"]) == 2
    # Codex 3중 검증
    assert body["failed_count"] == body["requested_count"] - body["success_count"]
    assert body["failed_count"] == len(body["warnings"])
    assert body["total"] == body["success_count"]


def test_empty_batch_request_rejected_by_schema(app_client_for_batch):
    """빈 배치 — BatchPredictRequest.min_length=1 가 거부 (422)."""
    resp = app_client_for_batch.post("/predict/batch", json={"requests": []})
    assert resp.status_code == 422, (
        "BatchPredictRequest.requests 의 min_length=1 가 빈 배치 거부해야 함"
    )


def test_legacy_total_field_still_works(app_client_for_batch):
    """기존 클라이언트가 body['total'] 읽어도 동일 값 (backward compat)."""
    with patch("serving.predictor._run_safety_net") as mock_sn, \
         patch("serving.predictor._run_duplicate_detector") as mock_dup:
        mock_sn.return_value = (RiskLevel.NORMAL, [], [])
        mock_dup.return_value = (0, [])
        resp = app_client_for_batch.post("/predict/batch", json=_payload(2))

    body = resp.json()
    # 기존 클라이언트 코드 경로 — total 만 읽음
    legacy_count = body["total"]
    # 신규 success_count 와 일치
    assert legacy_count == body["success_count"] == 2


# ─── batch max_length=1000 boundary 회귀 가드 — Codex 2026-05-07 #5 ──────────


def test_batch_1000_requests_allowed(app_client_for_batch):
    """schema 상한 정확히 1000건 → 200 OK + count 정합성 + total alias.

    schemas.py:145 의 max_length=1000 회귀 가드. 1001 거부와 함께 boundary 양면.
    """
    with patch("serving.predictor._run_safety_net") as mock_sn, \
         patch("serving.predictor._run_duplicate_detector") as mock_dup:
        mock_sn.return_value = (RiskLevel.NORMAL, [], [])
        mock_dup.return_value = (0, [])
        resp = app_client_for_batch.post("/predict/batch", json=_payload(1000))

    assert resp.status_code == 200, (
        f"1000건 요청은 schema 상한 정확히 매치 — 200 OK 예상, got {resp.status_code}"
    )
    body = resp.json()
    assert body["requested_count"] == 1000
    assert body["success_count"] == 1000
    assert body["failed_count"] == 0
    # backward compat alias
    assert body["total"] == body["success_count"] == 1000
    # 정합성 (Codex 합의 3중 검증)
    assert body["failed_count"] == body["requested_count"] - body["success_count"]
    assert body["failed_count"] == len(body["warnings"])


def test_batch_1001_requests_rejected_by_schema(app_client_for_batch):
    """1001건 → 422 + predictor 미호출 (Pydantic validation 단계 차단).

    router 내부 예측 루프 진입 안 함 — _run_safety_net call 0 회 검증.
    """
    with patch("serving.predictor._run_safety_net") as mock_sn, \
         patch("serving.predictor._run_duplicate_detector") as mock_dup:
        mock_sn.return_value = (RiskLevel.NORMAL, [], [])
        mock_dup.return_value = (0, [])
        resp = app_client_for_batch.post("/predict/batch", json=_payload(1001))

        # 422 — Pydantic ValidationError (max_length 위반)
        assert resp.status_code == 422, (
            f"1001건 → schema validation 거부 (422) 예상, got {resp.status_code}"
        )
        # router 의 for-loop 진입 안 했어야 함 — predictor 미호출
        assert mock_sn.call_count == 0, (
            f"1001건 거부는 schema validation 단계에서 차단되어야 함 — "
            f"_run_safety_net 호출 횟수: {mock_sn.call_count}"
        )
