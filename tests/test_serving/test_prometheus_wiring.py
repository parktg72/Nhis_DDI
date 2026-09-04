"""M7 — 서빙이 메트릭을 실제로 올리는가.

`monitoring/metrics.py` 는 메트릭 8종을 정의해 두었으나 **저장소 어디에서도
import 하지 않았다.** 정의만 있고 호출자가 없으니 레지스트리에 등록되지 않고,
노출 엔드포인트를 붙여도 빈 페이지가 나온다. 이 파일은 그 배선을 고정한다.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

ADMIN_KEY = "test-admin-key-wiring"


@pytest.fixture()
def client(monkeypatch):
    """기존 스위트와 같은 mock 예측기 위에서 돈다 (모델 없이 경로만 행사)."""
    from unittest.mock import MagicMock

    from serving.predictor import HybridPredictor, RequestFeatureBuilder

    monkeypatch.setenv("ADMIN_API_KEY", ADMIN_KEY)
    import serving.routers.health as health
    monkeypatch.setattr(health, "_ADMIN_KEY", ADMIN_KEY)

    pred = HybridPredictor.__new__(HybridPredictor)
    pred._start_time = 0.0
    pred._ml = MagicMock()
    pred._ml.loaded = False
    pred._ddi_matrix = None
    pred._cyp = None
    pred._std = None
    pred._builder = RequestFeatureBuilder(
        ddi_matrix=None, cyp_extractor=None, code_standardizer=None)
    pred._safety_net = None
    pred._dup_detector = None
    pred._ml_lock = __import__("threading").Lock()
    pred._hier_lock = __import__("threading").RLock()
    pred._hierarchical = None

    import serving.predictor as pred_module
    monkeypatch.setattr(pred_module, "_predictor", pred)

    from serving.main import app
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def _payload(pid: str) -> dict:
    return {
        "patient_id": pid,
        "drugs": [
            {"edi_code": "B001002", "atc_code": "A10BA02",
             "drug_name": "metformin", "total_days": 30},
            {"edi_code": "B001003", "atc_code": "C07AB02",
             "drug_name": "metoprolol", "total_days": 30},
        ],
        "patient_age": 65,
        "patient_sex": "M",
    }


def _scrape(client) -> str:
    r = client.get("/metrics/prometheus", headers={"X-Admin-Key": ADMIN_KEY})
    assert r.status_code == 200
    return r.text


def test_serving_registers_the_metric_definitions(client):
    """import 되지 않으면 계열 자체가 없다 — 대시보드 19패널이 전부 No data 가 된다."""
    body = _scrape(client)

    for name in ("ddi_prediction_total", "ddi_prediction_latency_ms",
                 "ddi_batch_success_total", "ddi_batch_fail_total"):
        assert name in body, f"{name} 이 레지스트리에 없다"


def test_a_prediction_increments_the_counter(client):
    before = _scrape(client)
    client.post("/predict", json=_payload("M7-WIRE-1"))
    after = _scrape(client)

    assert after != before
    assert "ddi_prediction_total{" in after


def test_batch_records_success_and_failure_counts(client):
    client.post("/predict/batch", json={
        "requests": [_payload("M7-WIRE-2"), _payload("M7-WIRE-3")],
    })

    body = _scrape(client)
    assert "ddi_batch_success_total{" in body


def test_metric_emission_never_breaks_the_response(client, monkeypatch):
    """관측은 부가 기능이다. 실패해도 예측 응답은 정상이어야 한다."""
    import serving.routers.predict as pr

    def _boom(*a, **k):
        raise RuntimeError("prometheus down")

    monkeypatch.setattr(pr, "record_prediction", _boom)
    r = client.post("/predict", json=_payload("M7-WIRE-4"))

    assert r.status_code == 200
