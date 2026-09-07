"""M7 — Prometheus 스크레이프 엔드포인트.

Grafana 대시보드 19패널이 참조할 스크레이프 대상이 없었다. 코드에는 메트릭이
정의돼 있으나 노출 경로가 pushgateway push 뿐이라, 대시보드는 가동될 수 없었다
(P1-5 "대시보드 가동 0패널").

`/metrics` 는 관리자 인증이 걸린 JSON 조회 계약이라 건드리지 않고 별도 경로를 낸다.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

ADMIN_KEY = "test-admin-key-m7"
PATH = "/metrics/prometheus"


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("ADMIN_API_KEY", ADMIN_KEY)
    import serving.routers.health as health
    monkeypatch.setattr(health, "_ADMIN_KEY", ADMIN_KEY)
    from serving.main import app
    return TestClient(app)


def test_exposition_returns_prometheus_text_format(client):
    r = client.get(PATH, headers={"X-Admin-Key": ADMIN_KEY})

    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    assert "ddi_prediction_total" in r.text


def test_exposition_requires_the_admin_key(client):
    """폐쇄망이라도 민감정보에서 파생된 집계다. 무인증 노출은 P1-12 와 어긋난다."""
    assert client.get(PATH).status_code == 422          # 헤더 누락
    assert client.get(PATH, headers={"X-Admin-Key": "wrong"}).status_code == 401


def test_exposition_says_why_when_the_client_library_is_absent(client, monkeypatch):
    """prometheus_client 미설치는 장애가 아니라 구성 문제다 — 503 으로 사유를 낸다."""
    import serving.routers.metrics as m
    monkeypatch.setattr(m, "_PROMETHEUS_AVAILABLE", False)

    r = client.get(PATH, headers={"X-Admin-Key": ADMIN_KEY})

    assert r.status_code == 503
    assert "prometheus_client" in r.json()["detail"]


def test_admin_json_metrics_endpoint_is_untouched(client):
    """기존 /metrics 계약은 그대로다 — 경로도 응답 형식도 바뀌지 않는다."""
    r = client.get("/metrics", headers={"X-Admin-Key": ADMIN_KEY})

    assert r.status_code in (200, 503)
    if r.status_code == 200:
        assert set(r.json()) == {"records", "count", "hours"}


# ── PR #22 리뷰 반영 ──────────────────────────────────────────────────────

SCRAPE_KEY = "scrape-only-key-m7"


def test_scrape_key_works_on_the_exposition_path(client, monkeypatch):
    """관리자 키는 모델 교체 권한까지 갖는다. 스크레이프 설정에 그 키를 넣지 않는다."""
    monkeypatch.setenv("METRICS_SCRAPE_KEY", SCRAPE_KEY)

    assert client.get(PATH, headers={"X-Admin-Key": SCRAPE_KEY}).status_code == 200


def test_scrape_key_does_not_open_the_admin_json_endpoint(client, monkeypatch):
    """권한 분리가 실제로 되는지 — 스크레이프 키로 관리 조회가 되면 분리가 아니다."""
    monkeypatch.setenv("METRICS_SCRAPE_KEY", SCRAPE_KEY)

    assert client.get("/metrics", headers={"X-Admin-Key": SCRAPE_KEY}).status_code == 401


def test_the_admin_key_stops_working_on_the_exposition_path_once_a_scrape_key_is_set(
        client, monkeypatch):
    monkeypatch.setenv("METRICS_SCRAPE_KEY", SCRAPE_KEY)

    assert client.get(PATH, headers={"X-Admin-Key": ADMIN_KEY}).status_code == 401


def test_without_a_scrape_key_the_admin_key_still_works(client, monkeypatch):
    """하위 호환 — 종전 구성이 깨지지 않는다."""
    monkeypatch.delenv("METRICS_SCRAPE_KEY", raising=False)

    assert client.get(PATH, headers={"X-Admin-Key": ADMIN_KEY}).status_code == 200


def test_exposition_serves_the_process_local_registry(client, monkeypatch):
    """worker 가 여럿이면 이 값은 집계가 아니다. 합산 경로는 넣지 않았다 —
    운영 worker 수가 확인되지 않았고, 확인되지 않은 배포 형태를 겨냥한 코드는
    남기지 않는다. worker 수는 A0 ⑤ 가 실측한다."""
    import serving.routers.metrics as m

    src = open(m.__file__, encoding="utf-8").read()
    assert "MultiProcessCollector" not in src
    assert "PROMETHEUS_MULTIPROC_DIR" not in src
