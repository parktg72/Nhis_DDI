"""M8 — 관리 엔드포인트가 실제로 감사 이벤트를 남기는가.

인증 실패는 핸들러에 도달하지 않으므로 **가드에서 남기지 않으면 어디에도 남지
않는다.** 보안 가치의 대부분이 거기 있다.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

ADMIN_KEY = "test-admin-key-m8"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("ADMIN_API_KEY", ADMIN_KEY)
    monkeypatch.setenv("DDI_AUDIT_JSONL_PATH", str(tmp_path / "audit.jsonl"))
    import monitoring.audit_log as al
    monkeypatch.setattr(al, "_LOG", None)

    import serving.routers.health as health
    monkeypatch.setattr(health, "_ADMIN_KEY", ADMIN_KEY)

    from serving.predictor import HybridPredictor, RequestFeatureBuilder
    pred = HybridPredictor.__new__(HybridPredictor)
    pred._start_time = 0.0
    pred._ml = MagicMock(); pred._ml.loaded = False
    pred._ddi_matrix = pred._cyp = pred._std = None
    pred._builder = RequestFeatureBuilder(
        ddi_matrix=None, cyp_extractor=None, code_standardizer=None)
    pred._safety_net = pred._dup_detector = None
    pred._ml_lock = __import__("threading").Lock()
    pred._hier_lock = __import__("threading").RLock()
    pred._hierarchical = None
    import serving.predictor as pm
    monkeypatch.setattr(pm, "_predictor", pred)

    from serving.main import app
    with TestClient(app, raise_server_exceptions=False) as c:
        c._audit_path = tmp_path / "audit.jsonl"
        yield c


def _events(c):
    p = c._audit_path
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def test_a_bad_key_is_recorded(client):
    """핸들러는 돌지 않는다 — 가드가 남기지 않으면 어디에도 남지 않는다."""
    client.post("/admin/reload", json={"model_path": "x"},
                headers={"X-Admin-Key": "wrong"})

    evs = [e for e in _events(client) if e["event"] == "auth_failed"]
    assert evs, "인증 실패가 기록되지 않았다"
    assert evs[0]["endpoint"] == "/admin/reload"
    assert evs[0]["key_kind"] == "admin"


def test_a_good_key_is_recorded_too(client):
    """실패만 남기면 '언제 무엇이 정상 수행됐는가' 를 추적할 수 없다."""
    client.get("/metrics", headers={"X-Admin-Key": ADMIN_KEY})

    assert any(e["event"] == "auth_ok" for e in _events(client))


def test_the_key_never_appears_in_the_log(client):
    client.post("/admin/reload", json={"model_path": "x"},
                headers={"X-Admin-Key": ADMIN_KEY})

    text = client._audit_path.read_text(encoding="utf-8")
    assert ADMIN_KEY not in text


def test_a_rejected_path_is_recorded_with_its_outcome(client):
    """허용 디렉터리 밖 경로 거부도 관리 조작 시도다."""
    client.post("/admin/reload", json={"model_path": "/etc/passwd"},
                headers={"X-Admin-Key": ADMIN_KEY})

    evs = [e for e in _events(client) if e["event"] == "admin_call"]
    assert evs, "관리 호출이 기록되지 않았다"
    assert evs[-1]["outcome"] == "rejected_path"


def test_the_recorded_target_is_not_an_absolute_host_path(client):
    """로그는 이동한다. 호스트 배치를 실어 보내지 않는다."""
    client.post("/admin/reload", json={"model_path": "/etc/passwd"},
                headers={"X-Admin-Key": ADMIN_KEY})

    for e in _events(client):
        assert not str(e.get("target", "")).startswith(("/", "C:", "H:"))


def test_metrics_call_records_the_call_not_the_rows(client):
    """`/metrics` 는 환자 단위 행을 돌려준다. 감사 이벤트에 그것이 들어가면 안 된다."""
    client.get("/metrics?hours=24", headers={"X-Admin-Key": ADMIN_KEY})

    text = client._audit_path.read_text(encoding="utf-8")
    assert "patient_id" not in text
    evs = [e for e in _events(client) if e.get("endpoint") == "/metrics"
           and e["event"] == "admin_call"]
    assert evs and evs[-1]["hours"] == 24


def test_audit_failure_does_not_break_the_admin_call(client, monkeypatch):
    """감사 기록 실패가 관리 조작을 막아서는 안 된다 — 조작은 이미 일어났고,
    막아 봐야 되돌려지지 않는다.

    실제 실패 지점은 **기록기**다(디스크·잠금). `audit()` 자체를 예외로 바꾸는
    것은 일어나지 않는 대체이므로 그렇게 시험하지 않는다."""
    import monitoring.audit_log as al

    class _Boom:
        def append(self, *a, **k):
            raise OSError("disk full")

    monkeypatch.setattr(al, "_get_log", lambda: _Boom())
    r = client.post("/admin/reload", json={"model_path": "/etc/passwd"},
                    headers={"X-Admin-Key": ADMIN_KEY})

    assert r.status_code == 400   # 감사 실패가 아니라 경로 거부로 끝나야 한다
