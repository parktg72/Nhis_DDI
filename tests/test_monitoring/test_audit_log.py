"""M8 — 관리 API 감사 이벤트.

관리 엔드포인트는 헤더 인증만 있고 **모델 교체 등 관리 조작의 행위자·시점을
추적할 수단이 없었다**(기술검토 P1-12). 이 로그가 그 절반이다.

**절반인 이유를 먼저 고정한다** — 공유 키가 하나뿐이라 이 로그는 **행위자를
특정하지 못한다.** 남는 것은 "어느 IP 에서 언제 무엇을" 까지다. 키 발급·회전·
폐기 정책이 없는 한 그 이상은 되지 않는다.
"""
from __future__ import annotations

import json

import pytest

from monitoring.audit_log import AuditLog, audit


def _lines(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


# ── 기록기 ───────────────────────────────────────────────────────────────
def test_events_are_appended_as_json_lines(tmp_path):
    p = tmp_path / "audit.jsonl"
    log = AuditLog(path=p)

    log.append({"event": "auth_ok", "endpoint": "/admin/reload"})
    log.append({"event": "auth_failed", "endpoint": "/admin/reload"})

    rows = _lines(p)
    assert [r["event"] for r in rows] == ["auth_ok", "auth_failed"]


def test_every_event_gets_a_timestamp_and_schema_version(tmp_path):
    p = tmp_path / "audit.jsonl"
    AuditLog(path=p).append({"event": "auth_ok"})

    row = _lines(p)[0]
    assert row["timestamp"].endswith("Z")
    assert row["schema_version"] >= 1


def test_the_key_is_never_written(tmp_path):
    """키가 로그에 들어가면 로그가 곧 비밀이 된다."""
    p = tmp_path / "audit.jsonl"
    log = AuditLog(path=p)

    log.append({"event": "auth_failed", "x_admin_key": "super-secret",
                "authorization": "Bearer abc", "key": "zzz"})

    text = p.read_text(encoding="utf-8")
    assert "super-secret" not in text
    assert "zzz" not in text
    assert "Bearer" not in text


def test_audit_never_raises(tmp_path, monkeypatch):
    """감사 기록 실패가 관리 조작을 막아서는 안 된다 — 조작은 이미 일어났다."""
    import monitoring.audit_log as al

    class _Boom:
        def append(self, *a, **k):
            raise RuntimeError("disk full")

    monkeypatch.setattr(al, "_get_log", lambda: _Boom())
    audit("auth_ok", endpoint="/admin/reload")   # 예외가 새면 실패


def test_audit_is_a_noop_when_no_path_is_configured(tmp_path, monkeypatch):
    """경로 미설정은 장애가 아니다. 다만 조용히 꺼지므로 런북이 그것을 적어야 한다."""
    import monitoring.audit_log as al
    monkeypatch.setattr(al, "_LOG", None)
    monkeypatch.delenv("DDI_AUDIT_JSONL_PATH", raising=False)

    audit("auth_ok", endpoint="/admin/reload")


# ── 무엇을 담지 않는가 ───────────────────────────────────────────────────
def test_no_request_body_and_no_patient_data(tmp_path):
    """`/metrics` 는 환자 단위 행을 돌려준다. 감사 이벤트는 **호출**을 남기지
    응답을 남기지 않는다."""
    p = tmp_path / "audit.jsonl"
    AuditLog(path=p).append({
        "event": "admin_call", "endpoint": "/metrics",
        "hours": 24, "returned_count": 100,
        "records": [{"patient_id": "P0001"}],       # 들어오면 안 되는 것
        "body": {"model_path": "x"},
    })

    text = p.read_text(encoding="utf-8")
    assert "patient_id" not in text
    assert "P0001" not in text
    assert "body" not in text
    row = _lines(p)[0]
    assert row["hours"] == 24 and row["returned_count"] == 100


# ── 교차검토 1차 반영 — 금지 목록 → 허용 목록 ────────────────────────────

def test_nested_secrets_do_not_survive(tmp_path):
    """첫 판은 최상위 이름만 봤다 — 중첩 dict·리스트 안의 비밀이 통과했다."""
    p = tmp_path / "audit.jsonl"
    AuditLog(path=p).append({
        "event": "auth_failed",
        "headers": {"Authorization": "SYNTHETIC_SECRET"},
        "details": [{"token": "SYNTHETIC_SECRET"}],
        "METRICS_SCRAPE_KEY": "SYNTHETIC_SECRET",
    })

    text = p.read_text(encoding="utf-8")
    assert "SYNTHETIC_SECRET" not in text
    assert "headers" not in text and "details" not in text


def test_only_allowlisted_fields_survive(tmp_path):
    p = tmp_path / "audit.jsonl"
    AuditLog(path=p).append({
        "event": "admin_call", "endpoint": "/admin/reload", "outcome": "ok",
        "무엇이든": "통과하면 안 된다", "extra_field": 1,
    })

    row = _lines(p)[0]
    assert set(row) == {"event", "endpoint", "outcome", "schema_version", "timestamp"}


def test_reserved_fields_cannot_be_overwritten(tmp_path):
    """호출자가 시각·스키마를 덮으면 로그의 시간축이 거짓이 된다."""
    p = tmp_path / "audit.jsonl"
    AuditLog(path=p).append({
        "event": "auth_ok", "timestamp": "1999-01-01T00:00:00Z", "schema_version": 99,
    })

    row = _lines(p)[0]
    assert row["timestamp"] != "1999-01-01T00:00:00Z"
    assert row["schema_version"] == 1
    assert row["event"] == "auth_ok"


@pytest.mark.parametrize("bad", ["/etc/passwd", "C:/model/x", "\\\\host\\share\\x", "//srv/x"])
def test_absolute_targets_are_dropped(tmp_path, bad):
    """로그는 이동한다. 호스트 배치를 실어 보내지 않는다."""
    p = tmp_path / "audit.jsonl"
    AuditLog(path=p).append({"event": "admin_call", "target": bad})

    assert "target" not in _lines(p)[0]


def test_a_relative_target_survives(tmp_path):
    p = tmp_path / "audit.jsonl"
    AuditLog(path=p).append({"event": "admin_call", "target": "hierarchical/cur"})

    assert _lines(p)[0]["target"] == "hierarchical/cur"


def test_long_values_are_truncated(tmp_path):
    p = tmp_path / "audit.jsonl"
    AuditLog(path=p).append({"event": "admin_call", "endpoint": "/x" * 500})

    assert len(_lines(p)[0]["endpoint"]) <= 200


def test_missing_filelock_is_recorded_as_degraded(tmp_path, monkeypatch):
    """프로세스 간 보호가 사라진 것을 조용히 넘기면 줄이 섞인 로그를 믿게 된다."""
    import builtins

    real_import = builtins.__import__

    def _no_filelock(name, *a, **k):
        if name == "filelock":
            raise ImportError("no filelock")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _no_filelock)
    log = AuditLog(path=tmp_path / "audit.jsonl")

    assert log.degraded_reason and "filelock" in log.degraded_reason


def test_a_lock_timeout_is_counted_not_just_swallowed(tmp_path):
    """상한을 두면 요청은 안 막히지만 **기록은 사라진다.** 세지 않으면
    "감사 기록 누락" 을 감지할 방법이 없다."""
    log = AuditLog(path=tmp_path / "audit.jsonl")
    if log._flock is None:
        pytest.skip("filelock 미설치 환경")

    class _Timeout:
        def __enter__(self):
            raise TimeoutError("lock held")

        def __exit__(self, *a):
            return False

    log._flock = _Timeout()
    with pytest.raises(TimeoutError):
        log.append({"event": "auth_ok"})

    assert log.lock_timeout_count == 1


def test_the_caller_still_survives_a_lock_timeout(tmp_path, monkeypatch):
    """`audit()` 는 그 예외를 삼킨다 — 관리 요청은 계속된다."""
    import monitoring.audit_log as al

    class _Boom:
        lock_timeout_count = 0

        def append(self, *a, **k):
            raise TimeoutError("lock held")

    monkeypatch.setattr(al, "_get_log", lambda: _Boom())
    audit("auth_ok", endpoint="/admin/reload")   # 예외가 새면 실패
