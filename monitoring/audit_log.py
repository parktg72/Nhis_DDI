# -*- coding: utf-8 -*-
"""M8 — 관리 API 감사 이벤트 로그.

관리 엔드포인트(모델 핫스왑·메트릭 조회)는 헤더 인증만 있고, **관리 조작의
행위자·시점을 추적할 수단이 없었다**(기술검토 P1-12). 이 파일이 그 절반이다.

## 절반인 이유 — 먼저 적는다

**이 로그는 행위자를 특정하지 못한다.** 공유 관리 키가 하나뿐이므로 남는 것은
"**어느 IP 에서 언제 무엇을**" 까지다. 누가 그 키를 들고 있었는지는 알 수 없다.

키 **발급·회전·폐기 정책이 없는 한** 그 이상은 되지 않는다. 그 정책은 P1-12 의
나머지 절반이며 이 파일이 만들지 않는다. **로그가 있다는 사실이 책임 추적이
된다고 읽히면 안 된다** — 그것이 이 문단이 있는 이유다.

## 담는 것 — 허용 목록이다

**금지 목록이 아니라 허용 목록으로 간다.** 첫 판은 "키·본문 같은 이름을 버린다" 는
금지 목록이었고, 교차검토가 **중첩 dict 안의 인증 헤더·리스트 안의 토큰·목록에 없던
키 이름이 그대로 통과**하는 것을 실증했다. 이름을 세어 막는 방식은 새 이름이 생기면
뚫린다.

허용하는 것은 **아래 표의 필드뿐이고, 값은 스칼라여야 하며, 길이가 제한된다.**
dict·리스트는 통째로 거부한다 — 감사 이벤트에 구조가 필요하면 그건 본문이지
이벤트가 아니다.

`event`·`timestamp`·`schema_version` 은 **기록기가 마지막에 붙인다.** 호출자가
덮어쓸 수 없다.

- **절대 경로를 거부한다.** 모델 경로는 허용 디렉터리 기준 상대로만 받는다.
  로그는 이동하므로 호스트 배치를 실어 보내지 않는다

예측 메트릭 기록과 **별도 파일**이다. 보존 기간도 접근 권한도 다르다.
경로는 `DDI_AUDIT_JSONL_PATH` 로 지정하며, 미설정이면 조용히 기록하지 않는다.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

AUDIT_SCHEMA_VERSION = 1

# 허용 필드와 그 최대 길이. 여기 없는 이름은 값이 무엇이든 버린다.
_ALLOWED_FIELDS: dict[str, int] = {
    "endpoint": 200,        # 경로만. 질의 문자열 없음
    "client_ip": 64,
    "key_kind": 16,         # admin | scrape
    "outcome": 64,
    "target": 400,          # 허용 디렉터리 기준 **상대** 경로
    "request_id": 40,       # 인증 이벤트와 조작 결과를 잇는 서버 생성 값
    "hours": 16,
    "returned_count": 16,
    "status_code": 8,
}

# 기록기가 마지막에 붙인다. 호출자가 덮어쓸 수 없다.
_RESERVED = frozenset({"event", "timestamp", "schema_version"})


def _looks_absolute(value: str) -> bool:
    """POSIX·Windows·UNC 절대 경로를 모두 본다."""
    v = value.replace("\\", "/")
    return v.startswith("/") or v.startswith("//") or (len(v) > 1 and v[1] == ":")


class AuditLog:
    """감사 이벤트를 JSON Lines 로 append 한다."""

    def __init__(self, path, lock_timeout: float = 2.0) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()
        self._lock_timeout = lock_timeout
        self.degraded_reason: Optional[str] = None
        # 잠금 대기로 놓친 이벤트 수. 상한을 두면 요청은 안 막히지만 **기록은
        # 사라진다** — 세지 않으면 "감사 기록 누락" 을 감지할 방법이 없다.
        self.lock_timeout_count: int = 0
        try:
            from filelock import FileLock
            self._flock = FileLock(str(self._path) + ".lock", timeout=lock_timeout)
        except ImportError:
            # filelock 이 없으면 **프로세스 간 보호가 없다.** 스레드 잠금만 남는다.
            # 조용히 넘어가지 않고 상태로 남긴다 — worker 가 여럿이면 줄이 섞인다.
            self._flock = None
            self.degraded_reason = "filelock 미설치 — 프로세스 간 직렬화 없음"
            logger.warning("감사 로그: %s", self.degraded_reason)

    @staticmethod
    def _sanitize(event: dict) -> dict:
        """허용 목록 통과분만 남긴다. 구조는 통째로 버린다."""
        out: dict = {}
        for k, v in event.items():
            if k in _RESERVED:
                continue                      # 기록기가 붙인다
            limit = _ALLOWED_FIELDS.get(k)
            if limit is None:
                continue                      # 허용 목록에 없음
            if v is None:
                out[k] = None
                continue
            if isinstance(v, bool) or not isinstance(v, (str, int, float)):
                continue                      # dict·리스트·객체는 버린다
            if isinstance(v, str):
                if k == "target" and _looks_absolute(v):
                    continue                  # 절대 경로는 남기지 않는다
                v = v[:limit]
            out[k] = v
        return out

    def append(self, event: dict) -> None:
        row = {
            **self._sanitize(event),
            # 예약 필드는 **마지막에** 붙인다 — 호출자가 덮어쓸 수 없다.
            "event": str(event.get("event", ""))[:64],
            "schema_version": AUDIT_SCHEMA_VERSION,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        line = json.dumps(row, ensure_ascii=False) + "\n"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            if self._flock is not None:
                try:
                    with self._flock:
                        with open(self._path, "a", encoding="utf-8") as fh:
                            fh.write(line)
                except Exception:
                    # 대기 상한을 넘겼다. 관리 요청을 막지 않되 **놓친 것을 센다.**
                    self.lock_timeout_count += 1
                    raise
            else:
                with open(self._path, "a", encoding="utf-8") as fh:
                    fh.write(line)


_LOG: Optional[AuditLog] = None


def _get_log() -> Optional[AuditLog]:
    global _LOG
    if _LOG is not None:
        return _LOG
    path = os.environ.get("DDI_AUDIT_JSONL_PATH", "").strip()
    if not path:
        return None
    _LOG = AuditLog(path)
    return _LOG


def audit(event: str, **fields) -> None:
    """감사 이벤트 기록. **예외를 내지 않는다.**

    기록 실패가 관리 조작을 막아서는 안 된다 — 조작은 이미 일어났고, 막아 봐야
    되돌려지지 않는다. 실패는 로그로만 남긴다.
    """
    try:
        log = _get_log()
        if log is None:
            return
        log.append({"event": event, **fields})
    except Exception:
        logger.warning("감사 이벤트 기록 실패 (event=%s)", event, exc_info=True)
