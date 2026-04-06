"""
filelock 기반 JSON Lines 메트릭 기록기.

멀티 워커(uvicorn --workers N) 환경에서 안전하게 metrics_live.jsonl에 기록한다.
fcntl 미사용 — filelock 라이브러리로 크로스 플랫폼 지원.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from filelock import FileLock, Timeout

from config import settings

logger = logging.getLogger(__name__)


class MetricsWriter:
    """JSON Lines 파일에 메트릭 레코드를 append하는 기록기."""

    def __init__(
        self,
        path: Optional[Path] = None,
        lock_timeout: Optional[float] = None,
    ) -> None:
        self._path = Path(path) if path is not None else settings.METRICS_JSONL_PATH
        self._lock_timeout = lock_timeout if lock_timeout is not None else settings.METRICS_JSONL_LOCK_TIMEOUT
        self._lock = FileLock(str(self._path) + ".lock")

    def append(self, record: dict) -> None:
        """레코드 1건을 jsonl 파일 끝에 추가.

        lock_timeout 초과 시 filelock.Timeout 예외 전파.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock.acquire(timeout=self._lock_timeout):
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()

    def read_recent(self, hours: int = 24) -> list[dict]:
        """최근 hours 시간 이내 레코드 반환.

        파일 없으면 [] 반환. 파싱 실패 줄은 skip + WARNING.
        읽기 lock_timeout 초과 시 [] 반환 + WARNING.
        """
        if not self._path.exists():
            return []
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        try:
            # NOTE: full file read under exclusive lock; only use for dashboard polling, not hot path
            with self._lock.acquire(timeout=self._lock_timeout):
                lines = self._path.read_text(encoding="utf-8").splitlines()
        except Timeout:
            logger.warning("MetricsWriter.read_recent(): 락 타임아웃 — 빈 목록 반환")
            return []

        results: list[dict] = []
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                ts_str = record.get("timestamp", "")
                ts = datetime.fromisoformat(ts_str)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if ts >= cutoff:
                    results.append(record)
            except Exception as e:
                logger.warning("metrics_live.jsonl 줄 %d 파싱 실패 (skip): %s", i + 1, e)
        return results


# ─────────────────────────────────────────────────────────────────────────────
# 싱글턴
# ─────────────────────────────────────────────────────────────────────────────

_writer: Optional[MetricsWriter] = None


def get_metrics_writer() -> MetricsWriter:
    """초기화된 MetricsWriter 싱글턴 반환. 초기화 전 호출 시 RuntimeError."""
    global _writer
    if _writer is None:
        raise RuntimeError(
            "MetricsWriter가 초기화되지 않았습니다. lifespan에서 init_metrics_writer() 호출 필요"
        )
    return _writer


def init_metrics_writer(path: Optional[Path] = None, lock_timeout: Optional[float] = None) -> MetricsWriter:
    """MetricsWriter 싱글턴을 초기화하고 반환."""
    global _writer
    _writer = MetricsWriter(path=path, lock_timeout=lock_timeout)
    return _writer
