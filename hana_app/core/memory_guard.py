"""
프로세스 레벨 RAM 사용량 감시 및 강제 제한

psutil 기반으로 RSS를 능동 모니터링하여 OOM 발생 전에
gc.collect() → 배치 축소 → 정상 종료 단계적 에스컬레이션.

Windows 11 호환 (resource.setrlimit 불가 환경).

사용 예시:
    guard = MemoryGuard(limit_mb=4096)

    # 배치 루프에서 수동 체크
    for batch in batches:
        guard.check()            # HARD_STOP 시 MemoryLimitExceeded 발생
        process(batch)

    # 컨텍스트 매니저로 백그라운드 모니터링
    with guard.monitor(interval_sec=2.0):
        heavy_work()

    # 적응적 배치 크기
    batch_size = guard.suggest_batch_size(current_batch=5000, row_bytes=800)
"""
from __future__ import annotations

import gc
import logging
import threading
from contextlib import contextmanager
from enum import Enum
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class MemoryStatus(Enum):
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    HARD_STOP = "hard_stop"


class MemoryLimitExceeded(Exception):
    """RAM 한도 초과 시 발생하는 예외 (MemoryError 대신 사용).

    OOM 전에 깨끗한 상태에서 발생하므로 UI에서 안전하게 catch 가능.
    """

    def __init__(self, rss_mb: int, limit_mb: int, message: str = ""):
        self.rss_mb = rss_mb
        self.limit_mb = limit_mb
        self.message = message or (
            f"RAM 사용량({rss_mb:,} MB)이 한도({limit_mb:,} MB)의 95%를 초과하여 "
            f"처리를 중단합니다. 데이터 범위를 줄이거나 배치 크기를 낮춰 주세요."
        )
        super().__init__(self.message)


def _get_rss_mb() -> int:
    """현재 프로세스의 RSS 메모리 사용량(MB). psutil 없으면 -1."""
    try:
        import psutil
        return int(psutil.Process().memory_info().rss / 1024 / 1024)
    except Exception:
        return -1


class MemoryGuard:
    """프로세스 레벨 RAM 사용량 감시기.

    3단계 에스컬레이션:
        WARNING  (80%): gc.collect() + 로그 경고
        CRITICAL (90%): gc.collect() + 배치 축소 권고
        HARD_STOP(95%): MemoryLimitExceeded 예외 발생 (OOM 전 안전 종료)

    Parameters
    ----------
    limit_mb : 최대 허용 RAM (MB)
    warn_pct : WARNING 임계값 (기본 0.80)
    critical_pct : CRITICAL 임계값 (기본 0.90)
    hard_stop_pct : HARD_STOP 임계값 (기본 0.95)
    on_warning : WARNING 시 콜백 (UI 로깅용)
    on_critical : CRITICAL 시 콜백 (UI 로깅용)
    """

    def __init__(
        self,
        limit_mb: int,
        warn_pct: float = 0.80,
        critical_pct: float = 0.90,
        hard_stop_pct: float = 0.95,
        on_warning: Optional[Callable[[str], None]] = None,
        on_critical: Optional[Callable[[str], None]] = None,
    ):
        self.limit_mb = max(256, limit_mb)
        self.warn_mb = int(self.limit_mb * warn_pct)
        self.critical_mb = int(self.limit_mb * critical_pct)
        self.hard_stop_mb = int(self.limit_mb * hard_stop_pct)
        self._on_warning = on_warning
        self._on_critical = on_critical
        self._gc_count = 0
        self._last_status = MemoryStatus.NORMAL
        self._monitor_stop = threading.Event()

    # ── 핵심 API ──────────────────────────────────────────────────────────

    def rss_mb(self) -> int:
        """현재 프로세스 RSS (MB)."""
        return _get_rss_mb()

    def headroom_mb(self) -> int:
        """남은 여유 메모리 (MB). psutil 사용 불가 시 limit_mb 반환."""
        rss = self.rss_mb()
        if rss < 0:
            return self.limit_mb
        return max(0, self.limit_mb - rss)

    def status(self) -> MemoryStatus:
        """현재 메모리 상태 등급 반환 (예외 발생 없음)."""
        rss = self.rss_mb()
        if rss < 0:
            return MemoryStatus.NORMAL  # psutil 없으면 안전 모드
        if rss >= self.hard_stop_mb:
            return MemoryStatus.HARD_STOP
        if rss >= self.critical_mb:
            return MemoryStatus.CRITICAL
        if rss >= self.warn_mb:
            return MemoryStatus.WARNING
        return MemoryStatus.NORMAL

    def check(self) -> MemoryStatus:
        """메모리 상태를 확인하고, 단계별 대응 실행.

        - NORMAL: 패스
        - WARNING: gc.collect() + 로그
        - CRITICAL: gc.collect() + 콜백 호출
        - HARD_STOP: MemoryLimitExceeded 예외 발생

        Returns
        -------
        MemoryStatus (HARD_STOP 제외 — 예외로 전환됨)

        Raises
        ------
        MemoryLimitExceeded : RSS >= hard_stop_mb
        """
        rss = self.rss_mb()
        if rss < 0:
            return MemoryStatus.NORMAL

        st = self.status()

        if st == MemoryStatus.WARNING:
            if self._last_status == MemoryStatus.NORMAL:
                gc.collect()
                self._gc_count += 1
                msg = (
                    f"[MemoryGuard] WARNING: {rss:,} MB / {self.limit_mb:,} MB "
                    f"({rss / self.limit_mb * 100:.0f}%) — gc.collect() 실행"
                )
                logger.warning(msg)
                if self._on_warning:
                    self._on_warning(msg)
            self._last_status = st
            return st

        if st == MemoryStatus.CRITICAL:
            gc.collect()
            self._gc_count += 1
            rss_after = self.rss_mb()
            msg = (
                f"[MemoryGuard] CRITICAL: {rss:,} MB → gc 후 {rss_after:,} MB / "
                f"{self.limit_mb:,} MB ({rss_after / self.limit_mb * 100:.0f}%) — "
                f"배치 축소 권장"
            )
            logger.warning(msg)
            if self._on_critical:
                self._on_critical(msg)
            self._last_status = st
            # gc 후 HARD_STOP 아래로 내려왔으면 CRITICAL로 유지
            if rss_after >= self.hard_stop_mb:
                raise MemoryLimitExceeded(rss_after, self.limit_mb)
            return st

        if st == MemoryStatus.HARD_STOP:
            gc.collect()
            rss_after = self.rss_mb()
            if rss_after >= self.hard_stop_mb:
                raise MemoryLimitExceeded(rss_after, self.limit_mb)
            # gc 후 회복되면 CRITICAL로 다운그레이드
            self._last_status = MemoryStatus.CRITICAL
            return MemoryStatus.CRITICAL

        self._last_status = MemoryStatus.NORMAL
        return MemoryStatus.NORMAL

    def suggest_batch_size(
        self,
        current_batch: int,
        row_bytes: int = 800,
        min_batch: int = 100,
    ) -> int:
        """현재 여유 메모리 기반으로 적절한 배치 크기 제안.

        row_bytes: 행 1개당 예상 메모리 사용량 (바이트)
        안전 계수 2를 적용 (pandas 내부 복사 고려).
        """
        headroom = self.headroom_mb()
        if headroom <= 0:
            return min_batch
        # 여유 메모리의 50%를 배치에 할당 (나머지는 중간 계산용)
        available_bytes = headroom * 1024 * 1024 // 2
        suggested = max(min_batch, available_bytes // max(row_bytes, 1))
        return min(current_batch, suggested)

    # ── 백그라운드 모니터 ─────────────────────────────────────────────────

    @contextmanager
    def monitor(self, interval_sec: float = 2.0):
        """백그라운드 스레드에서 주기적으로 메모리 체크.

        HARD_STOP 도달 시 메인 스레드에서 다음 check() 호출 때 예외 발생.
        (백그라운드에서는 gc.collect()만 수행, 예외는 발생시키지 않음)
        """
        self._monitor_stop.clear()

        def _bg_check():
            while not self._monitor_stop.is_set():
                rss = self.rss_mb()
                if rss >= self.critical_mb:
                    gc.collect()
                    self._gc_count += 1
                    rss_after = self.rss_mb()
                    logger.info(
                        "[MemoryGuard BG] gc.collect(): %d MB → %d MB",
                        rss, rss_after,
                    )
                self._monitor_stop.wait(interval_sec)

        t = threading.Thread(target=_bg_check, daemon=True, name="MemoryGuard-BG")
        t.start()
        try:
            yield self
        finally:
            self._monitor_stop.set()
            t.join(timeout=interval_sec + 1)

    # ── 유틸리티 ──────────────────────────────────────────────────────────

    def info(self) -> str:
        rss = self.rss_mb()
        return (
            f"MemoryGuard: limit={self.limit_mb:,} MB, "
            f"warn={self.warn_mb:,} MB, critical={self.critical_mb:,} MB, "
            f"hard_stop={self.hard_stop_mb:,} MB, "
            f"rss={rss:,} MB, gc_count={self._gc_count}"
        )

    def __repr__(self) -> str:
        return f"MemoryGuard(limit_mb={self.limit_mb})"


# ─────────────────────────────────────────────────────────────────────────────
# NullGuard: guard=None일 때 안전하게 사용할 수 있는 no-op 구현
# ─────────────────────────────────────────────────────────────────────────────

class _NullGuard:
    """MemoryGuard의 no-op 대체품. 모든 메서드가 안전하게 아무 것도 하지 않음."""
    limit_mb = 0

    def rss_mb(self) -> int:
        return -1

    def headroom_mb(self) -> int:
        return 999999

    def status(self) -> MemoryStatus:
        return MemoryStatus.NORMAL

    def check(self) -> MemoryStatus:
        return MemoryStatus.NORMAL

    def suggest_batch_size(self, current_batch: int, **kwargs) -> int:
        return current_batch

    @contextmanager
    def monitor(self, interval_sec: float = 2.0):
        yield self

    def info(self) -> str:
        return "NullGuard (비활성)"


_NULL_GUARD = _NullGuard()


def get_guard(guard: Optional[MemoryGuard] = None) -> MemoryGuard | _NullGuard:
    """guard가 None이면 NullGuard 반환. 호출자가 None 체크 없이 사용 가능."""
    return guard if guard is not None else _NULL_GUARD
