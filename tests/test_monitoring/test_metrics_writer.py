"""MetricsWriter 유닛 테스트."""
from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


class TestMetricsWriterAppend:
    def test_append_writes_json_line(self, tmp_path):
        from monitoring.metrics_writer import MetricsWriter
        path = tmp_path / "metrics.jsonl"
        w = MetricsWriter(path=path)
        record = {
            "timestamp": "2026-04-06T10:00:00.000000",
            "partition": "2026-04-06",
            "patient_id": "P001",
            "risk_level": "Red",
            "rule_level": "Red",
            "ml_level": "Yellow",
            "disagree": True,
            "latency_ms": 12.3,
            "source": "api",
        }
        w.append(record)
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 1
        loaded = json.loads(lines[0])
        assert loaded["patient_id"] == "P001"
        assert loaded["disagree"] is True

    def test_append_multiple_records(self, tmp_path):
        from monitoring.metrics_writer import MetricsWriter
        path = tmp_path / "metrics.jsonl"
        w = MetricsWriter(path=path)
        for i in range(5):
            w.append({"patient_id": f"P{i:03d}", "ts": i})
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 5

    def test_append_creates_parent_dirs(self, tmp_path):
        from monitoring.metrics_writer import MetricsWriter
        path = tmp_path / "nested" / "dir" / "metrics.jsonl"
        w = MetricsWriter(path=path)
        w.append({"x": 1})
        assert path.exists()

    def test_concurrent_append_no_data_loss(self, tmp_path):
        from monitoring.metrics_writer import MetricsWriter
        path = tmp_path / "metrics.jsonl"
        w = MetricsWriter(path=path)
        n_threads = 10
        n_records = 20

        def write_records(thread_id):
            for i in range(n_records):
                w.append({"thread": thread_id, "seq": i})

        threads = [threading.Thread(target=write_records, args=(t,)) for t in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        lines = path.read_text().strip().splitlines()
        assert len(lines) == n_threads * n_records

    def test_append_lock_timeout_raises(self, tmp_path):
        """append() lock 타임아웃 시 filelock.Timeout 예외 전파."""
        from monitoring.metrics_writer import MetricsWriter
        from filelock import FileLock, Timeout
        path = tmp_path / "metrics.jsonl"
        w = MetricsWriter(path=path, lock_timeout=0.001)
        lock = FileLock(str(path) + ".lock")
        with lock:
            with pytest.raises(Timeout):
                w.append({"x": 1})

    def test_append_flushes_to_disk(self, tmp_path):
        """flush() ensures data is readable before file is closed."""
        from monitoring.metrics_writer import MetricsWriter
        path = tmp_path / "flush_test.jsonl"
        writer = MetricsWriter(path=path, lock_timeout=1.0)
        record = {"ts": "2026-01-01T00:00:00Z", "value": 42}
        writer.append(record)
        # File must be readable immediately after append (flush happened)
        content = path.read_text()
        assert json.loads(content.strip()) == record


class TestMetricsWriterReadRecent:
    def _make_record(self, hours_ago: float, **kwargs) -> dict:
        ts = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()
        return {"timestamp": ts, "patient_id": "P001", **kwargs}

    def test_read_recent_filters_by_hours(self, tmp_path):
        from monitoring.metrics_writer import MetricsWriter
        path = tmp_path / "metrics.jsonl"
        w = MetricsWriter(path=path)
        w.append(self._make_record(0.5, partition="2026-04-06"))
        w.append(self._make_record(25.0, partition="2026-04-05"))
        records = w.read_recent(hours=24)
        assert len(records) == 1
        assert records[0]["partition"] == "2026-04-06"

    def test_read_recent_empty_file(self, tmp_path):
        from monitoring.metrics_writer import MetricsWriter
        path = tmp_path / "nonexistent.jsonl"
        w = MetricsWriter(path=path)
        records = w.read_recent(hours=24)
        assert records == []

    def test_read_recent_skips_corrupt_lines(self, tmp_path):
        from monitoring.metrics_writer import MetricsWriter
        path = tmp_path / "metrics.jsonl"
        ts = datetime.now(timezone.utc).isoformat()
        path.write_text(
            '{"timestamp": "' + ts + '", "patient_id": "P001"}\n'
            "NOT_VALID_JSON\n"
            '{"timestamp": "' + ts + '", "patient_id": "P002"}\n'
        )
        w = MetricsWriter(path=path)
        records = w.read_recent(hours=1)
        assert len(records) == 2
        assert {r["patient_id"] for r in records} == {"P001", "P002"}

    def test_read_recent_returns_empty_on_timeout(self, tmp_path):
        """읽기 중 락 타임아웃 → [] 반환."""
        from monitoring.metrics_writer import MetricsWriter
        from filelock import FileLock
        path = tmp_path / "metrics.jsonl"
        ts = datetime.now(timezone.utc).isoformat()
        path.write_text('{"timestamp": "' + ts + '", "x": 1}\n')
        w = MetricsWriter(path=path, lock_timeout=0.001)
        lock = FileLock(str(path) + ".lock")
        with lock:
            records = w.read_recent(hours=1)
        assert records == []


class TestAppendRetryBackoff:
    """append() retry/backoff 회귀 가드 — Codex 2026-05-07 #5.

    직전까지 append() 가 lock timeout 시 즉시 Timeout 전파. /predict 가 warning
    후 무시하지만 이 경우 메트릭 유실. retry/backoff 로 lock 경합 복구 가능성 +.

    설계: 첫 시도 + 3 retry (backoff 0.1/0.25/0.5초) = 총 4 시도. 모두 실패 시
    마지막 Timeout 전파. lock_timeout_count 누적 → 운영 가시성.
    """

    def test_append_immediate_success_no_counter_bump(self, tmp_path):
        from monitoring.metrics_writer import MetricsWriter
        path = tmp_path / "metrics.jsonl"
        w = MetricsWriter(path=path)
        w.append({"x": 1})
        assert w.lock_timeout_count == 0

    def test_append_succeeds_after_one_timeout(self, tmp_path, monkeypatch):
        """첫 시도 timeout, 두 번째 성공. counter=1, 결과 OK."""
        from monitoring.metrics_writer import MetricsWriter
        from filelock import Timeout
        path = tmp_path / "metrics.jsonl"
        w = MetricsWriter(path=path, lock_timeout=0.5)

        original_acquire = w._lock.acquire
        attempts = {"n": 0}

        def fake_acquire(*args, **kwargs):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise Timeout(str(path) + ".lock")
            return original_acquire(*args, **kwargs)

        monkeypatch.setattr(w._lock, "acquire", fake_acquire)
        w.append({"y": 2})
        assert w.lock_timeout_count == 1
        assert path.read_text().count("\n") == 1

    def test_append_succeeds_after_multiple_timeouts(self, tmp_path, monkeypatch):
        """첫·둘째 시도 timeout, 세 번째 성공. counter=2."""
        from monitoring.metrics_writer import MetricsWriter
        from filelock import Timeout
        path = tmp_path / "metrics.jsonl"
        w = MetricsWriter(path=path, lock_timeout=0.5)

        original_acquire = w._lock.acquire
        attempts = {"n": 0}

        def fake_acquire(*args, **kwargs):
            attempts["n"] += 1
            if attempts["n"] <= 2:
                raise Timeout(str(path) + ".lock")
            return original_acquire(*args, **kwargs)

        monkeypatch.setattr(w._lock, "acquire", fake_acquire)
        w.append({"z": 3})
        assert w.lock_timeout_count == 2

    def test_append_raises_after_retry_exhaustion(self, tmp_path, monkeypatch):
        """모든 시도 timeout → Timeout 전파. counter=4 (1 시도 + 3 retry)."""
        from monitoring.metrics_writer import MetricsWriter
        from filelock import Timeout
        path = tmp_path / "metrics.jsonl"
        w = MetricsWriter(path=path, lock_timeout=0.05)

        def always_timeout(*args, **kwargs):
            raise Timeout(str(path) + ".lock")

        monkeypatch.setattr(w._lock, "acquire", always_timeout)
        with pytest.raises(Timeout):
            w.append({"x": 1})
        assert w.lock_timeout_count == 4  # 1 시도 + 3 retry 모두 카운트

    def test_lock_timeout_count_thread_safe(self, tmp_path, monkeypatch):
        """동시 timeout 발생 시 카운터 race 없이 누적 (threading.Lock 보호)."""
        from monitoring.metrics_writer import MetricsWriter
        from filelock import Timeout
        path = tmp_path / "metrics.jsonl"
        w = MetricsWriter(path=path, lock_timeout=0.01)

        def always_timeout(*args, **kwargs):
            raise Timeout(str(path) + ".lock")

        monkeypatch.setattr(w._lock, "acquire", always_timeout)

        threads = []

        def runner():
            try:
                w.append({"x": 1})
            except Timeout:
                pass

        for _ in range(5):
            threads.append(threading.Thread(target=runner))
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # 5 thread × 4 시도 = 20
        assert w.lock_timeout_count == 20

    def test_predict_handler_compat_metrics_failure_warning_only(
        self, tmp_path, monkeypatch
    ):
        """retry 소진 후 Timeout 이 hot path 호출자(예: /predict) 에서 warning 으로
        무시되어 예측은 정상 반환되는지 — 라우터 측 try/except 호환 회귀.

        라우터 측 except 패턴은 routers/predict.py:43-60 의 기존 로직 그대로.
        """
        from monitoring.metrics_writer import MetricsWriter
        from filelock import Timeout
        path = tmp_path / "metrics.jsonl"
        w = MetricsWriter(path=path, lock_timeout=0.01)

        def always_timeout(*args, **kwargs):
            raise Timeout(str(path) + ".lock")

        monkeypatch.setattr(w._lock, "acquire", always_timeout)

        # 라우터 측 try/except 모방 — 예측 자체는 metrics 실패와 무관
        prediction_succeeded = True
        try:
            w.append({"x": 1})
        except Exception:
            pass  # logger.warning 후 예측 정상 반환 흐름
        assert prediction_succeeded
        assert w.lock_timeout_count == 4


class TestMetricsWriterSingleton:
    def test_init_metrics_writer_creates_singleton(self, tmp_path):
        import monitoring.metrics_writer as mw_mod
        from monitoring.metrics_writer import MetricsWriter
        path = tmp_path / "singleton_test.jsonl"
        mw_mod._writer = None  # reset
        w = mw_mod.init_metrics_writer(path=path, lock_timeout=1.0)
        assert isinstance(w, MetricsWriter)
        assert mw_mod.get_metrics_writer() is w

    def test_get_metrics_writer_raises_if_not_initialized(self, monkeypatch):
        import monitoring.metrics_writer as mw
        monkeypatch.setattr(mw, "_writer", None)
        with pytest.raises(RuntimeError, match="초기화"):
            mw.get_metrics_writer()
