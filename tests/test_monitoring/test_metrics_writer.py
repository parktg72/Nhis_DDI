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
