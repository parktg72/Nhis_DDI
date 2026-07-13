"""hana_app/core/etl_logger.py 단위 테스트."""
import json

import pytest


@pytest.fixture
def log_path(tmp_path):
    return tmp_path / "etl_log.jsonl"


def test_append_creates_file(log_path, monkeypatch):
    """append_etl_log() 호출 시 파일이 생성되고 JSON 라인이 추가됨."""
    import hana_app.core.etl_logger as m
    monkeypatch.setattr(m, "ETL_LOG_PATH", log_path)

    m.append_etl_log(
        period_from="2023/01", period_to="2023/12",
        row_count=100000, elapsed_sec=42.1,
    )
    assert log_path.exists()
    record = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert record["period_from"] == "2023/01"
    assert record["row_count"] == 100000
    assert record["status"] == "ok"
    assert record["error"] == ""


def test_append_multiple_lines(log_path, monkeypatch):
    """append 두 번 호출 시 두 줄이 기록됨."""
    import hana_app.core.etl_logger as m
    monkeypatch.setattr(m, "ETL_LOG_PATH", log_path)

    m.append_etl_log("2023/01", "2023/06", 50000, 20.0)
    m.append_etl_log("2023/07", "2023/12", 55000, 22.3)

    lines = [l for l in log_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 2


def test_append_error_status(log_path, monkeypatch):
    """status='error' 로 기록 가능."""
    import hana_app.core.etl_logger as m
    monkeypatch.setattr(m, "ETL_LOG_PATH", log_path)

    m.append_etl_log("2023/01", "2023/12", 0, 5.0, status="error", error="connection timeout")
    record = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert record["status"] == "error"
    assert "timeout" in record["error"]


def test_load_returns_empty_when_no_file(tmp_path, monkeypatch):
    """파일 없으면 빈 리스트 반환."""
    import hana_app.core.etl_logger as m
    monkeypatch.setattr(m, "ETL_LOG_PATH", tmp_path / "missing.jsonl")
    assert m.load_etl_log() == []


def test_load_returns_recent_n(log_path, monkeypatch):
    """load_etl_log(n=2) 는 마지막 2건만 반환, 최신이 index 0."""
    import hana_app.core.etl_logger as m
    monkeypatch.setattr(m, "ETL_LOG_PATH", log_path)

    for i in range(5):
        m.append_etl_log(f"2023/0{i+1}", f"2023/0{i+1}", i * 1000, float(i))

    records = m.load_etl_log(n=2)
    assert len(records) == 2
    assert records[0]["period_from"] == "2023/05"   # 최신이 먼저
    assert records[1]["period_from"] == "2023/04"


def test_load_skips_malformed_lines(log_path, monkeypatch):
    """깨진 줄은 건너뜀."""
    import hana_app.core.etl_logger as m
    monkeypatch.setattr(m, "ETL_LOG_PATH", log_path)

    log_path.write_text(
        '{"ts":"t","period_from":"a","period_to":"b","row_count":1,"elapsed_sec":1.0,"status":"ok","error":""}\n'
        'NOT_JSON\n',
        encoding="utf-8",
    )
    records = m.load_etl_log()
    assert len(records) == 1
