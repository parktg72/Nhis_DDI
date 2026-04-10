"""ETL 실행 이력을 JSONL 파일에 영속 저장하는 헬퍼."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

ETL_LOG_PATH = Path(__file__).parent.parent / "etl_log.jsonl"


def append_etl_log(
    period_from: str,
    period_to: str,
    row_count: int,
    elapsed_sec: float,
    status: str = "ok",
    error: str = "",
) -> None:
    """ETL 실행 결과 1건을 etl_log.jsonl 에 append한다."""
    record = {
        "ts": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "period_from": period_from,
        "period_to": period_to,
        "row_count": row_count,
        "elapsed_sec": round(elapsed_sec, 1),
        "status": status,
        "error": error,
    }
    ETL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(ETL_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning("ETL 로그 기록 실패: %s", e)


def load_etl_log(n: int = 50) -> list[dict]:
    """etl_log.jsonl 에서 최근 n건을 최신순으로 반환한다."""
    if not ETL_LOG_PATH.exists():
        return []
    records: list[dict] = []
    for line in ETL_LOG_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except Exception:
            pass
    return list(reversed(records[-n:]))
