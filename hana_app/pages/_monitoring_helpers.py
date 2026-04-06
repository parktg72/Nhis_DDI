"""Streamlit 모니터링 페이지용 데이터 로딩 헬퍼."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def load_recent_metrics(path: Path, hours: int = 24) -> list[dict]:
    """metrics_live.jsonl에서 최근 hours 시간 이내 레코드를 반환한다."""
    path = Path(path)
    if not path.exists():
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    results = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
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
            logger.warning("metrics 줄 %d 파싱 실패 (skip): %s", i + 1, e)
    return results


def load_drift_report(monitoring_dir: Path, partition: str) -> dict | None:
    """drift_{partition}.json을 로드한다. 없으면 None 반환."""
    path = Path(monitoring_dir) / f"drift_{partition}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("drift 리포트 파싱 실패 (%s): %s", path, e)
        return None


def load_alerts(monitoring_dir: Path, partitions: list[str]) -> list[dict]:
    """최근 partitions의 alerts_*.json을 합쳐서 반환한다."""
    monitoring_dir = Path(monitoring_dir)
    all_alerts = []
    for p in partitions:
        path = monitoring_dir / f"alerts_{p}.json"
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                all_alerts.extend(data)
            elif isinstance(data, dict) and "alerts" in data:
                all_alerts.extend(data["alerts"])
        except Exception as e:
            logger.warning("알림 파일 파싱 실패 (%s): %s", path, e)
    return all_alerts


def compute_disagree_rate(records: list[dict]) -> float:
    """Rule/ML 불일치율을 계산한다."""
    if not records:
        return 0.0
    return sum(1 for r in records if r.get("disagree")) / len(records)


def psi_status_label(psi: float) -> str:
    """PSI 값에 따른 상태 레이블 반환."""
    if psi < 0.10:
        return "🟢 Stable"
    elif psi < 0.25:
        return "🟡 Warning"
    return "🔴 Drift"


def get_recent_partitions(monitoring_dir: Path, prefix: str = "drift_", n: int = 7) -> list[str]:
    """monitoring_dir에서 prefix로 시작하는 최근 n개 파티션 날짜를 반환한다."""
    monitoring_dir = Path(monitoring_dir)
    partitions = []
    for f in monitoring_dir.glob(f"{prefix}*.json"):
        name = f.stem  # e.g., "drift_2026-04-06"
        date_part = name[len(prefix):]
        if len(date_part) == 10:  # YYYY-MM-DD
            partitions.append(date_part)
    return sorted(partitions, reverse=True)[:n]
