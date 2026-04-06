"""
메트릭 조회 엔드포인트

GET /metrics  - 최근 24시간 예측 메트릭 조회 (X-Admin-Key 인증 필수)
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from monitoring.metrics_writer import get_metrics_writer
from serving.routers.health import _require_admin

logger = logging.getLogger(__name__)

router = APIRouter(tags=["metrics"])


class MetricsResponse(BaseModel):
    records: list[dict]
    count: int
    hours: int


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics(
    hours: int = 24,
    _: None = Depends(_require_admin),
) -> MetricsResponse:
    """최근 N시간 예측 메트릭 조회.

    X-Admin-Key 헤더 인증 필수. ADMIN_API_KEY 미설정 시 503 반환.
    """
    try:
        records = get_metrics_writer().read_recent(hours=hours)
    except Exception:
        logger.warning("메트릭 읽기 실패 — 빈 목록 반환", exc_info=True)
        records = []
    return MetricsResponse(records=records, count=len(records), hours=hours)
