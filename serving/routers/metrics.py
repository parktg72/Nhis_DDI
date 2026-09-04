"""
메트릭 조회 엔드포인트

GET /metrics             - 최근 24시간 예측 메트릭 조회 (JSON, X-Admin-Key 필수)
GET /metrics/prometheus  - Prometheus 스크레이프용 텍스트 노출 (X-Admin-Key 필수)

`/metrics` 는 종전 JSON 조회 계약이므로 형식을 바꾸지 않는다. Prometheus 노출은
별도 경로로 낸다 — 종전에는 노출 경로가 pushgateway push 뿐이어서 대시보드가
스크레이프할 대상이 없었다(M7).

인증을 유지하는 이유 — 폐쇄망이라도 이 값은 민감정보에서 파생된 집계다. 무인증
노출은 관리 엔드포인트 인증 정책과 어긋난다. Prometheus 는 스크레이프 설정에
헤더를 넣을 수 있다(배포 런북 참조).
"""
import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

# 메트릭 **정의**를 등록시키기 위한 import. 이것이 없으면 어느 라우터가 먼저
# import 되었는지에 따라 노출 내용이 달라진다. 노출하는 쪽이 등록을 소유한다.
import monitoring.metrics  # noqa: F401
from monitoring.metrics_writer import get_metrics_writer
from serving.routers.health import _require_admin

logger = logging.getLogger(__name__)

try:
    from prometheus_client import CONTENT_TYPE_LATEST, REGISTRY, generate_latest
    _PROMETHEUS_AVAILABLE = True
except ImportError:      # 폐쇄망 최소 설치 대비 — 장애가 아니라 구성 문제다
    _PROMETHEUS_AVAILABLE = False

router = APIRouter(tags=["metrics"])


@router.get("/metrics/prometheus", response_class=PlainTextResponse)
async def get_metrics_prometheus(_: None = Depends(_require_admin)) -> PlainTextResponse:
    """Prometheus 텍스트 노출. 스크레이프 대상이 되는 유일한 경로다."""
    if not _PROMETHEUS_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="prometheus_client 미설치 — 스크레이프 노출 불가 (인메모리 폴백 모드)",
        )
    return PlainTextResponse(
        content=generate_latest(REGISTRY).decode("utf-8"),
        media_type=CONTENT_TYPE_LATEST,
    )


class MetricsResponse(BaseModel):
    records: list[dict]
    count: int
    hours: int


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics(
    hours: int = Query(default=24, ge=1, le=8760),
    _: None = Depends(_require_admin),
) -> MetricsResponse:
    """최근 N시간 예측 메트릭 조회.

    X-Admin-Key 헤더 인증 필수. ADMIN_API_KEY 미설정 시 503 반환.
    hours: 1 ~ 8760 (최대 365일)
    """
    try:
        records = get_metrics_writer().read_recent(hours=hours)
    except RuntimeError as exc:
        logger.error("MetricsWriter 초기화 안 됨: %s", exc)
        raise HTTPException(status_code=503, detail="메트릭 서비스 초기화되지 않음")
    except Exception:
        logger.warning("메트릭 읽기 실패 — 빈 목록 반환", exc_info=True)
        records = []
    return MetricsResponse(records=records, count=len(records), hours=hours)
