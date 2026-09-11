"""
메트릭 조회 엔드포인트

GET /metrics             - 최근 24시간 예측 메트릭 조회 (JSON, X-Admin-Key 필수)
GET /metrics/prometheus  - Prometheus 스크레이프용 텍스트 노출

`/metrics` 는 종전 JSON 조회 계약이므로 형식을 바꾸지 않는다. Prometheus 노출은
별도 경로로 낸다 — 종전에는 노출 경로가 pushgateway push 뿐이어서 대시보드가
스크레이프할 대상이 없었다(M7).

**인증** — 폐쇄망이라도 이 값은 민감정보에서 파생된 집계이므로 무인증으로 열지
않는다. 다만 관리자 키는 모델 교체 권한까지 갖는다. 스크레이프에 그 키를 주면
Prometheus 설정 파일에 관리 권한이 들어간다. 그래서 `METRICS_SCRAPE_KEY` 를
따로 받고, 이 키는 **노출 경로에서만** 통한다. 미설정 시에는 종전대로 관리자
키를 받는다(하위 호환).

**worker 가 하나일 때만 맞는 값이다.** 이 경로는 **요청을 받은 프로세스의**
레지스트리를 내보낸다. uvicorn 을 여러 worker 로 띄우면 총량이 누락되고 카운터가
리셋된 것처럼 보인다.

합산 수집기 경로를 넣었다가 **뺐다.** 근거로 삼았던 `--workers 4` 가 저장소에서
DEPRECATED 로 표시된 컨테이너 파일과 진입점 docstring 에만 있었고, 운영 실행
경로(Windows 폐쇄망)의 worker 수는 확인되지 않았다. 확인되지 않은 배포 형태를
겨냥해 코드를 남기지 않는다. **worker 수는 A0 ⑤ 가 실측한다** — 노출을 여러 번
읽어 서로 다른 프로세스 시작 시각이 몇 개인지 센다. 하나가 아니면 이 값은
집계가 아니며, 그때 합산 경로를 다시 검토한다.
"""
import logging
import os

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

# 메트릭 **정의**를 등록시키기 위한 import. 이것이 없으면 어느 라우터가 먼저
# import 되었는지에 따라 노출 내용이 달라진다. 노출하는 쪽이 등록을 소유한다.
import monitoring.metrics  # noqa: F401
from monitoring.audit_log import audit
from monitoring.metrics_writer import get_metrics_writer
from serving.routers.health import _require_admin

logger = logging.getLogger(__name__)

try:
    from prometheus_client import CONTENT_TYPE_LATEST, REGISTRY, generate_latest
    _PROMETHEUS_AVAILABLE = True
except ImportError:      # 폐쇄망 최소 설치 대비 — 장애가 아니라 구성 문제다
    _PROMETHEUS_AVAILABLE = False

router = APIRouter(tags=["metrics"])


def _require_scrape(
    request: Request,
    x_admin_key: str = Header(..., alias="X-Admin-Key"),
) -> None:
    """노출 경로 전용 인증.

    `METRICS_SCRAPE_KEY` 가 설정돼 있으면 그 키만 받는다 — 스크레이프 설정에
    관리 권한을 넣지 않기 위해서다. 미설정이면 관리자 키로 되돌아간다.
    """
    import hmac

    from serving.routers import health as _health

    ep = _health._audit_endpoint(request)
    ip = _health._client_ip(request)
    scrape_key = os.environ.get("METRICS_SCRAPE_KEY", "").strip()
    if scrape_key:
        if hmac.compare_digest(x_admin_key, scrape_key):
            audit("auth_ok", endpoint=ep, client_ip=ip, key_kind="scrape")
            return
        audit("auth_failed", endpoint=ep, client_ip=ip, key_kind="scrape")
        raise HTTPException(status_code=401, detail="스크레이프 인증 실패")

    if not _health._ADMIN_KEY:
        audit("auth_unconfigured", endpoint=ep, client_ip=ip, key_kind="scrape")
        raise HTTPException(
            status_code=503,
            detail="METRICS_SCRAPE_KEY·ADMIN_API_KEY 모두 미설정: 노출 경로 비활성화",
        )
    if not hmac.compare_digest(x_admin_key, _health._ADMIN_KEY):
        audit("auth_failed", endpoint=ep, client_ip=ip, key_kind="admin")
        raise HTTPException(status_code=401, detail="관리자 인증 실패")
    audit("auth_ok", endpoint=ep, client_ip=ip, key_kind="admin")


@router.get("/metrics/prometheus", response_class=PlainTextResponse)
async def get_metrics_prometheus(_: None = Depends(_require_scrape)) -> PlainTextResponse:
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
        # **호출**을 남기지 응답을 남기지 않는다 — 이 응답은 환자 단위 행이다.
        audit("admin_call", endpoint="/metrics", outcome="ok",
              hours=hours, returned_count=len(records))
    except RuntimeError as exc:
        logger.error("MetricsWriter 초기화 안 됨: %s", exc)
        audit("admin_call", endpoint="/metrics", outcome="writer_uninitialized",
              hours=hours, status_code=503)
        raise HTTPException(status_code=503, detail="메트릭 서비스 초기화되지 않음")
    except Exception:
        logger.warning("메트릭 읽기 실패 — 빈 목록 반환", exc_info=True)
        # 빈 목록이 "기록이 없다" 로 읽히지 않도록 읽기 실패 자체를 남긴다.
        audit("admin_call", endpoint="/metrics", outcome="read_failed", hours=hours)
        records = []
    return MetricsResponse(records=records, count=len(records), hours=hours)
