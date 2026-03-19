"""
FastAPI 미들웨어

- 요청 로깅: 메서드, 경로, 상태코드, 소요시간
- 요청 ID 헤더 주입 (X-Request-ID)
- 에러 핸들링: 예상치 못한 예외를 500으로 변환
"""
from __future__ import annotations

import logging
import time
import uuid

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """요청/응답 로깅 + Request-ID 주입."""

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = str(uuid.uuid4())[:8]
        start = time.perf_counter()

        # 요청 헤더에 ID 추가
        request.state.request_id = request_id

        try:
            response = await call_next(request)
        except Exception as exc:
            logger.exception("[%s] 처리되지 않은 예외: %s", request_id, exc)
            response = JSONResponse(
                status_code=500,
                content={"detail": "내부 서버 오류", "request_id": request_id},
            )

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "[%s] %s %s → %d (%.1fms)",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Elapsed-Ms"] = f"{elapsed_ms:.1f}"
        return response
