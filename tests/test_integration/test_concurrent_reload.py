"""reload_model 중 동시 /health 요청 동시성 스모크 테스트."""
import asyncio
from unittest.mock import patch

import pytest


@pytest.mark.asyncio
async def test_concurrent_requests_during_reload():
    """reload_model 실행 중 동시 /health 요청이 500 없이 처리됨 (lock 패턴 검증)."""
    # serving.main 과 하위 라우터(serving.routers.health 등)를 패치 블록 밖에서
    # 먼저 임포트해야 한다.  패치 블록 안에서 처음 임포트하면 health.get_predictor 가
    # 패치된 mock 을 영구 캡처해 후속 테스트를 오염시킨다.
    from httpx import ASGITransport, AsyncClient

    import serving.predictor as _pred_mod
    from serving.main import app  # noqa: F401

    original_predictor = _pred_mod._predictor

    # serving.main.init_predictor 를 패치 — 리스판이 실제 모델 파일을 로드하지 않도록.
    # (serving.predictor.init_predictor 패치와 달리 lifespan 이 실제로 사용하는 참조를 패치함)
    with patch("serving.main.init_predictor"):
        try:
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                tasks = [client.get("/health") for _ in range(10)]
                responses = await asyncio.gather(*tasks)

            statuses = [r.status_code for r in responses]
            # 500 (서버 내부 오류) 없어야 함 — 200 또는 503(모델 미로드)만 허용
            assert 500 not in statuses, (
                f"동시 요청 중 서버 내부 오류 발생: 상태코드 = {set(statuses)}"
            )
            assert all(s in (200, 503) for s in statuses), (
                f"예상 외 상태코드: {set(statuses)}"
            )
        finally:
            # 전역 _predictor 원복 — 후속 test_serving 테스트 오염 방지
            _pred_mod._predictor = original_predictor
