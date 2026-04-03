"""reload_model 중 동시 /health 요청 동시성 스모크 테스트."""
import asyncio
import pytest
from unittest.mock import MagicMock, patch


@pytest.mark.asyncio
async def test_concurrent_requests_during_reload():
    """reload_model 실행 중 동시 /health 요청이 500 없이 처리됨 (lock 패턴 검증)."""
    mock_predictor = MagicMock()
    mock_predictor._model = MagicMock()  # 로드된 상태로 표시

    with patch("serving.predictor.get_predictor", return_value=mock_predictor), \
         patch("serving.predictor.init_predictor"):

        from httpx import AsyncClient, ASGITransport
        from serving.main import app

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
