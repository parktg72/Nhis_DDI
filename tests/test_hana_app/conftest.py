"""tests/test_hana_app 전용 픽스처."""
import pytest


@pytest.fixture(autouse=True)
def reset_fallback_conn():
    """각 테스트 전후로 _fallback_conn을 새 인스턴스로 교체.

    _fallback_conn은 모듈 레벨 객체이므로 테스트 간 연결 상태가
    누출되는 것을 방지한다.
    """
    from hana_app.core import db as _db_module
    from hana_app.core.db import HANAConnection

    _db_module._fallback_conn = HANAConnection()
    yield
    _db_module._fallback_conn = HANAConnection()
