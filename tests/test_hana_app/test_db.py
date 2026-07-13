"""hana_app/core/db.py 단위 테스트."""
from unittest.mock import patch

import pytest

from hana_app.core.db import HANAConnection, get_connection


class TestGetConnection:
    def test_none_returns_fallback(self):
        """session_state=None → _fallback_conn 반환."""
        from hana_app.core import db as _db_module
        result = get_connection(None)
        assert result is _db_module._fallback_conn

    def test_no_arg_returns_fallback(self):
        """인자 없이 호출 → _fallback_conn 반환 (하위 호환)."""
        from hana_app.core import db as _db_module
        result = get_connection()
        assert result is _db_module._fallback_conn

    def test_creates_conn_per_session(self):
        """session_state별로 별도 HANAConnection 생성."""
        s1: dict = {}
        s2: dict = {}
        c1 = get_connection(s1)
        c2 = get_connection(s2)
        assert isinstance(c1, HANAConnection)
        assert isinstance(c2, HANAConnection)
        assert c1 is not c2

    def test_same_session_returns_same_conn(self):
        """동일 session_state는 같은 객체 반환."""
        s: dict = {}
        c1 = get_connection(s)
        c2 = get_connection(s)
        assert c1 is c2

    def test_stores_conn_in_session_state(self):
        """연결 객체가 session_state['hana_conn']에 저장됨."""
        s: dict = {}
        conn = get_connection(s)
        assert s["hana_conn"] is conn


class TestEnsureConnected:
    CREDS = {"host": "h", "port": 30015, "user": "u", "password": "p"}

    def test_reconnects_when_disconnected(self):
        """is_connected()=False → connect() 호출."""
        conn = HANAConnection()
        with patch.object(conn, "is_connected", return_value=False):
            with patch.object(conn, "connect") as mock_connect:
                conn.ensure_connected(self.CREDS)
        mock_connect.assert_called_once_with(
            host="h", port=30015, user="u", password="p"
        )

    def test_skips_when_already_connected(self):
        """is_connected()=True → connect() 미호출."""
        conn = HANAConnection()
        with patch.object(conn, "is_connected", return_value=True):
            with patch.object(conn, "connect") as mock_connect:
                conn.ensure_connected(self.CREDS)
        mock_connect.assert_not_called()

    def test_ttl_cache_skips_is_connected(self):
        """TTL 캐시 유효 시 is_connected() 호출 없이 통과."""
        import time
        conn = HANAConnection()
        session: dict = {"_conn_ok_until": time.monotonic() + 100}
        with patch.object(conn, "is_connected") as mock_check:
            conn.ensure_connected(self.CREDS, session_state=session)
        mock_check.assert_not_called()

    def test_ttl_cache_set_after_connect(self):
        """연결 후 session_state['_conn_ok_until'] 가 미래 시각으로 설정됨."""
        import time
        conn = HANAConnection()
        session: dict = {}
        with patch.object(conn, "is_connected", return_value=False):
            with patch.object(conn, "connect"):
                conn.ensure_connected(self.CREDS, session_state=session, ttl_seconds=5)
        assert session.get("_conn_ok_until", 0) > time.monotonic()

    def test_propagates_connect_exception(self):
        """connect() 실패 시 예외 전파."""
        conn = HANAConnection()
        with patch.object(conn, "is_connected", return_value=False):
            with patch.object(conn, "connect", side_effect=RuntimeError("DB down")):
                with pytest.raises(RuntimeError, match="DB down"):
                    conn.ensure_connected(self.CREDS)

    def test_stale_ttl_calls_is_connected(self):
        """TTL 만료 시 is_connected() 가 호출된다."""
        import time
        conn = HANAConnection()
        session: dict = {"_conn_ok_until": time.monotonic() - 1}  # 이미 만료
        with patch.object(conn, "is_connected", return_value=True) as mock_check:
            conn.ensure_connected(self.CREDS, session_state=session)
        mock_check.assert_called_once()

    def test_string_port_coerced_to_int(self):
        """port가 문자열로 전달되면 int로 변환해 connect()를 호출한다."""
        conn = HANAConnection()
        creds_str_port = {"host": "h", "port": "30015", "user": "u", "password": "p"}
        with patch.object(conn, "is_connected", return_value=False):
            with patch.object(conn, "connect") as mock_connect:
                conn.ensure_connected(creds_str_port)
        mock_connect.assert_called_once_with(host="h", port=30015, user="u", password="p")

    def test_ttl_set_when_already_connected_with_session(self):
        """이미 연결된 상태 + session_state 제공 시에도 TTL이 갱신된다."""
        import time
        conn = HANAConnection()
        session: dict = {}
        with patch.object(conn, "is_connected", return_value=True):
            with patch.object(conn, "connect") as mock_connect:
                conn.ensure_connected(self.CREDS, session_state=session, ttl_seconds=5)
        mock_connect.assert_not_called()
        assert session.get("_conn_ok_until", 0) > time.monotonic()
