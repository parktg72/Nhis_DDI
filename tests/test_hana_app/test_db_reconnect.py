"""db._execute_with_reconnect + 4 helper retry 회귀 가드 — Codex 2026-05-07 #4.

직전까지 query_df 만 1회 reconnect+retry 적용. get_row_count / preview /
get_date_range / get_distinct_values 4개는 cursor 직접 열고 예외 그대로 전파 →
Streamlit UI 미리보기/검증 경로에서 세션 만료 시 사용자가 재시도해야 했음.

본 테스트는:
  - _execute_with_reconnect 가 정상 케이스에서 cursor 작업 결과 그대로 반환
  - 첫 시도 실패 + connection 끊김 → reconnect() 호출 후 재시도 → 성공
  - 첫 시도 실패 + connection 정상 → reconnect 호출 안 하고 즉시 raise
  - 4 helper 각각이 mock conn 통해 SQL 실행 + retry 가드 사용
  - retry 정책 1회만 (query_df 와 일관)
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from hana_app.core.db import HANAConnection


def _make_conn_with_password() -> HANAConnection:
    """reconnect 동작을 위해 _password 가 필요. 실제 connect 우회."""
    c = HANAConnection()
    c._host = "h"
    c._port = 30015
    c._user = "u"
    c._password = "p"
    return c


# ─── _execute_with_reconnect 단위 ─────────────────────────────────────────────

class TestExecuteWithReconnect:

    def test_normal_path_returns_callback_result(self):
        c = HANAConnection()
        cur = MagicMock()
        c.conn = MagicMock()
        c.conn.cursor.return_value = cur

        result = c._execute_with_reconnect(lambda cursor: 42)
        assert result == 42
        cur.close.assert_called_once()

    def test_reconnects_when_connection_lost(self):
        """첫 시도 실패 + is_connected=False → reconnect() 호출 후 재시도."""
        c = _make_conn_with_password()
        cur1, cur2 = MagicMock(), MagicMock()
        # 첫 호출 시 cur1 (실패) → reconnect 후 cur2 (성공)
        c.conn = MagicMock()
        c.conn.cursor.side_effect = [cur1, cur2]

        c.is_connected = MagicMock(return_value=False)
        c.reconnect = MagicMock()

        calls = {"n": 0}

        def run(cursor):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("session expired")
            return "second_attempt_ok"

        result = c._execute_with_reconnect(run)
        assert result == "second_attempt_ok"
        c.reconnect.assert_called_once()
        # 두 cursor 모두 close 보장
        cur1.close.assert_called_once()
        cur2.close.assert_called_once()

    def test_no_reconnect_when_connection_alive(self):
        """첫 시도 실패하지만 connection 살아있으면 reconnect 안 하고 raise."""
        c = _make_conn_with_password()
        cur = MagicMock()
        c.conn = MagicMock()
        c.conn.cursor.return_value = cur
        c.is_connected = MagicMock(return_value=True)
        c.reconnect = MagicMock()

        with pytest.raises(ValueError, match="bad query"):
            c._execute_with_reconnect(
                lambda cursor: (_ for _ in ()).throw(ValueError("bad query"))
            )
        c.reconnect.assert_not_called()

    def test_retry_only_once(self):
        """retry 정책: 1회만 (query_df 와 일관). 두 번째 시도도 실패하면 raise."""
        c = _make_conn_with_password()
        c.conn = MagicMock()
        c.conn.cursor.return_value = MagicMock()
        c.is_connected = MagicMock(return_value=False)
        c.reconnect = MagicMock()

        attempts = {"n": 0}

        def always_fail(cursor):
            attempts["n"] += 1
            raise RuntimeError("still broken")

        with pytest.raises(RuntimeError, match="still broken"):
            c._execute_with_reconnect(always_fail)
        # 첫 시도 + 1회 재시도 = 총 2회만
        assert attempts["n"] == 2
        c.reconnect.assert_called_once()

    def test_no_password_skips_reconnect(self):
        """password 없으면 reconnect 시도 안 하고 즉시 raise (recover 불가)."""
        c = HANAConnection()  # password 미설정
        c.conn = MagicMock()
        c.conn.cursor.return_value = MagicMock()
        c.is_connected = MagicMock(return_value=False)
        c.reconnect = MagicMock()

        with pytest.raises(RuntimeError, match="boom"):
            c._execute_with_reconnect(
                lambda cursor: (_ for _ in ()).throw(RuntimeError("boom"))
            )
        c.reconnect.assert_not_called()


# ─── 4 helper 통합 — _execute_with_reconnect 사용 검증 ────────────────────────

class TestHelperRetryIntegration:

    def _make_conn(self):
        c = _make_conn_with_password()
        c.conn = MagicMock()
        return c

    def test_get_row_count_uses_reconnect_helper(self):
        c = self._make_conn()
        cur = MagicMock()
        cur.fetchone.return_value = (12345,)
        c.conn.cursor.return_value = cur

        assert c.get_row_count("SCHEMA", "TBL") == 12345
        cur.execute.assert_called_once()
        cur.close.assert_called_once()

    def test_get_row_count_retries_on_session_loss(self):
        c = self._make_conn()
        cur1, cur2 = MagicMock(), MagicMock()
        cur1.execute.side_effect = RuntimeError("session timeout")
        cur2.fetchone.return_value = (777,)
        c.conn.cursor.side_effect = [cur1, cur2]
        c.is_connected = MagicMock(return_value=False)
        c.reconnect = MagicMock()

        assert c.get_row_count("SCHEMA", "TBL") == 777
        c.reconnect.assert_called_once()

    def test_preview_uses_reconnect_helper(self):
        c = self._make_conn()
        cur = MagicMock()
        cur.description = [("A",), ("B",)]
        cur.fetchall.return_value = [(1, 2), (3, 4)]
        c.conn.cursor.return_value = cur

        df = c.preview("SCHEMA", "TBL", limit=10)
        assert list(df.columns) == ["A", "B"]
        assert len(df) == 2

    def test_preview_retries_on_session_loss(self):
        c = self._make_conn()
        cur1 = MagicMock()
        cur1.execute.side_effect = RuntimeError("session timeout")
        cur2 = MagicMock()
        cur2.description = [("X",)]
        cur2.fetchall.return_value = [(1,)]
        c.conn.cursor.side_effect = [cur1, cur2]
        c.is_connected = MagicMock(return_value=False)
        c.reconnect = MagicMock()

        df = c.preview("SCHEMA", "TBL")
        assert len(df) == 1
        c.reconnect.assert_called_once()

    def test_get_date_range_uses_reconnect_helper(self):
        c = self._make_conn()
        cur = MagicMock()
        cur.fetchone.return_value = ("20240101", "20241231")
        c.conn.cursor.return_value = cur

        result = c.get_date_range("SCHEMA", "TBL", "DATE_COL")
        assert result == {"min": "20240101", "max": "20241231"}

    def test_get_date_range_retries_on_session_loss(self):
        c = self._make_conn()
        cur1, cur2 = MagicMock(), MagicMock()
        cur1.execute.side_effect = RuntimeError("session timeout")
        cur2.fetchone.return_value = ("20230101", "20230531")
        c.conn.cursor.side_effect = [cur1, cur2]
        c.is_connected = MagicMock(return_value=False)
        c.reconnect = MagicMock()

        result = c.get_date_range("SCHEMA", "TBL", "DT")
        assert result == {"min": "20230101", "max": "20230531"}
        c.reconnect.assert_called_once()

    def test_get_distinct_values_uses_reconnect_helper(self):
        c = self._make_conn()
        cur = MagicMock()
        cur.fetchall.return_value = [("A",), ("B",), ("C",)]
        c.conn.cursor.return_value = cur

        result = c.get_distinct_values("SCHEMA", "TBL", "COL", limit=10)
        assert result == ["A", "B", "C"]

    def test_get_distinct_values_retries_on_session_loss(self):
        c = self._make_conn()
        cur1, cur2 = MagicMock(), MagicMock()
        cur1.execute.side_effect = RuntimeError("session timeout")
        cur2.fetchall.return_value = [("Z",)]
        c.conn.cursor.side_effect = [cur1, cur2]
        c.is_connected = MagicMock(return_value=False)
        c.reconnect = MagicMock()

        result = c.get_distinct_values("SCHEMA", "TBL", "COL")
        assert result == ["Z"]
        c.reconnect.assert_called_once()
