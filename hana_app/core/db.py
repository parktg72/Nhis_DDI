"""
SAP HANA DB 연결 관리
"""
from __future__ import annotations

import re
from typing import Any, Callable, TypeVar

import pandas as pd

T = TypeVar("T")

_SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_$#]{1,128}$")


def _assert_safe_identifier(value: str, label: str = "identifier") -> None:
    """schema/table/column 이름이 HANA 식별자 규칙을 벗어나면 ValueError 발생.

    double-quote 이스케이프만으로는 충분하지 않으므로
    외부 입력을 카탈로그에서 검증하기 전에 1차 형식 방어선으로 사용.
    """
    if not _SAFE_IDENTIFIER_RE.match(value):
        raise ValueError(
            f"안전하지 않은 {label}: {value!r}. "
            "영문자·숫자·언더스코어·$·# 만 허용됩니다."
        )


class HANAConnection:
    """hdbcli 래퍼 – 연결 수명 관리 및 편의 메서드 제공."""

    def __init__(self) -> None:
        self.conn = None
        self._host = ""
        self._port = 30015
        self._user = ""
        self._password = ""

    # ── 연결 / 해제 ────────────────────────────────────────────────────────

    def connect(self, host: str, port: int, user: str, password: str) -> None:
        from hdbcli import dbapi  # 폐쇄망 설치 후에만 import

        if self.conn:
            self.close()
        self.conn = dbapi.connect(
            address=str(host).strip(),
            port=int(port),
            user=str(user).strip(),
            password=str(password),
        )
        self._host = host
        self._port = port
        self._user = user
        self._password = password

    def reconnect(self) -> None:
        """저장된 자격증명으로 HANA 세션 재연결 (장시간 추출 중 세션 만료 대응)."""
        if not self._password:
            raise RuntimeError("재연결 불가: 저장된 자격증명 없음 (connect() 미호출)")
        self.connect(self._host, self._port, self._user, self._password)

    def close(self) -> None:
        if self.conn:
            try:
                self.conn.close()
            except Exception:
                pass
        self.conn = None

    def ensure_connected(
        self,
        creds: dict,
        session_state: dict | None = None,
        ttl_seconds: int = 5,
    ) -> None:
        """연결이 끊겼으면 creds로 자동 재연결.

        creds 구조: {"host": str, "port": int, "user": str, "password": str}

        session_state가 제공되면 TTL 캐시를 사용해 is_connected() DB 왕복을
        ttl_seconds 동안 생략한다 (Streamlit rerun 마다 호출되는 경우 성능 보호).

        이미 연결된 상태면 아무것도 하지 않는다.
        재연결 실패 시 hdbcli 예외를 그대로 전파한다.
        """
        import time

        # creds 필수 키 검증
        required = {"host", "port", "user", "password"}
        missing_keys = required - creds.keys()
        if missing_keys:
            raise ValueError(
                f"ensure_connected(): creds에 필수 키가 없습니다: {sorted(missing_keys)}"
            )

        now = time.monotonic()
        cache_key = "_conn_ok_until"

        # TTL 캐시 유효 → is_connected() 생략
        if session_state is not None:
            if now < session_state.get(cache_key, 0):
                return

        if not self.is_connected():
            self.connect(
                host=creds["host"],
                port=int(creds["port"]),
                user=creds["user"],
                password=creds["password"],
            )

        # 연결 확인 후 TTL 갱신
        if session_state is not None:
            session_state[cache_key] = now + ttl_seconds

    # ── 상태 ───────────────────────────────────────────────────────────────

    def is_connected(self) -> bool:
        if not self.conn:
            return False
        try:
            cur = self.conn.cursor()
            cur.execute("SELECT 1 FROM DUMMY")
            return True
        except Exception:
            return False

    def server_info(self) -> dict[str, str]:
        if not self.conn:
            return {}
        try:
            cur = self.conn.cursor()
            cur.execute("SELECT * FROM M_DATABASE LIMIT 1")
            cols = [d[0] for d in cur.description]
            rows = cur.fetchone()
            return dict(zip(cols, rows)) if rows else {}
        except Exception:
            return {}

    # ── 스키마 / 테이블 목록 ──────────────────────────────────────────────

    def get_schemas(self, filter_prefix: str = "") -> list[str]:
        cur = self.conn.cursor()
        try:
            if filter_prefix:
                cur.execute(
                    "SELECT SCHEMA_NAME FROM SCHEMAS "
                    "WHERE SCHEMA_NAME LIKE ? ORDER BY SCHEMA_NAME",
                    (f"{filter_prefix}%",),
                )
            else:
                cur.execute(
                    "SELECT SCHEMA_NAME FROM SCHEMAS "
                    "WHERE HAS_PRIVILEGES = 'TRUE' ORDER BY SCHEMA_NAME"
                )
            return [r[0] for r in cur.fetchall()]
        finally:
            cur.close()

    def get_tables(self, schema: str) -> list[str]:
        cur = self.conn.cursor()
        try:
            cur.execute(
                "SELECT TABLE_NAME FROM TABLES WHERE SCHEMA_NAME = ? ORDER BY TABLE_NAME",
                (schema,),
            )
            return [r[0] for r in cur.fetchall()]
        finally:
            cur.close()

    def get_columns(self, schema: str, table: str) -> list[dict[str, str]]:
        cur = self.conn.cursor()
        try:
            cur.execute(
                "SELECT COLUMN_NAME, DATA_TYPE_NAME, IS_NULLABLE "
                "FROM TABLE_COLUMNS "
                "WHERE SCHEMA_NAME = ? AND TABLE_NAME = ? "
                "ORDER BY POSITION",
                (schema, table),
            )
            return [
                {"name": r[0], "type": r[1], "nullable": r[2]}
                for r in cur.fetchall()
            ]
        finally:
            cur.close()

    # ── cursor 재연결 helper ────────────────────────────────────────────────

    def _execute_with_reconnect(self, run: Callable[[Any], T]) -> T:
        """cursor 작업을 1회 reconnect+retry 로 보호 (Codex 2026-05-07 #4).

        query_df 가 직접 갖던 retry 패턴을 단발 helper(get_row_count/preview/
        get_date_range/get_distinct_values) 4건과 일관 적용. query_df 자체는
        chunked fetchmany 가독성 보존 위해 그대로 유지 (옵션 A).
        """
        for attempt in range(2):
            try:
                cur = self.conn.cursor()
                try:
                    return run(cur)
                finally:
                    cur.close()
            except Exception:
                if attempt == 0 and self._password and not self.is_connected():
                    self.reconnect()
                    continue
                raise
        raise RuntimeError("unreachable")  # pragma: no cover

    def get_row_count(self, schema: str, table: str) -> int:
        _assert_safe_identifier(schema, "schema")
        _assert_safe_identifier(table, "table")

        def _run(cur: Any) -> int:
            cur.execute(f'SELECT COUNT(*) FROM "{schema}"."{table}"')
            return cur.fetchone()[0]

        return self._execute_with_reconnect(_run)

    # ── 데이터 조회 ────────────────────────────────────────────────────────

    def preview(
        self, schema: str, table: str, limit: int = 50, offset: int = 0
    ) -> pd.DataFrame:
        _assert_safe_identifier(schema, "schema")
        _assert_safe_identifier(table, "table")
        limit = int(limit)
        offset = int(offset)
        if not (1 <= limit <= 10_000):
            raise ValueError(f"limit는 1~10,000 범위여야 합니다. (입력값: {limit})")
        if offset < 0:
            raise ValueError(f"offset은 0 이상이어야 합니다. (입력값: {offset})")

        def _run(cur: Any) -> pd.DataFrame:
            cur.execute(
                f'SELECT * FROM "{schema}"."{table}" LIMIT {limit} OFFSET {offset}'
            )
            cols = [d[0] for d in cur.description]
            return pd.DataFrame(cur.fetchall(), columns=cols)

        return self._execute_with_reconnect(_run)

    def query_df(
        self,
        sql: str,
        params: list[Any] | None = None,
        chunksize: int = 50_000,
    ) -> pd.DataFrame:
        """SQL 실행 → DataFrame 반환 (fetchmany 청크 방식으로 OOM 방지).

        chunksize : 한 번에 가져올 행 수 (기본 50,000).
        세션 만료(장시간 추출 중 HANA 세션 종료)가 감지되면 재연결 후 1회 재시도.
        """
        for attempt in range(2):
            try:
                cur = self.conn.cursor()
                try:
                    if params:
                        cur.execute(sql, params)
                    else:
                        cur.execute(sql)
                    cols = [d[0] for d in cur.description]

                    chunks: list[pd.DataFrame] = []
                    while True:
                        rows = cur.fetchmany(chunksize)
                        if not rows:
                            break
                        chunks.append(pd.DataFrame(rows, columns=cols))

                    if not chunks:
                        return pd.DataFrame(columns=cols)
                    if len(chunks) == 1:
                        return chunks[0]
                    return pd.concat(chunks, ignore_index=True)
                finally:
                    cur.close()
            except Exception as exc:
                # 세션이 끊긴 경우만 재연결 후 재시도 (쿼리 오류 258/139 등은 재시도 안 함)
                if attempt == 0 and self._password and not self.is_connected():
                    self.reconnect()
                    continue
                raise

    def get_date_range(self, schema: str, table: str, date_col: str) -> dict[str, str]:
        """테이블의 날짜 컬럼 최솟값·최댓값 반환."""
        _assert_safe_identifier(schema, "schema")
        _assert_safe_identifier(table, "table")
        _assert_safe_identifier(date_col, "column")

        def _run(cur: Any) -> dict[str, str]:
            cur.execute(
                f'SELECT MIN("{date_col}"), MAX("{date_col}") FROM "{schema}"."{table}"'
            )
            row = cur.fetchone()
            return {"min": str(row[0] or ""), "max": str(row[1] or "")}

        return self._execute_with_reconnect(_run)

    def get_distinct_values(
        self, schema: str, table: str, col: str, limit: int = 100
    ) -> list[str]:
        _assert_safe_identifier(schema, "schema")
        _assert_safe_identifier(table, "table")
        _assert_safe_identifier(col, "column")
        limit = int(limit)
        if not (1 <= limit <= 10_000):
            raise ValueError(f"limit는 1~10,000 범위여야 합니다. (입력값: {limit})")

        def _run(cur: Any) -> list[str]:
            cur.execute(
                f'SELECT DISTINCT "{col}" FROM "{schema}"."{table}" '
                f"WHERE \"{col}\" IS NOT NULL ORDER BY \"{col}\" LIMIT {limit}"
            )
            return [str(r[0]) for r in cur.fetchall()]

        return self._execute_with_reconnect(_run)


# public alias — table_validator 등 외부 모듈에서 사용
assert_safe_identifier = _assert_safe_identifier

# ── 전역 폴백 (테스트 / CLI / 비Streamlit 환경용) ─────────────────────────────
_fallback_conn = HANAConnection()


def get_connection(session_state: dict | None = None) -> HANAConnection:
    """세션별 격리된 HANAConnection 반환.

    session_state가 None이거나 생략되면 _fallback_conn 반환
    (테스트 / CLI / 비Streamlit 환경 하위 호환).
    Streamlit 환경에서는 반드시 st.session_state를 전달한다.
    """
    if session_state is None:
        return _fallback_conn
    if "hana_conn" not in session_state:
        session_state["hana_conn"] = HANAConnection()
    return session_state["hana_conn"]
