"""
SAP HANA DB 연결 관리
"""
from __future__ import annotations

import re
from typing import Any

import pandas as pd

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

    def close(self) -> None:
        if self.conn:
            try:
                self.conn.close()
            except Exception:
                pass
        self.conn = None

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

    def get_row_count(self, schema: str, table: str) -> int:
        _assert_safe_identifier(schema, "schema")
        _assert_safe_identifier(table, "table")
        cur = self.conn.cursor()
        try:
            cur.execute(f'SELECT COUNT(*) FROM "{schema}"."{table}"')
            return cur.fetchone()[0]
        finally:
            cur.close()

    # ── 데이터 조회 ────────────────────────────────────────────────────────

    def preview(
        self, schema: str, table: str, limit: int = 50, offset: int = 0
    ) -> pd.DataFrame:
        _assert_safe_identifier(schema, "schema")
        _assert_safe_identifier(table, "table")
        cur = self.conn.cursor()
        try:
            cur.execute(
                f'SELECT * FROM "{schema}"."{table}" LIMIT {limit} OFFSET {offset}'
            )
            cols = [d[0] for d in cur.description]
            return pd.DataFrame(cur.fetchall(), columns=cols)
        finally:
            cur.close()

    def query_df(
        self,
        sql: str,
        params: list[Any] | None = None,
        chunksize: int = 50_000,
    ) -> pd.DataFrame:
        """SQL 실행 → DataFrame 반환 (fetchmany 청크 방식으로 OOM 방지).

        chunksize : 한 번에 가져올 행 수 (기본 50,000).
        """
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

    def get_date_range(self, schema: str, table: str, date_col: str) -> dict[str, str]:
        """테이블의 날짜 컬럼 최솟값·최댓값 반환."""
        _assert_safe_identifier(schema, "schema")
        _assert_safe_identifier(table, "table")
        _assert_safe_identifier(date_col, "column")
        cur = self.conn.cursor()
        try:
            cur.execute(
                f'SELECT MIN("{date_col}"), MAX("{date_col}") FROM "{schema}"."{table}"'
            )
            row = cur.fetchone()
            return {"min": str(row[0] or ""), "max": str(row[1] or "")}
        finally:
            cur.close()

    def get_distinct_values(
        self, schema: str, table: str, col: str, limit: int = 100
    ) -> list[str]:
        _assert_safe_identifier(schema, "schema")
        _assert_safe_identifier(table, "table")
        _assert_safe_identifier(col, "column")
        cur = self.conn.cursor()
        try:
            cur.execute(
                f'SELECT DISTINCT "{col}" FROM "{schema}"."{table}" '
                f"WHERE \"{col}\" IS NOT NULL ORDER BY \"{col}\" LIMIT {limit}"
            )
            return [str(r[0]) for r in cur.fetchall()]
        finally:
            cur.close()


# ── 전역 싱글톤 (Streamlit session_state 에서 참조) ─────────────────────────

_global_conn = HANAConnection()


def get_connection() -> HANAConnection:
    return _global_conn
