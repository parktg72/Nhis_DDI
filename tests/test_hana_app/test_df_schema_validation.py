"""hana_etl._validate_df_columns + fetch_*_by_date / fetch_t* schema 검증
회귀 가드 — Codex 2026-05-07 #4.

배경: 직전까지 fetch 결과 DataFrame 의 column presence 검증 부재. SQL SELECT
contract 가 깨지거나 (HANA 측 column rename / driver 동작 변경 등) downstream
이 silent 잘못된 schema 처리. 또 patient_ids=[] 시 빈 DataFrame 이 columns 없는
상태로 반환되어 downstream schema validation 호환 안 됨.

본 PR:
  - _validate_df_columns(df, required, context) helper — missing 시 ValueError
  - _query_paged_by_pid 에 expected_columns 인자 추가 — 빈 DF 도 columns 계약 유지
  - fetch_t{20,30,40,60} + fetch_t{20,30,40,60}_by_date 8개 함수 적용

테스트:
  - helper 단위 (정상 / missing / 빈 DF)
  - FakeConn 정상 → fetch 성공
  - FakeConn missing column → ValueError
  - patient_ids=[] → 빈 DF 라도 expected columns 유지
  - 기존 _normalize_yyyymmdd 테스트 호환
"""
from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import pandas as pd
import pytest

from hana_app.core.hana_etl import (
    HANAExtractor,
    _normalize_yyyymmdd,
    _validate_df_columns,
)


# ─── helper 단위 ────────────────────────────────────────────────────────────


def test_validate_passes_when_all_required_present():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "extra": [5, 6]})
    _validate_df_columns(df, ["a", "b"], "test")  # 예외 없이 통과


def test_validate_passes_for_empty_df_with_columns_contract():
    """빈 DF 도 columns 계약 유지되면 통과 (patient_ids=[] 경로)."""
    df = pd.DataFrame(columns=["a", "b"])
    assert df.empty
    _validate_df_columns(df, ["a", "b"], "test")  # 예외 없이 통과


def test_validate_raises_when_required_missing():
    df = pd.DataFrame({"a": [1]})  # 'b' 없음
    with pytest.raises(ValueError, match="schema drift"):
        _validate_df_columns(df, ["a", "b"], "fetch_test")


def test_validate_message_includes_context_and_actual_preview():
    df = pd.DataFrame({"x1": [], "x2": [], "x3": []})
    with pytest.raises(ValueError) as exc:
        _validate_df_columns(df, ["x1", "missing_col"], "my_context")
    msg = str(exc.value)
    assert "my_context" in msg
    assert "missing_col" in msg


# ─── fetch 통합 — FakeConn 으로 정상 / missing / 빈 결과 검증 ─────────────────

_T20_COLS = {
    "bill_no": "CMN_KEY", "patient_id": "INDI_DSCM_NO",
    "institution_id": "MDCARE_SYM", "start_date": "MDCARE_STRT_DT",
    "sex": "SEX_TYPE", "age_id": "SUJIN_POTM_AGE_ID",
    "institution_type": "YOYANG_CLSFC_CD",
    "yyyymm": "MDCARE_STRT_YYYYMM",
}
_T20_TABLE = {"schema": "NHISBASE", "table": "HBMT_TBGJME20"}


def _make_extractor(conn):
    return HANAExtractor(
        conn,
        {"t20": _T20_TABLE},
        {"t20": _T20_COLS},
    )


def test_fetch_t20_by_date_passes_with_correct_columns():
    """FakeConn 가 정상 columns DataFrame 반환 → fetch 성공."""
    captured = {}

    class FakeConn:
        def query_df(self, sql, params):
            captured["sql"] = sql
            return pd.DataFrame(
                {
                    "CMN_KEY": ["a"], "INDI_DSCM_NO": ["p1"],
                    "MDCARE_SYM": ["m"], "MDCARE_STRT_DT": ["20240101"],
                    "SEX_TYPE": ["F"], "SUJIN_POTM_AGE_ID": [50],
                    "YOYANG_CLSFC_CD": ["01"],
                }
            )

    extractor = _make_extractor(FakeConn())
    df = extractor.fetch_t20_by_date("20240101", "20241231")
    assert len(df) == 1
    assert "CMN_KEY" in df.columns


def test_fetch_t20_by_date_raises_on_missing_column():
    """FakeConn 가 required column 누락된 DataFrame 반환 → ValueError."""

    class BadConn:
        def query_df(self, sql, params):
            # MDCARE_STRT_DT 누락
            return pd.DataFrame(
                {
                    "CMN_KEY": ["a"], "INDI_DSCM_NO": ["p1"],
                    "MDCARE_SYM": ["m"],
                    "SEX_TYPE": ["F"], "SUJIN_POTM_AGE_ID": [50],
                    "YOYANG_CLSFC_CD": ["01"],
                }
            )

    extractor = _make_extractor(BadConn())
    with pytest.raises(ValueError, match="schema drift"):
        extractor.fetch_t20_by_date("20240101", "20241231")


def test_fetch_t20_by_date_empty_patient_ids_keeps_columns_contract():
    """patient_ids=[] 시 빈 DF 라도 expected columns 유지 → schema validation 통과."""

    class FakeConn:
        def query_df(self, sql, params):
            # patient_ids=[] 분기에서는 _query_paged_by_pid 가 SQL 호출 안 함
            raise AssertionError("query_df should not be called for empty patient_ids")

    extractor = _make_extractor(FakeConn())
    df = extractor.fetch_t20_by_date(
        "20240101", "20241231", patient_ids=[]
    )
    assert df.empty
    # Codex #4 핵심: 빈 결과도 columns 계약 유지
    expected_cols = [
        "CMN_KEY", "INDI_DSCM_NO", "MDCARE_SYM", "MDCARE_STRT_DT",
        "SEX_TYPE", "SUJIN_POTM_AGE_ID", "YOYANG_CLSFC_CD",
    ]
    for col in expected_cols:
        assert col in df.columns, (
            f"빈 DF 라도 '{col}' 컬럼 계약 유지되어야 함 (downstream schema validation 호환)"
        )


def test_fetch_t20_by_date_empty_patient_ids_passes_validate():
    """빈 DF + columns 계약 → _validate_df_columns 통과 (ValueError 없음)."""

    extractor = _make_extractor(MagicMock())
    # 호출 자체가 ValueError 안 냄 — fetch 정상 종료
    df = extractor.fetch_t20_by_date(
        "20240101", "20241231", patient_ids=[]
    )
    assert df.empty
    assert "CMN_KEY" in df.columns


def test_fetch_t20_by_date_iso_date_normalize_still_works():
    """ISO 입력 정규화 (직전 #ETL) + 새 schema validation 함께 통과."""
    captured = {}

    class FakeConn:
        def query_df(self, sql, params):
            captured["params"] = params
            return pd.DataFrame(
                {
                    "CMN_KEY": ["a"], "INDI_DSCM_NO": ["p1"],
                    "MDCARE_SYM": ["m"], "MDCARE_STRT_DT": ["20240115"],
                    "SEX_TYPE": ["M"], "SUJIN_POTM_AGE_ID": [40],
                    "YOYANG_CLSFC_CD": ["01"],
                }
            )

    extractor = _make_extractor(FakeConn())
    df = extractor.fetch_t20_by_date("2024-01-15", date(2024, 12, 31))
    assert len(df) == 1
    assert captured["params"][0] == "20240115"
    assert captured["params"][1] == "20241231"
