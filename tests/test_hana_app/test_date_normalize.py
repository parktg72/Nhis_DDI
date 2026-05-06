"""hana_etl._normalize_yyyymmdd 단위 테스트.

배경 (lay_out/6_테이블명_260428/HBMT_TBGJME{20,30,40,60}.txt):
  HANA 처방 테이블 4종의 MDCARE_STRT_DT 컬럼이 `NVARCHAR(8)` (YYYYMMDD 문자열).
  CAST AS DATE 가 아니라 입력 정규화 (str/date 객체 모두 8자리 YYYYMMDD 로 강제)
  가 정답 — Codex/Qwen 권고를 schema 1차 자료 근거로 정정 (2026-05-06).

helper 가 보호하는 결함:
  - 호출자가 "2024-01-01" 같은 ISO 형식을 넘기면 lexicographic 비교 실패 → 무음 0 row
  - datetime.date 객체 직접 바인딩 시 driver 형식 변환 불확정
"""
from __future__ import annotations

from datetime import date

import pytest

from hana_app.core.hana_etl import _normalize_yyyymmdd


def test_passthrough_yyyymmdd_string():
    assert _normalize_yyyymmdd("20240101") == "20240101"


def test_iso_format_dash_normalized():
    assert _normalize_yyyymmdd("2024-01-01") == "20240101"


def test_iso_format_slash_normalized():
    assert _normalize_yyyymmdd("2024/01/01") == "20240101"


def test_datetime_date_object_normalized():
    assert _normalize_yyyymmdd(date(2024, 1, 1)) == "20240101"


def test_whitespace_stripped():
    assert _normalize_yyyymmdd("  20240101  ") == "20240101"


def test_invalid_length_raises():
    with pytest.raises(ValueError, match="YYYYMMDD"):
        _normalize_yyyymmdd("2024010")  # 7자리


def test_non_digit_raises():
    with pytest.raises(ValueError, match="YYYYMMDD"):
        _normalize_yyyymmdd("2024Jan01")


def test_empty_string_raises():
    with pytest.raises(ValueError, match="YYYYMMDD"):
        _normalize_yyyymmdd("")


def test_none_raises():
    with pytest.raises((ValueError, TypeError)):
        _normalize_yyyymmdd(None)  # type: ignore[arg-type]


def test_invalid_month_raises():
    """13월 같은 잘못된 월 — len/digit 검증만으론 통과, strptime 으로 reject."""
    with pytest.raises(ValueError):
        _normalize_yyyymmdd("20241332")


def test_invalid_day_raises():
    """2월 30일 같은 잘못된 일 — len/digit 검증만으론 통과, strptime 으로 reject."""
    with pytest.raises(ValueError):
        _normalize_yyyymmdd("20240230")


def test_leap_year_feb29_passes():
    """윤년 2월 29일은 유효 — 정상 통과 보장."""
    assert _normalize_yyyymmdd("20240229") == "20240229"


def test_non_leap_year_feb29_raises():
    """비윤년 2월 29일은 invalid — strptime 으로 reject."""
    with pytest.raises(ValueError):
        _normalize_yyyymmdd("20230229")


_T20_COLS = {
    "bill_no": "CMN_KEY", "patient_id": "INDI_DSCM_NO",
    "institution_id": "MDCARE_SYM", "start_date": "MDCARE_STRT_DT",
    "sex": "SEX_TYPE", "age_id": "SUJIN_POTM_AGE_ID",
    "institution_type": "YOYANG_CLSFC_CD",
}
_T30_COLS = {
    "bill_no": "CMN_KEY", "patient_id": "INDI_DSCM_NO",
    "start_date": "MDCARE_STRT_DT",
    "drug_code": "WK_COMPN_CD", "drug_code_alt": "RVSN_WK_COMPN_CD",
    "edi_code": "MCARE_DIV_CD", "efmdc": "EFMDC_CLSF_NO",
    "dose_once": "TIME1_MDCT_CPCT", "dose_freq": "DD1_MQTY_FREQ",
    "total_days": "TOT_MCNT",
}
_T40_COLS = {
    "bill_no": "CMN_KEY", "patient_id": "INDI_DSCM_NO",
    "start_date": "MDCARE_STRT_DT",
    "sick_code": "MCEX_SICK_SYM", "sick_type": "SICK_CLSF_TYPE",
}
_T60_COLS = {
    # T60 production config (hana_app/hana_config.json:81): drug_code=GNL_NM_CD.
    # T60 schema (HBMT_TBGJME60.txt) 에 WK_COMPN_CD 컬럼 없음 — GNL_NM_CD/RVSN_WK_COMPN_CD 만 존재.
    "bill_no": "CMN_KEY", "patient_id": "INDI_DSCM_NO",
    "institution_id": "MDCARE_SYM", "start_date": "MDCARE_STRT_DT",
    "drug_code": "GNL_NM_CD", "drug_code_alt": "RVSN_WK_COMPN_CD",
    "edi_code": "MCARE_DIV_CD",
    "dose_once": "MPRSC_TIME1_TUYAK_CPCT", "dose_freq": "MPRSC_DD1_TUYAK_CPCT",
    "total_days": "TOT_MCNT", "sick_code": "SICK_SYM1",
}


@pytest.mark.parametrize("fetch_attr,table_key,cols", [
    ("fetch_t20_by_date", "t20", _T20_COLS),
    ("fetch_t30_by_date", "t30", _T30_COLS),
    ("fetch_t40_by_date", "t40", _T40_COLS),
    ("fetch_t60_by_date", "t60", _T60_COLS),
])
def test_fetch_by_date_normalizes_input(fetch_attr, table_key, cols):
    """fetch_t{20,30,40,60}_by_date 4개 모두 ISO/date 입력 정규화 회귀 가드."""
    from hana_app.core.hana_etl import HANAExtractor

    captured = {}

    class FakeConn:
        def query_df(self, sql, params):
            captured["sql"] = sql
            captured["params"] = params
            import pandas as pd
            return pd.DataFrame()

    table_cfg = {table_key: {"schema": "NHISBASE", "table": f"HBMT_TBGJME{table_key[1:]}"}}
    col_cfg = {table_key: cols}
    extractor = HANAExtractor(FakeConn(), table_cfg, col_cfg)

    getattr(extractor, fetch_attr)("2024-01-01", date(2024, 12, 31))

    assert captured["params"][0] == "20240101", (
        f"{fetch_attr}: date_from 정규화 누락 — bound={captured['params'][0]!r}"
    )
    assert captured["params"][1] == "20241231", (
        f"{fetch_attr}: date_to 정규화 누락 — bound={captured['params'][1]!r}"
    )
