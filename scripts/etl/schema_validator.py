"""
T20/T30/T40/T60/요양기관 스키마 검증

실제 NHIS 레이아웃 기준 (lay_out/t20.txt ~ t60.txt, 요양기관.txt)
검증 항목:
  - 필수 컬럼 존재 여부
  - 날짜 형식 (YYYYMMDD / YYYYMM)
  - 코드 범위 (성별, 기관종류, 지급여부)
  - 수치 범위 (투여일수 > 0)
"""
from __future__ import annotations

import re
from typing import Optional

import pandas as pd

from .models import (
    T20_SCHEMA, T30_SCHEMA, T40_SCHEMA, T60_SCHEMA, YOYANG_SCHEMA,
    ValidationResult,
)


_DATE8_RE  = re.compile(r"^\d{8}$")   # YYYYMMDD
_DATE6_RE  = re.compile(r"^\d{6}$")   # YYYYMM
_SEX_CODES = {"1", "2"}               # SEX_TYPE: 1=남, 2=여
_PAY_CODES = {"1", "2", "0"}          # PAY_YN:   1=지급, 2=불지급, 0=미결

# YOYANG_CLSFC_CD — 요양기관 종별 코드 (2자리)
_YOYANG_TYPE_CODES = {
    "01",  # 상급종합병원
    "03",  # 종합병원
    "05",  # 병원
    "06",  # 요양병원
    "07",  # 정신병원
    "11",  # 의원
    "21",  # 약국
    "28",  # 보건소
    "29",  # 보건지소
    "31",  # 치과병원
    "32",  # 치과의원
    "33",  # 한방병원
    "34",  # 한의원
    "41",  # 조산원
}

# 필수(NOT NULL) 컬럼 — 각 테이블별
_T20_REQUIRED  = ["CMN_KEY", "INDI_DSCM_NO", "MDCARE_STRT_DT", "MDCARE_SYM"]
_T30_REQUIRED  = ["CMN_KEY", "INDI_DSCM_NO", "TOT_MCNT"]  # WK_COMPN_CD는 RVSN_WK_COMPN_CD 대체 허용
_T40_REQUIRED  = ["CMN_KEY", "INDI_DSCM_NO", "MCEX_SICK_SYM"]
_T60_REQUIRED  = ["CMN_KEY", "INDI_DSCM_NO", "MPRSC_GRANT_NO", "TOT_MCNT"]
_YOY_REQUIRED  = ["STD_YYYY", "MDCARE_SYM", "YOYANG_CLSFC_CD"]


# ─────────────────────────────────────────────────────────────────────────────
# 내부 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

def _check_columns(df: pd.DataFrame, schema: dict[str, str]) -> list[str]:
    """스키마에 정의된 컬럼 중 DataFrame에 없는 것 목록."""
    return [c for c in schema if c not in df.columns]


def _check_date_format(df: pd.DataFrame, col: str, pattern: re.Pattern) -> int:
    """날짜 형식 위반 건수 (null 제외)."""
    if col not in df.columns:
        return 0
    mask = df[col].dropna().astype(str).str.fullmatch(pattern.pattern)
    return int((~mask).sum())


def _check_nulls(df: pd.DataFrame, required_cols: list[str]) -> list[str]:
    """필수 컬럼 중 null 비율 5% 초과 목록."""
    violations = []
    for c in required_cols:
        if c in df.columns:
            null_rate = df[c].isna().mean()
            if null_rate > 0.05:
                violations.append(f"{c}: null {null_rate:.1%}")
    return violations


def _check_code_range(df: pd.DataFrame, col: str, valid_codes: set[str]) -> int:
    """유효 코드 범위 벗어난 건수 (null 제외)."""
    if col not in df.columns:
        return 0
    return int((~df[col].dropna().astype(str).isin(valid_codes)).sum())


# ─────────────────────────────────────────────────────────────────────────────
# 테이블별 검증 함수
# ─────────────────────────────────────────────────────────────────────────────

def validate_t20(df: pd.DataFrame) -> ValidationResult:
    """진료명세서(T20) 검증. 필수 컬럼(_T20_REQUIRED)만 missing 체크."""
    missing = [c for c in _T20_REQUIRED if c not in df.columns]
    if missing:
        return ValidationResult(
            table="T20",
            total_rows=len(df),
            valid_rows=0,
            invalid_rows=len(df),
            missing_cols=missing,
        )

    # 날짜 형식
    invalid = (
        _check_date_format(df, "MDCARE_STRT_DT", _DATE8_RE)
        + _check_date_format(df, "MDCARE_STRT_YYYYMM", _DATE6_RE)
    )

    # 성별 코드
    invalid += _check_code_range(df, "SEX_TYPE", _SEX_CODES)

    # 지급여부
    invalid += _check_code_range(df, "PAY_YN", _PAY_CODES)

    null_viols = _check_nulls(df, _T20_REQUIRED)
    return ValidationResult(
        table="T20",
        total_rows=len(df),
        valid_rows=max(0, len(df) - invalid),
        invalid_rows=invalid,
        null_violations=null_viols,
    )


def validate_t30(df: pd.DataFrame) -> ValidationResult:
    """진료내역(T30) 검증 — WK_COMPN_CD 기반 DDI 매칭 핵심 테이블."""
    missing = [c for c in _T30_REQUIRED if c not in df.columns]
    if missing:
        return ValidationResult(
            table="T30",
            total_rows=len(df),
            valid_rows=0,
            invalid_rows=len(df),
            missing_cols=missing,
        )

    # WK_COMPN_CD / RVSN_WK_COMPN_CD: 테이블 수준 컬럼 존재 확인
    has_wk_col = "WK_COMPN_CD" in df.columns
    has_rvsn_col = "RVSN_WK_COMPN_CD" in df.columns
    if not has_wk_col and not has_rvsn_col:
        return ValidationResult(
            table="T30",
            total_rows=len(df),
            valid_rows=0,
            invalid_rows=len(df),
            missing_cols=["WK_COMPN_CD (또는 RVSN_WK_COMPN_CD)"],
        )

    # 행 단위: 두 코드 모두 null인 행은 prescriptions_from_df에서 무음 삭제됨
    wk_null = df["WK_COMPN_CD"].isna() if has_wk_col else pd.Series(True, index=df.index)
    rvsn_null = df["RVSN_WK_COMPN_CD"].isna() if has_rvsn_col else pd.Series(True, index=df.index)
    both_missing_count = int((wk_null & rvsn_null).sum())

    # 투여일수 음수/0 체크
    invalid = both_missing_count  # 행 단위 성분코드 누락을 invalid에 포함
    if "TOT_MCNT" in df.columns:
        invalid += int((df["TOT_MCNT"].fillna(0) <= 0).sum())

    # 1일횟수 음수 체크
    if "DD1_MQTY_FREQ" in df.columns:
        invalid += int((df["DD1_MQTY_FREQ"].fillna(0) < 0).sum())

    # 날짜 형식
    invalid += _check_date_format(df, "MDCARE_STRT_DT", _DATE8_RE)

    # 성별 코드
    invalid += _check_code_range(df, "SEX_TYPE", _SEX_CODES)

    null_viols = _check_nulls(df, _T30_REQUIRED)
    if both_missing_count:
        null_viols.append(
            f"WK_COMPN_CD+RVSN_WK_COMPN_CD 둘 다 null: {both_missing_count}행 (ETL에서 무음 삭제됨)"
        )
    return ValidationResult(
        table="T30",
        total_rows=len(df),
        valid_rows=max(0, len(df) - invalid),
        invalid_rows=invalid,
        null_violations=null_viols,
    )


def validate_t40(df: pd.DataFrame) -> ValidationResult:
    """상병내역(T40) 검증.

    Note: T40은 진단 코드 테이블이며 인구통계(성별/연령)는 T20에 있음.
    """
    missing = [c for c in _T40_REQUIRED if c not in df.columns]
    if missing:
        return ValidationResult(
            table="T40",
            total_rows=len(df),
            valid_rows=0,
            invalid_rows=len(df),
            missing_cols=missing,
        )

    invalid = 0

    # SEX_TYPE 코드 범위
    invalid += _check_code_range(df, "SEX_TYPE", _SEX_CODES)

    # 날짜 형식
    invalid += _check_date_format(df, "MDCARE_STRT_DT", _DATE8_RE)

    # ICD-10 형식 간단 체크 (A~Z 시작, 3자 이상)
    if "MCEX_SICK_SYM" in df.columns:
        bad_icd = (~df["MCEX_SICK_SYM"].dropna().astype(str)
                   .str.match(r"^[A-Z]\d{2}")).sum()
        invalid += int(bad_icd)

    null_viols = _check_nulls(df, _T40_REQUIRED)
    return ValidationResult(
        table="T40",
        total_rows=len(df),
        valid_rows=max(0, len(df) - invalid),
        invalid_rows=invalid,
        null_violations=null_viols,
    )


def validate_t60(df: pd.DataFrame) -> ValidationResult:
    """처방전내역(T60) 검증 — 원외처방 약품 목록 (의원/병원 → 약국)."""
    missing = [c for c in _T60_REQUIRED if c not in df.columns]
    if missing:
        return ValidationResult(
            table="T60",
            total_rows=len(df),
            valid_rows=0,
            invalid_rows=len(df),
            missing_cols=missing,
        )

    # 투여일수
    invalid = 0
    if "TOT_MCNT" in df.columns:
        invalid += int((df["TOT_MCNT"].fillna(0) <= 0).sum())

    # 날짜 형식
    invalid += _check_date_format(df, "MDCARE_STRT_DT", _DATE8_RE)

    # 성별 코드
    invalid += _check_code_range(df, "SEX_TYPE", _SEX_CODES)

    # 지급여부
    invalid += _check_code_range(df, "PAY_YN", _PAY_CODES)

    null_viols = _check_nulls(df, _T60_REQUIRED)
    return ValidationResult(
        table="T60",
        total_rows=len(df),
        valid_rows=max(0, len(df) - invalid),
        invalid_rows=invalid,
        null_violations=null_viols,
    )


def validate_yoyang(df: pd.DataFrame) -> ValidationResult:
    """요양기관 현황 검증."""
    missing = [c for c in _YOY_REQUIRED if c not in df.columns]
    if missing:
        return ValidationResult(
            table="YOYANG",
            total_rows=len(df),
            valid_rows=0,
            invalid_rows=len(df),
            missing_cols=missing,
        )

    invalid = 0

    # 종별 코드
    invalid += _check_code_range(df, "YOYANG_CLSFC_CD", _YOYANG_TYPE_CODES)

    # 기준년도 형식 (4자리 숫자)
    if "STD_YYYY" in df.columns:
        bad = (~df["STD_YYYY"].dropna().astype(str)
               .str.fullmatch(r"\d{4}")).sum()
        invalid += int(bad)

    null_viols = _check_nulls(df, _YOY_REQUIRED)
    return ValidationResult(
        table="YOYANG",
        total_rows=len(df),
        valid_rows=max(0, len(df) - invalid),
        invalid_rows=invalid,
        null_violations=null_viols,
    )


# 하위 호환 — 이전 코드에서 validate_t50 참조 시 동작 유지
validate_t50 = validate_yoyang


# ─────────────────────────────────────────────────────────────────────────────
# 통합 검증
# ─────────────────────────────────────────────────────────────────────────────

def validate_all(
    t20: pd.DataFrame,
    t30: pd.DataFrame,
    t40: Optional[pd.DataFrame] = None,
    t60: Optional[pd.DataFrame] = None,
    yoyang: Optional[pd.DataFrame] = None,
) -> dict[str, ValidationResult]:
    """모든 테이블 일괄 검증. 필수: T20·T30, 선택: T40·T60·요양기관."""
    results: dict[str, ValidationResult] = {
        "T20": validate_t20(t20),
        "T30": validate_t30(t30),
    }
    if t40 is not None:
        results["T40"] = validate_t40(t40)
    if t60 is not None:
        results["T60"] = validate_t60(t60)
    if yoyang is not None:
        results["YOYANG"] = validate_yoyang(yoyang)
    return results
