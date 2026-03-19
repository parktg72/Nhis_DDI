"""
T20/T30/T40/T50 스키마 검증
- 필수 컬럼 존재 여부
- 날짜 형식 (YYYYMMDD)
- 코드 범위 (성별, 기관종류)
"""
from __future__ import annotations

import re
from typing import Optional

import pandas as pd

from .models import T20_SCHEMA, T30_SCHEMA, T40_SCHEMA, T50_SCHEMA, ValidationResult


_DATE_RE = re.compile(r"^\d{8}$")
_SEX_CODES = {"1", "2"}
_INST_TYPE_CODES = {"1", "2", "3", "11", "21", "28", "29", "31"}  # 의원/병원/종합/약국 등


def _check_columns(df: pd.DataFrame, schema: dict[str, str]) -> list[str]:
    return [c for c in schema if c not in df.columns]


def _check_dates(df: pd.DataFrame, col: str) -> int:
    """YYYYMMDD 형식 위반 건수."""
    if col not in df.columns:
        return 0
    mask = df[col].dropna().astype(str).str.match(_DATE_RE)
    return int((~mask).sum())


def _check_nulls(df: pd.DataFrame, cols: list[str]) -> list[str]:
    """필수 컬럼 중 null 비율 5% 초과 컬럼 목록."""
    violations = []
    for c in cols:
        if c in df.columns:
            null_rate = df[c].isna().mean()
            if null_rate > 0.05:
                violations.append(f"{c}: null {null_rate:.1%}")
    return violations


def validate_t20(df: pd.DataFrame) -> ValidationResult:
    missing = _check_columns(df, T20_SCHEMA)
    if missing:
        return ValidationResult(
            table="T20",
            total_rows=len(df),
            valid_rows=0,
            invalid_rows=len(df),
            missing_cols=missing,
        )

    date_err = _check_dates(df, "MDCARE_STRT_DT") + _check_dates(df, "MDCARE_END_DT")

    # 날짜 역전 검사
    if "MDCARE_STRT_DT" in df.columns and "MDCARE_END_DT" in df.columns:
        reversed_mask = df["MDCARE_STRT_DT"] > df["MDCARE_END_DT"]
        date_err += int(reversed_mask.sum())

    null_viols = _check_nulls(df, ["MDCARE_BILL_NO", "BNFCR_PSEUDO", "MDCARE_STRT_DT"])
    invalid = date_err
    return ValidationResult(
        table="T20",
        total_rows=len(df),
        valid_rows=len(df) - invalid,
        invalid_rows=invalid,
        null_violations=null_viols,
    )


def validate_t30(df: pd.DataFrame) -> ValidationResult:
    missing = _check_columns(df, T30_SCHEMA)
    if missing:
        return ValidationResult(
            table="T30",
            total_rows=len(df),
            valid_rows=0,
            invalid_rows=len(df),
            missing_cols=missing,
        )

    # 투여일수 음수/0 체크
    invalid = 0
    if "MEDTIME_FRQ_CNT" in df.columns:
        bad = (df["MEDTIME_FRQ_CNT"].fillna(0) <= 0).sum()
        invalid += int(bad)

    null_viols = _check_nulls(df, ["MDCARE_BILL_NO", "EDI_CD"])
    return ValidationResult(
        table="T30",
        total_rows=len(df),
        valid_rows=len(df) - invalid,
        invalid_rows=invalid,
        null_violations=null_viols,
    )


def validate_t40(df: pd.DataFrame) -> ValidationResult:
    missing = _check_columns(df, T40_SCHEMA)
    if missing:
        return ValidationResult(
            table="T40",
            total_rows=len(df),
            valid_rows=0,
            invalid_rows=len(df),
            missing_cols=missing,
        )

    invalid = 0
    if "SEX_TP_CD" in df.columns:
        bad = (~df["SEX_TP_CD"].astype(str).isin(_SEX_CODES)).sum()
        invalid += int(bad)

    null_viols = _check_nulls(df, ["BNFCR_PSEUDO"])
    return ValidationResult(
        table="T40",
        total_rows=len(df),
        valid_rows=len(df) - invalid,
        invalid_rows=invalid,
        null_violations=null_viols,
    )


def validate_t50(df: pd.DataFrame) -> ValidationResult:
    missing = _check_columns(df, T50_SCHEMA)
    if missing:
        return ValidationResult(
            table="T50",
            total_rows=len(df),
            valid_rows=0,
            invalid_rows=len(df),
            missing_cols=missing,
        )

    null_viols = _check_nulls(df, ["INST_PSEUDO"])
    return ValidationResult(
        table="T50",
        total_rows=len(df),
        valid_rows=len(df),
        invalid_rows=0,
        null_violations=null_viols,
    )


def validate_all(
    t20: pd.DataFrame,
    t30: pd.DataFrame,
    t40: Optional[pd.DataFrame] = None,
    t50: Optional[pd.DataFrame] = None,
) -> dict[str, ValidationResult]:
    results: dict[str, ValidationResult] = {
        "T20": validate_t20(t20),
        "T30": validate_t30(t30),
    }
    if t40 is not None:
        results["T40"] = validate_t40(t40)
    if t50 is not None:
        results["T50"] = validate_t50(t50)
    return results
