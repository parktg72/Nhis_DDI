"""
데이터 품질 검사
- null 비율
- 중복 레코드
- 날짜 이상 (미래 날짜)
- WK_COMPN_CD 미매핑 비율
- 투여일수 이상값

실제 NHIS 레이아웃 기준 (lay_out/t20.txt, t30.txt)
"""
from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd

from .models import QualityReport

logger = logging.getLogger(__name__)

_TODAY = datetime.today().strftime("%Y%m%d")


def check_t20(df: pd.DataFrame) -> QualityReport:
    """진료명세서(T20) 품질 검사."""
    report = QualityReport(table="T20", total_rows=len(df))

    # null 비율 (핵심 컬럼)
    for col in ["CMN_KEY", "INDI_DSCM_NO", "MDCARE_STRT_DT", "MDCARE_SYM"]:
        if col in df.columns:
            report.null_rates[col] = float(df[col].isna().mean())

    # 중복 (CMN_KEY는 명세서 PK)
    if "CMN_KEY" in df.columns:
        report.duplicate_rate = float(df["CMN_KEY"].duplicated().mean())
        if report.duplicate_rate >= 0.05:
            report.warnings.append(f"중복율 {report.duplicate_rate:.1%} (임계값 5% 초과)")

    # 미래 날짜
    if "MDCARE_STRT_DT" in df.columns:
        future = int((df["MDCARE_STRT_DT"].astype(str) > _TODAY).sum())
        if future > 0:
            report.date_anomalies = future
            report.warnings.append(f"미래 시작일 {future}건")

    return report


def check_t30(
    df: pd.DataFrame,
    wk_compn_unknown_rate: float = 0.0,
) -> QualityReport:
    """진료내역(T30) 품질 검사 — WK_COMPN_CD 기반."""
    report = QualityReport(table="T30", total_rows=len(df))
    report.wk_compn_unknown_rate = wk_compn_unknown_rate

    # null 비율
    for col in ["CMN_KEY", "INDI_DSCM_NO", "WK_COMPN_CD", "TOT_MCNT"]:
        if col in df.columns:
            report.null_rates[col] = float(df[col].isna().mean())

    # 투여일수 이상값
    if "TOT_MCNT" in df.columns:
        bad_zero = int((df["TOT_MCNT"].fillna(0) <= 0).sum())
        if bad_zero:
            report.warnings.append(f"투여일수 0 이하: {bad_zero}건")
        bad_long = int((df["TOT_MCNT"] > 365).sum())
        if bad_long:
            report.warnings.append(f"투여일수 365일 초과: {bad_long}건")

    # WK_COMPN_CD 미매핑 경고
    if wk_compn_unknown_rate >= 0.30:
        report.warnings.append(
            f"주성분코드(WK_COMPN_CD) 미매핑율 {wk_compn_unknown_rate:.1%} (임계값 30% 초과)"
        )

    # 중복 (동일 명세서+EDI 코드)
    if {"CMN_KEY", "MCARE_DIV_CD"}.issubset(df.columns):
        report.duplicate_rate = float(
            df.duplicated(subset=["CMN_KEY", "MCARE_DIV_CD"]).mean()
        )
        if report.duplicate_rate >= 0.05:
            report.warnings.append(f"중복율 {report.duplicate_rate:.1%} (임계값 5% 초과)")

    return report


# 하위 호환 — pipeline.py에서 edi_unknown_rate 파라미터로 호출하는 경우 지원
def check_t30_legacy(
    df: pd.DataFrame,
    edi_unknown_rate: float = 0.0,
) -> QualityReport:
    return check_t30(df, wk_compn_unknown_rate=edi_unknown_rate)


def check_all(
    t20: pd.DataFrame,
    t30: pd.DataFrame,
    wk_compn_unknown_rate: float = 0.0,
    # 하위 호환 파라미터
    edi_unknown_rate: float = 0.0,
) -> dict[str, QualityReport]:
    # 두 파라미터 중 큰 값 사용 (하위 호환)
    unknown_rate = max(wk_compn_unknown_rate, edi_unknown_rate)
    return {
        "T20": check_t20(t20),
        "T30": check_t30(t30, wk_compn_unknown_rate=unknown_rate),
    }


def print_quality_summary(reports: dict[str, QualityReport]) -> None:
    print("\n" + "=" * 60)
    print("[데이터 품질 검사 결과]")
    print("=" * 60)
    for table, rep in reports.items():
        status = "PASS" if rep.passed else "FAIL"
        print(f"  {table} [{status}] {rep.total_rows:,}행")
        for col, rate in rep.null_rates.items():
            if rate > 0:
                print(f"    null {col}: {rate:.1%}")
        if rep.duplicate_rate > 0:
            print(f"    중복율: {rep.duplicate_rate:.1%}")
        if rep.date_anomalies > 0:
            print(f"    날짜이상: {rep.date_anomalies}건")
        for w in rep.warnings:
            print(f"    [경고] {w}")
    print("=" * 60)
