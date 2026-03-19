"""
데이터 품질 검사
- null 비율
- 중복 레코드
- 날짜 이상 (역전, 미래 날짜)
- EDI 미매핑 비율
- 투여일수 이상값
"""
from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd

from .models import QualityReport

logger = logging.getLogger(__name__)

_TODAY = datetime.today().strftime("%Y%m%d")


def check_t20(df: pd.DataFrame) -> QualityReport:
    report = QualityReport(table="T20", total_rows=len(df))

    # null 비율
    for col in ["MDCARE_BILL_NO", "BNFCR_PSEUDO", "MDCARE_STRT_DT", "MDCARE_END_DT"]:
        if col in df.columns:
            report.null_rates[col] = float(df[col].isna().mean())

    # 중복
    if "MDCARE_BILL_NO" in df.columns:
        report.duplicate_rate = float(df["MDCARE_BILL_NO"].duplicated().mean())

    # 날짜 역전
    if {"MDCARE_STRT_DT", "MDCARE_END_DT"}.issubset(df.columns):
        rev = (df["MDCARE_STRT_DT"].astype(str) > df["MDCARE_END_DT"].astype(str))
        report.date_anomalies = int(rev.sum())
        if report.date_anomalies:
            report.warnings.append(f"날짜 역전 {report.date_anomalies}건 (start > end)")

    # 미래 날짜
    if "MDCARE_STRT_DT" in df.columns:
        future = (df["MDCARE_STRT_DT"].astype(str) > _TODAY).sum()
        if future > 0:
            report.warnings.append(f"미래 시작일 {future}건")

    if report.duplicate_rate >= 0.05:
        report.warnings.append(f"중복율 {report.duplicate_rate:.1%} (임계값 5% 초과)")

    return report


def check_t30(
    df: pd.DataFrame,
    edi_unknown_rate: float = 0.0,
) -> QualityReport:
    report = QualityReport(table="T30", total_rows=len(df))
    report.edi_unknown_rate = edi_unknown_rate

    for col in ["MDCARE_BILL_NO", "EDI_CD", "MEDTIME_FRQ_CNT"]:
        if col in df.columns:
            report.null_rates[col] = float(df[col].isna().mean())

    # 투여일수 이상값
    if "MEDTIME_FRQ_CNT" in df.columns:
        # 0 이하
        bad_zero = int((df["MEDTIME_FRQ_CNT"].fillna(0) <= 0).sum())
        if bad_zero:
            report.warnings.append(f"투여일수 0 이하: {bad_zero}건")
        # 365일 초과 (1년 초과 처방은 이상)
        bad_long = int((df["MEDTIME_FRQ_CNT"] > 365).sum())
        if bad_long:
            report.warnings.append(f"투여일수 365일 초과: {bad_long}건")

    if edi_unknown_rate >= 0.30:
        report.warnings.append(f"EDI 미매핑율 {edi_unknown_rate:.1%} (임계값 30% 초과)")

    # 중복 (동일 명세서+EDI)
    if {"MDCARE_BILL_NO", "EDI_CD"}.issubset(df.columns):
        report.duplicate_rate = float(
            df.duplicated(subset=["MDCARE_BILL_NO", "EDI_CD"]).mean()
        )

    return report


def check_all(
    t20: pd.DataFrame,
    t30: pd.DataFrame,
    edi_unknown_rate: float = 0.0,
) -> dict[str, QualityReport]:
    return {
        "T20": check_t20(t20),
        "T30": check_t30(t30, edi_unknown_rate=edi_unknown_rate),
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
