"""
ETL 공유 데이터 모델
건강보험 청구 데이터 스키마 및 결과 데이터클래스 정의
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# 청구 데이터 스키마 (컬럼명 → dtype)
# ─────────────────────────────────────────────────────────────────────────────

T20_SCHEMA: dict[str, str] = {
    "MDCARE_STRT_DT": "str",    # 요양개시일자 YYYYMMDD
    "MDCARE_END_DT":  "str",    # 요양종료일자 YYYYMMDD
    "MDCARE_BILL_NO": "str",    # 명세서ID
    "BNFCR_PSEUDO":   "str",    # 수진자 가명ID (SHA-256)
    "INST_PSEUDO":    "str",    # 요양기관 가명ID
    "SICK_SYM":       "str",    # 주상병코드 (ICD-10)
}

T30_SCHEMA: dict[str, str] = {
    "MDCARE_BILL_NO":  "str",   # 명세서ID (T20 FK)
    "EDI_CD":          "str",   # 의약품 EDI 코드
    "DOSG_ONCE":       "float", # 1회 투여량
    "DOSG_FREQ_DY":    "int",   # 1일 투여횟수
    "MEDTIME_FRQ_CNT": "int",   # 총 투여일수
}

T40_SCHEMA: dict[str, str] = {
    "BNFCR_PSEUDO": "str",   # 수진자 가명ID
    "SEX_TP_CD":    "str",   # 성별 (1=남, 2=여)
    "BTH_YYYY":     "str",   # 출생연도 YYYY
}

T50_SCHEMA: dict[str, str] = {
    "INST_PSEUDO":  "str",   # 요양기관 가명ID
    "CLNC_TP_CD":   "str",   # 기관종류 (1=의원, 3=병원, 11=종합, 21=약국)
}


# ─────────────────────────────────────────────────────────────────────────────
# 처방 레코드 (T20+T30 조인 결과)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PrescriptionRecord:
    """단일 처방 항목 (약물 한 종류)."""
    patient_id: str
    institution_id: str
    bill_no: str
    edi_code: str
    atc_code: Optional[str]
    drug_name: Optional[str]
    start_date: date
    end_date: date          # start_date + total_days - 1
    total_days: int
    dose_once: float
    dose_freq: int
    sick_code: Optional[str] = None
    institution_type: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# 동시복용 약물 쌍
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DrugOverlapPair:
    """두 약물의 동시복용 기간 정보."""
    patient_id: str
    drug_a_edi: str
    drug_b_edi: str
    drug_a_atc: Optional[str]
    drug_b_atc: Optional[str]
    drug_a_name: Optional[str]
    drug_b_name: Optional[str]
    overlap_start: date
    overlap_end: date
    overlap_days: int        # 중첩 일수
    window_start: date       # 90일 윈도우 시작
    window_end: date         # 90일 윈도우 종료


# ─────────────────────────────────────────────────────────────────────────────
# 환자별 피처 벡터 (Feature Store 저장 단위)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PatientFeatures:
    """환자 한 명의 90일 윈도우 피처 집합."""
    patient_id: str
    window_start: date
    window_end: date

    # 다재약물 기본 피처
    drug_count: int = 0             # 고유 약물 수
    drug_count_7d: int = 0          # 최근 7일 동시 복용 수
    institution_count: int = 0      # 처방 요양기관 수

    # DDI 피처
    ddi_contraindicated: int = 0    # Contraindicated DDI 쌍 수
    ddi_major: int = 0              # Major DDI 쌍 수
    ddi_moderate: int = 0           # Moderate DDI 쌍 수
    ddi_minor: int = 0              # Minor DDI 쌍 수
    triple_whammy: bool = False     # Triple Whammy 해당 여부
    qt_risk_count: int = 0          # QT연장 위험약물 수

    # 중복약물 피처
    dup_same_ingredient: int = 0    # 동일 성분 중복 수
    dup_atc5: int = 0               # ATC 5단계 중복
    dup_atc4: int = 0               # ATC 4단계 중복 (효능군)
    dup_atc3: int = 0               # ATC 3단계 중복

    # 환자 인구통계 (T40)
    age: Optional[int] = None
    sex: Optional[str] = None       # M/F

    # 위험도 레이블 (Rule-based)
    risk_level: str = "Normal"      # Red/Yellow/Green/Normal
    risk_reasons: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# 스키마 검증 결과
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    table: str
    total_rows: int
    valid_rows: int
    invalid_rows: int
    missing_cols: list[str] = field(default_factory=list)
    type_errors: list[str] = field(default_factory=list)
    null_violations: list[str] = field(default_factory=list)

    @property
    def valid_rate(self) -> float:
        return self.valid_rows / self.total_rows if self.total_rows > 0 else 0.0

    @property
    def passed(self) -> bool:
        return not self.missing_cols and self.valid_rate >= 0.95


# ─────────────────────────────────────────────────────────────────────────────
# 품질 검사 결과
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QualityReport:
    table: str
    total_rows: int
    null_rates: dict[str, float] = field(default_factory=dict)
    duplicate_rate: float = 0.0
    date_anomalies: int = 0         # 날짜 역전 건수 (start > end)
    edi_unknown_rate: float = 0.0   # ATC 매핑 불가 EDI 비율
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return (
            self.duplicate_rate < 0.05
            and self.date_anomalies == 0
            and self.edi_unknown_rate < 0.30
        )


# ─────────────────────────────────────────────────────────────────────────────
# 파이프라인 실행 결과
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """전체 ETL 파이프라인 실행 결과 요약."""
    partition: str                          # 처리 파티션 (YYYYMM)
    total_patients: int = 0
    total_prescriptions: int = 0
    total_drug_items: int = 0
    overlap_pairs: int = 0
    features_written: int = 0
    red_count: int = 0
    yellow_count: int = 0
    green_count: int = 0
    normal_count: int = 0
    validation: Optional[ValidationResult] = None
    quality: Optional[QualityReport] = None
    elapsed_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return len(self.errors) == 0
