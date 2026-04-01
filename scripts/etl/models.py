"""
ETL 공유 데이터 모델
건강보험 청구 데이터 스키마 및 결과 데이터클래스 정의

실제 NHIS 레이아웃 기준 (lay_out/t20.txt ~ t60.txt, 요양기관.txt)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# 청구 데이터 스키마 (컬럼명 → dtype)
# 출처: lay_out/t20.txt, t30.txt, t40.txt, t60.txt, 요양기관.txt
# ─────────────────────────────────────────────────────────────────────────────

T20_SCHEMA: dict[str, str] = {
    # 식별자
    "CMN_KEY":                  "str",   # 공통키 (명세서 ID, 29자) — PK
    "INDI_DSCM_NO":             "str",   # 개인식별번호 (가명화)
    "MDCARE_SYM":               "str",   # 요양기관기호 (11자)
    # 날짜
    "MDCARE_STRT_YYYYMM":       "str",   # 요양개시년월 YYYYMM
    "MDCARE_STRT_DT":           "str",   # 요양개시일자 YYYYMMDD
    "HIRA_EXM_YYYYMM":          "str",   # 심사평가원심사년월
    # 상병 (주·부상병 최대 5개)
    "SICK_SYM1":                "str",   # 상병기호1 — 주상병 (ICD-10, 처방전 포함)
    "SICK_SYM2":                "str",   # 상병기호2 — 부상병
    "SICK_SYM3":                "str",   # 상병기호3
    "SICK_SYM4":                "str",   # 상병기호4
    "SICK_SYM5":                "str",   # 상병기호5
    # 원외처방
    "INSOUT_PRSC_GRANT_NO":     "str",   # 원외처방교부번호 (13자)
    "INSOUT_PRSC_MEDI_ITM_SU":  "int",   # 원외처방약품품목수
    "TOT_PRSC_DD_CNT":          "int",   # 총처방일수
    # 인구통계 (T20에 수진자 정보 포함)
    "SEX_TYPE":                 "str",   # 성별구분 (1=남, 2=여)
    "SUJIN_POTM_AGE_ID":        "str",   # 수진시점연령ID
    "YEND_POTM_AGE_ID":         "str",   # 연말시점연령ID
    # 행정
    "FORM_CD":                  "str",   # 서식코드
    "PAY_YN":                   "str",   # 지급여부
    "MCARE_TP":                 "str",   # 진료형태
    "WMED_OTMED_TYPE":          "str",   # 양방한방구분
    "YOYANG_CLSFC_CD":          "str",   # 요양기관종별코드
}

T30_SCHEMA: dict[str, str] = {
    # 식별자
    "CMN_KEY":          "str",    # 공통키 (T20 FK, 29자)
    "INDI_DSCM_NO":     "str",    # 개인식별번호
    "MDCARE_STRT_DT":   "str",    # 요양개시일자 YYYYMMDD
    # 약품 코드 — DDI 핵심 필드
    "WK_COMPN_CD":      "str",    # 주성분코드 (9자) ★ DDI 매칭 기준
    "RVSN_WK_COMPN_CD": "str",    # 보정주성분코드 (9자) ★ WK_COMPN_CD 우선, 없으면 사용
    "EFMDC_CLSF_NO":    "str",    # 약효분류번호 (5자) — 효능군중복 탐지용
    "MCARE_DIV_CD":     "str",    # 진료분류코드 (9자) = EDI 약품코드
    "MCARE_DIV_CD_NM":  "str",    # 진료분류코드명
    # 용법
    "TIME1_MDCT_CPCT":  "float",  # 1회투여용량
    "DD1_MQTY_FREQ":    "float",  # 1일투여량횟수
    "TOT_MCNT":         "int",    # 총투여일수 ★
    "DRUG_MDCT_CPCT":   "float",  # 약물투여용량 (누적)
    # 비용
    "UPRC":             "float",  # 단가
    "AMT":              "float",  # 금액
    # 행정
    "MDCARE_STRT_YYYYMM": "str",  # 요양개시년월
    "PAY_YN":           "str",    # 지급여부
    "FORM_CD":          "str",    # 서식코드
    "SEX_TYPE":         "str",    # 성별구분
    "SUJIN_POTM_AGE_ID":"str",    # 수진시점연령ID
}

T40_SCHEMA: dict[str, str] = {
    # 식별자 — T40은 상병내역(진단 목록)이며, 인구통계는 T20에 있음
    "CMN_KEY":           "str",   # 공통키 (T20 FK, 29자)
    "SICK_DESC_SEQ_NO":  "str",   # 상병내역일련번호 (2자)
    "INDI_DSCM_NO":      "str",   # 개인식별번호
    "MDCARE_STRT_DT":    "str",   # 요양개시일자 YYYYMMDD
    # 상병
    "MCEX_SICK_SYM":     "str",   # 요양급여상병기호 (ICD-10, 6자) ★
    "SICK_CLSF_TYPE":    "str",   # 상병분류구분 (주/부상병 구분)
    # 행정
    "DD_TYPE":           "str",   # 일구분
    "SEX_TYPE":          "str",   # 성별구분
    "SUJIN_POTM_AGE_ID": "str",   # 수진시점연령ID
    "YEND_POTM_AGE_ID":  "str",   # 연말시점연령ID
    "FORM_CD":           "str",   # 서식코드
    "PAY_YN":            "str",   # 지급여부
    "MDCARE_SYM":        "str",   # 요양기관기호
}

T60_SCHEMA: dict[str, str] = {
    # 식별자 — T60은 원외처방전 내역 (의원/병원 발행 처방전)
    "CMN_KEY":                  "str",    # 공통키 (T20 FK, 29자)
    "MPRSC_GRANT_NO":           "str",    # 처방전교부번호 (20자) ★ T20.INSOUT_PRSC_GRANT_NO와 연결
    "MPRSC_SEQ_NO":             "int",    # 처방전일련번호
    "MCARE_DESC_LN_NO":         "str",    # 진료내역줄번호 (4자)
    "INDI_DSCM_NO":             "str",    # 개인식별번호
    "MDCARE_STRT_DT":           "str",    # 요양개시일자 YYYYMMDD
    # 약품 코드
    "GNL_NM_CD":                "str",    # 일반명코드 (9자) ★ 원외처방 약품 식별
    "RVSN_WK_COMPN_CD":         "str",    # 보정주성분코드 (9자) ★ DDI 매칭용
    "MCARE_DIV_CD":             "str",    # 진료분류코드 (9자) = EDI 약품코드
    "MCARE_DIV_CD_NM":          "str",    # 진료분류코드명
    # 용법
    "MPRSC_TIME1_TUYAK_CPCT":   "float",  # 1회투약용량
    "MPRSC_DD1_TUYAK_CPCT":     "float",  # 1일투약용량
    "TOT_MCNT":                 "int",    # 총투여일수 ★
    # 비용
    "UPRC":                     "float",  # 단가
    "AMT":                      "float",  # 금액
    # 상병
    "SICK_SYM1":                "str",    # 상병기호1 (주상병)
    "SICK_SYM2":                "str",    # 상병기호2 (부상병)
    # 행정
    "MDCARE_STRT_YYYYMM":       "str",    # 요양개시년월
    "HIRA_EXM_YYYYMM":          "str",    # 심사평가원심사년월
    "FORM_CD":                  "str",    # 서식코드
    "PAY_YN":                   "str",    # 지급여부
    "SEX_TYPE":                 "str",    # 성별구분
    "SUJIN_POTM_AGE_ID":        "str",    # 수진시점연령ID
    "CLSF_PTTN_CD":             "str",    # 분류유형코드
    "DMD_TP_CD":                "str",    # 청구형태코드
    "MDCARE_SYM":               "str",    # 요양기관기호
}

YOYANG_SCHEMA: dict[str, str] = {
    # PK (기준년도 + 요양기관기호)
    "STD_YYYY":               "str",    # 기준년도 YYYY ★ PK(1)
    "MDCARE_SYM":             "str",    # 요양기관기호 (8자) ★ PK(2)  ← T20.MDCARE_SYM FK
    "STD_YYYYMM":             "str",    # 기준년월
    "EX_DT":                  "str",    # 추출일자 YYYYMMDD
    # 기관 분류
    "YOYANG_CLSFC_CD":        "str",    # 요양기관종별구분 (2자) ★ T20.YOYANG_CLSFC_CD와 동일
    "YOYANG_DETAIL_CLSFC_CD": "str",    # 요양기관세부종별코드 (2자)
    # 기관 정보
    "INST_NM":                "str",    # 요양기관이름
    "ADDR":                   "str",    # 요양기관주소
    "ADDR_SGG_CD":            "str",    # 시군구코드 (5자)
    "FDAT_TYPE_CD":           "str",    # 설립구분
    # 인력
    "CNT_MDR_TOT":            "float",  # 의과의사수
    "CNT_MDR_SPC":            "float",  # 의과전문의수
    "CNT_NRS_TOT":            "float",  # 간호사수
    "CNT_PHRM_TOT":           "float",  # 약사수
    # 병상
    "CNT_BED_PERM":           "float",  # 허가병상수
    # 상태
    "T20_Y":                  "str",    # T20 청구여부 (진료명세서 청구 기관 여부)
    "TMPCLS_Y":               "str",    # 휴업여부
    "HX_JOINED_Y":            "str",    # 요양기관현황연계 여부
}

# 하위 호환 — 이전 코드에서 T50_SCHEMA 참조 시 동작 유지
T50_SCHEMA = YOYANG_SCHEMA


# ─────────────────────────────────────────────────────────────────────────────
# 처방 레코드 (T20 + T30/T60 조인 결과)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PrescriptionRecord:
    """단일 처방 항목 (약물 한 종류).

    T30 기반 원내 투약 또는 T60 기반 원외처방을 통일된 형식으로 표현.
    - patient_id    ← INDI_DSCM_NO
    - institution_id← MDCARE_SYM  (T20)
    - bill_no       ← CMN_KEY     (T20)
    - wk_compn_cd   ← WK_COMPN_CD (T30) 또는 RVSN_WK_COMPN_CD (T30/T60)  ★ NHIS 주성분 식별자
    - edi_code      ← MCARE_DIV_CD (T30/T60)                               ★ EDI 약품코드
    - atc_code      ← CodeStandardizer 매핑 결과 (DrugBank)                 ★ DDI 매트릭스 조회용
    - gnl_nm_cd     ← GNL_NM_CD   (T60, 원외처방 전용)
    - efmdc_clsf_no ← EFMDC_CLSF_NO (T30)                                  ★ 효능군중복용
    - total_days    ← TOT_MCNT
    - dose_once     ← TIME1_MDCT_CPCT (T30) / MPRSC_TIME1_TUYAK_CPCT (T60)
    - dose_freq     ← DD1_MQTY_FREQ (T30)  / MPRSC_DD1_TUYAK_CPCT (T60)
    - sick_code     ← SICK_SYM1 (T20/T60) 또는 MCEX_SICK_SYM (T40)
    - source        ← "T30" (원내) | "T60" (원외처방)
    """
    patient_id:     str
    institution_id: str
    bill_no:        str           # CMN_KEY
    wk_compn_cd:    str           # 주성분코드 (NHIS 약물 식별자) ★
    edi_code:       Optional[str] = None  # MCARE_DIV_CD (EDI 코드)
    atc_code:       Optional[str] = None  # DrugBank ATC 코드 (DDI 매트릭스 조회용)
    gnl_nm_cd:      Optional[str] = None  # 일반명코드 (T60 원외처방)
    efmdc_clsf_no:  Optional[str] = None  # 약효분류번호 (효능군중복)
    drug_name:      Optional[str] = None
    start_date:     date = field(default_factory=date.today)
    end_date:       date = field(default_factory=date.today)  # start + total_days - 1
    total_days:     int = 1
    dose_once:      float = 1.0
    dose_freq:      int = 1
    sick_code:      Optional[str] = None
    sex:            Optional[str] = None   # SEX_TYPE: "1"=남, "2"=여
    age_id:         Optional[str] = None   # SUJIN_POTM_AGE_ID
    institution_type: Optional[str] = None # YOYANG_CLSFC_CD
    source:         str = "T30"            # "T30" | "T60"


# ─────────────────────────────────────────────────────────────────────────────
# 동시복용 약물 쌍
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DrugOverlapPair:
    """두 약물의 동시복용 기간 정보."""
    patient_id:      str
    # 약물 A
    drug_a_wk_compn: str           # WK_COMPN_CD (NHIS 주성분코드)
    drug_a_edi:      Optional[str]  # MCARE_DIV_CD (EDI)
    drug_a_atc:      Optional[str]  # ATC 코드 (DDI 매트릭스용)
    drug_a_name:     Optional[str]
    # 약물 B
    drug_b_wk_compn: str
    drug_b_edi:      Optional[str]
    drug_b_atc:      Optional[str]
    drug_b_name:     Optional[str]
    # 중첩 기간
    overlap_start:  date
    overlap_end:    date
    overlap_days:   int        # 중첩 일수
    window_start:   date       # 90일 윈도우 시작
    window_end:     date       # 90일 윈도우 종료


# ─────────────────────────────────────────────────────────────────────────────
# 환자별 피처 벡터 (Feature Store 저장 단위)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PatientFeatures:
    """환자 한 명의 90일 윈도우 피처 집합."""
    patient_id:   str   # INDI_DSCM_NO
    window_start: date
    window_end:   date

    # 다재약물 기본 피처
    drug_count:         int = 0   # 고유 약물 수 (WK_COMPN_CD 기준)
    drug_count_7d:      int = 0   # 최근 7일 동시 복용 수
    institution_count:  int = 0   # 처방 요양기관 수

    # DDI 피처
    ddi_contraindicated: int = 0  # Contraindicated DDI 쌍 수
    ddi_major:           int = 0  # Major DDI 쌍 수
    ddi_moderate:        int = 0  # Moderate DDI 쌍 수
    ddi_minor:           int = 0  # Minor DDI 쌍 수
    triple_whammy:       bool = False  # Triple Whammy 해당 여부
    qt_risk_count:       int = 0       # QT연장 위험약물 수

    # 중복약물 피처
    dup_same_ingredient: int = 0  # 동일 주성분 중복 수 (WK_COMPN_CD)
    dup_atc5:            int = 0  # ATC 5단계 중복
    dup_atc4:            int = 0  # ATC 4단계 중복 (효능군)
    dup_atc3:            int = 0  # ATC 3단계 중복
    dup_efmdc:           int = 0  # 약효분류(EFMDC_CLSF_NO) 중복

    # 고위험 약물 플래그 (CLINICAL_STANDARDS 위험도 판정 기준)
    has_high_risk_drug:    bool = False  # 고위험 약물 포함 여부
    has_renal_risk_drug:   bool = False  # 신기능 저하 위험 약물 포함
    has_hepatic_risk_drug: bool = False  # 간기능 저하 위험 약물 포함

    # CYP450 피처 (약물대사 상호작용 위험)
    cyp_risk_score:       float = 0.0   # CYP 전체 위험 점수 (가중합)
    cyp_max_enzyme_risk:  float = 0.0   # 가장 위험한 단일 CYP 효소 위험도
    cyp_high_risk_pairs:  int = 0       # 억제제+기질 동시복용 쌍 수

    # 환자 인구통계
    age:     Optional[int] = None
    sex:     Optional[str] = None  # "1"=남, "2"=여 (SEX_TYPE)
    age_id:  Optional[str] = None  # SUJIN_POTM_AGE_ID (연령 범주 ID)

    # 위험도 레이블 (Rule-based)
    risk_level:   str = "Normal"  # Red/Yellow/Green/Normal
    risk_reasons: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# 스키마 검증 결과
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    table:           str
    total_rows:      int
    valid_rows:      int
    invalid_rows:    int
    missing_cols:    list[str] = field(default_factory=list)
    type_errors:     list[str] = field(default_factory=list)
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
    table:             str
    total_rows:        int
    null_rates:        dict[str, float] = field(default_factory=dict)
    duplicate_rate:    float = 0.0
    date_anomalies:    int = 0   # 날짜 형식 오류 건수
    wk_compn_unknown_rate: float = 0.0  # WK_COMPN_CD 매핑 불가 비율
    warnings:          list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return (
            self.duplicate_rate < 0.05
            and self.date_anomalies == 0
            and self.wk_compn_unknown_rate < 0.30
        )


# ─────────────────────────────────────────────────────────────────────────────
# 파이프라인 실행 결과
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """전체 ETL 파이프라인 실행 결과 요약."""
    partition:            str       # 처리 파티션 (YYYYMM)
    total_patients:       int = 0
    total_prescriptions:  int = 0
    total_drug_items:     int = 0
    overlap_pairs:        int = 0
    features_written:     int = 0
    red_count:            int = 0
    yellow_count:         int = 0
    green_count:          int = 0
    normal_count:         int = 0
    validation:           Optional[ValidationResult] = None
    quality:              Optional[QualityReport] = None
    elapsed_seconds:      float = 0.0
    errors:               list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return len(self.errors) == 0
