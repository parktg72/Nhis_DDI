"""
Pydantic 요청/응답 스키마

설계 원칙:
  - 입력: 환자의 현재 처방 약물 목록 (EDI 코드 기반)
  - 출력: 위험도 등급 + 이유 + ML 확률 + 탐지된 DDI 목록
  - 하이브리드: Rule 등급과 ML 등급을 모두 반환 (신뢰성·설명가능성)
"""
from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ─────────────────────────────────────────────────────────────────────────────
# 공통 Enum
# ─────────────────────────────────────────────────────────────────────────────

class RiskLevel(str, Enum):
    RED    = "Red"
    YELLOW = "Yellow"
    GREEN  = "Green"
    NORMAL = "Normal"

    @property
    def order(self) -> int:
        return {"Red": 3, "Yellow": 2, "Green": 1, "Normal": 0}[self.value]

    def __gt__(self, other: "RiskLevel") -> bool:
        return self.order > other.order

    @classmethod
    def max(cls, a: "RiskLevel", b: "RiskLevel") -> "RiskLevel":
        return a if a.order >= b.order else b


class Severity(str, Enum):
    CONTRAINDICATED = "Contraindicated"
    MAJOR           = "Major"
    MODERATE        = "Moderate"
    MINOR           = "Minor"
    UNKNOWN         = "Unknown"


# ─────────────────────────────────────────────────────────────────────────────
# 요청 스키마
# ─────────────────────────────────────────────────────────────────────────────

class DrugItem(BaseModel):
    """단일 처방 약물."""
    edi_code:    str   = Field(..., description="건보 EDI 약품코드")
    atc_code:    Optional[str] = Field(None, description="ATC 코드 (없으면 자동 조회)")
    drug_name:   Optional[str] = Field(None, description="약품명")
    total_days:  int   = Field(..., ge=1, le=365, description="총 투여일수")
    dose_once:   float = Field(1.0, gt=0, description="1회 투여량")
    dose_freq:   int   = Field(1, ge=1, le=10, description="1일 투여횟수")
    start_date:  Optional[date] = Field(None, description="투여 시작일 (없으면 오늘)")
    institution_id: Optional[str] = Field(None, description="요양기관 ID")

    @field_validator("edi_code")
    @classmethod
    def edi_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("EDI 코드는 빈 값 불가")
        return v.strip()


class PredictRequest(BaseModel):
    """위험도 예측 요청."""
    patient_id:   str          = Field(..., description="환자 식별자 (가명처리된 ID)")
    drugs:        list[DrugItem] = Field(..., min_length=1, description="처방 약물 목록")
    patient_age:  Optional[int]  = Field(None, ge=0, le=120, description="나이")
    patient_sex:  Optional[str]  = Field(None, pattern="^[MF]$", description="성별 (M/F)")
    reference_date: Optional[date] = Field(None, description="기준일 (기본: 오늘)")

    @field_validator("drugs")
    @classmethod
    def at_least_one_drug(cls, v: list[DrugItem]) -> list[DrugItem]:
        if len(v) == 0:
            raise ValueError("약물이 1개 이상 필요")
        return v

    @model_validator(mode="after")
    def set_default_dates(self) -> "PredictRequest":
        today = date.today()
        for drug in self.drugs:
            if drug.start_date is None:
                drug.start_date = today
        if self.reference_date is None:
            self.reference_date = today
        return self


# ─────────────────────────────────────────────────────────────────────────────
# 응답 스키마
# ─────────────────────────────────────────────────────────────────────────────

class DDIAlert(BaseModel):
    """탐지된 약물상호작용 단일 알림."""
    drug_a:      str
    drug_b:      str
    severity:    Severity
    description: Optional[str] = None
    source:      str = "Unknown"  # "DrugBank" | "HIRA_DUR" | "Rule"


class PredictResponse(BaseModel):
    """위험도 예측 응답."""
    patient_id:     str
    risk_level:     RiskLevel
    rule_level:     RiskLevel        = Field(description="Rule-based Safety Net 등급")
    ml_level:       Optional[RiskLevel] = Field(None, description="ML 모델 등급 (None=미사용)")
    ml_probability: Optional[float]    = Field(None, ge=0.0, le=1.0,
                                                description="ML Red 예측 확률")
    drug_count:     int
    ddi_alerts:     list[DDIAlert]   = Field(default_factory=list)
    risk_reasons:   list[str]        = Field(default_factory=list)
    intervention:   str              = Field(description="권장 개입 주기")
    reference_date: date


class BatchPredictRequest(BaseModel):
    """배치 예측 요청 (다수 환자)."""
    requests: list[PredictRequest] = Field(..., min_length=1, max_length=1000)


class BatchPredictResponse(BaseModel):
    """배치 예측 응답."""
    results:       list[PredictResponse]
    total:         int
    red_count:     int
    yellow_count:  int
    green_count:   int
    normal_count:  int
    elapsed_ms:    float


class HealthResponse(BaseModel):
    """헬스체크 응답."""
    status:       str    # "ok" | "degraded"
    model_loaded: bool
    rule_loaded:  bool
    version:      str
    uptime_sec:   float


class ModelInfoResponse(BaseModel):
    """모델 정보."""
    model_type:    str
    partition:     Optional[str]
    n_features:    Optional[int]
    threshold:     Optional[float]
    feature_names: Optional[list[str]] = None


# ─────────────────────────────────────────────────────────────────────────────
# 개입 주기 매핑
# ─────────────────────────────────────────────────────────────────────────────

INTERVENTION_MAP: dict[RiskLevel, str] = {
    RiskLevel.RED:    "즉각 개입 (당일 약사 면담 필요)",
    RiskLevel.YELLOW: "월 1회 복약 상담",
    RiskLevel.GREEN:  "분기 1회 복약 상담",
    RiskLevel.NORMAL: "정기 모니터링",
}
