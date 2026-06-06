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


class DLPredictionResult(BaseModel):
    """운영 DL 추론 결과.

    현재는 Rule/ML 최종등급을 바꾸지 않는 보조 결과로만 반환한다.
    """
    run_id:            Optional[str] = None
    encoding_strategy: str
    predicted_label:   str
    score:             float = Field(ge=0.0, le=1.0)
    probabilities:     dict[str, float]
    known_drug_count:  int = Field(ge=0)
    unknown_drug_count: int = Field(ge=0)


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

    # 계층 분류 (Stage 1 Red + Stage 2 Yellow-subtype) 확장 필드
    yellow_subtype: Optional[str]         = Field(
        None,
        description="Yellow 세부 라벨 (Y_TRIPLE/Y_DOUBLE/Y_DDI_MAJOR/Y_DDI_MOD/Y_DUP/Y_FRAG/Y_OTHER) — 계층 모드에서만 채워짐",
    )
    stage2_probs:   Optional[dict[str, float]] = Field(
        None,
        description="Stage 2 다중 클래스 확률 분포 — 계층 모드, Red 비확정일 때만",
    )
    red_suspect:    bool = Field(
        False,
        description="Stage 1 'τ_review ≤ p_red < τ_red' 구간 — 운영팀 검수 큐 표시",
    )
    action:         Optional[str] = Field(
        None,
        description="세부 라벨별 개입 액션 (Y_TRIPLE/Y_DDI_MAJOR=의료인 전화, Y_DOUBLE/Y_DDI_MOD/Y_FRAG=문자 알림, Y_DUP=문서+문자 등)",
    )
    dl_prediction:  Optional[DLPredictionResult] = Field(
        None,
        description="운영 DL 보조 추론 결과. 현재 최종 risk_level 결정에는 반영하지 않음",
    )
    dl_error:       Optional[str] = Field(
        None,
        description="DL 보조 추론 실패 사유. Rule/ML 응답은 계속 반환",
    )


class BatchPredictRequest(BaseModel):
    """배치 예측 요청 (다수 환자)."""
    requests: list[PredictRequest] = Field(..., min_length=1, max_length=1000)


class BatchPredictResponse(BaseModel):
    """배치 예측 응답.

    Codex 2026-05-07 #6 — 직전 `total` 이 success 건수임에도 클라이언트가 입력
    건수로 오해 가능. 명시적 카운트 분리:
      - requested_count: 입력된 요청 환자 수 (= len(req.requests))
      - success_count:   예측 성공 환자 수 (= len(results))
      - failed_count:    예측 실패 환자 수 (= len(warnings) = requested - success)
      - total:           DEPRECATED — success_count 와 동일 값 (backward compat)
    """
    results:         list[PredictResponse]
    requested_count: int = Field(description="요청된 환자 수 (입력 건수)")
    success_count:   int = Field(description="예측 성공 환자 수")
    failed_count:    int = Field(description="예측 실패 환자 수 (= requested_count - success_count)")
    total:           int = Field(
        description="DEPRECATED: success_count alias (backward compat). 신규 클라이언트는 success_count 사용"
    )
    red_count:       int
    yellow_count:    int
    green_count:     int
    normal_count:    int
    elapsed_ms:      float
    warnings:        list[str] = Field(default_factory=list, description="부분 실패 경고 목록")


class HealthResponse(BaseModel):
    """헬스체크 응답."""
    status:       str    # "ok" | "degraded"
    model_loaded: bool   # 단일 ML 또는 계층 모델 둘 중 하나라도 로드되면 True
    rule_loaded:  bool
    version:      str
    uptime_sec:   float
    model_mode:           Optional[str]  = Field(
        None,
        description='로드된 모델 모드: "single" | "hierarchical" | "both" | "none"',
    )
    hierarchical_loaded:  Optional[bool] = Field(
        None,
        description="계층 분류기(Stage1/Stage2) 로드 여부",
    )
    # Codex 2026-05-07 #1 — feature schema drift 운영 visibility
    schema_drift: list[str] = Field(
        default_factory=list,
        description=(
            "lenient 모드로 로드된 모델의 _BUILDER_KNOWN_COLS 외 컬럼 목록. "
            "non-empty 면 status='degraded' 자동 전환."
        ),
    )
    feature_schema_lenient: bool = Field(
        False,
        description="FEATURE_SCHEMA_LENIENT 환경 변수 활성 상태 (env trail, sunset 무관)",
    )
    # Codex 2026-05-07 #6-followup — env 켜졌지만 sunset 으로 실제 차단된 상태 명확화
    feature_schema_lenient_allowed: bool = Field(
        False,
        description=(
            "lenient 가 실제로 효력 발휘 가능한지 (env 활성 AND sunset 안). "
            "feature_schema_lenient=True 인데 본 값이 False 면 'env 는 켜졌지만 "
            "sunset deadline 으로 차단됨' 의미."
        ),
    )
    feature_schema_lenient_sunset_date: Optional[str] = Field(
        None,
        description=(
            "lenient escape hatch sunset deadline (ISO YYYY-MM-DD). today >= 본 "
            "값이면 차단. env FEATURE_SCHEMA_LENIENT_SUNSET_DATE 또는 코드 default."
        ),
    )
    degraded_reasons: list[str] = Field(
        default_factory=list,
        description="degraded 사유 목록 (예: 'feature_schema_drift: 2 unknown columns')",
    )
    dl_loaded: Optional[bool] = Field(
        None,
        description="운영 DL bundle manifest/hash/lookback 검증 후 로드 여부",
    )
    dl_lookback_days: Optional[int] = Field(
        None,
        description="로드된 DL bundle 의 lookback_days",
    )
    dl_bundle_run_id: Optional[str] = Field(
        None,
        description="로드된 DL bundle MANIFEST.json run_id",
    )
    dl_schema_version: Optional[str] = Field(
        None,
        description="로드된 DL bundle schema_version",
    )


class ModelInfoResponse(BaseModel):
    """모델 정보."""
    model_type:    str
    partition:     Optional[str]
    n_features:    Optional[int]
    threshold:     Optional[float]
    feature_names: Optional[list[str]] = None
    # Codex 2026-05-07 #1 — model artifact 의 schema drift trail (디버깅/감사용).
    # /health 가 운영 알림용이라면 /model/info 는 staff 가 깊이 들여다볼 때 사용.
    schema_drift:  list[str] = Field(
        default_factory=list,
        description="lenient 모드로 로드된 경우 _BUILDER_KNOWN_COLS 외 컬럼 목록",
    )
    dl_loaded: Optional[bool] = None
    dl_lookback_days: Optional[int] = None
    dl_bundle_run_id: Optional[str] = None
    dl_schema_version: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# 개입 주기 매핑
# ─────────────────────────────────────────────────────────────────────────────

# 2026-06-07 개입 위계: Red=즉각 개입. Yellow 는 subtype 별 action(약사전화/문자안내/모니터링)이
# 실질 — intervention 은 레벨 일반값. Green·Normal = 관여 안 함.
INTERVENTION_MAP: dict[RiskLevel, str] = {
    RiskLevel.RED:    "즉각 개입",
    RiskLevel.YELLOW: "복약 상담",
    RiskLevel.GREEN:  "관여 안 함",
    RiskLevel.NORMAL: "관여 안 함",
}
