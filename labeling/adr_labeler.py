"""
ADR 프록시 레이블러 (후향적 레이블 생성)

PROJECT_PLAN 4.4 레이블링 전략 2차:
  유해사례(ADR) 발생 여부를 후향적 레이블로 활용.
  처방 후 90일 내 관련 상병코드(ICD-10) 발생 시 ADR 발생으로 판정.

ADR 프록시 5종 (PROJECT_PLAN 4.4):
  1. 출혈 위험     : 와파린/DOAC + NSAIDs → K92, D68, I60-I62
  2. 급성신부전    : Triple Whammy       → N17, E87
  3. 디곡신 독성   : 디곡신 + 아미오다론/베라파밀 → I49, R11
  4. 세로토닌 증후군: SSRI + MAOi/Triptan → G25, R56
  5. 저혈당        : 인슐린/설포닐우레아 과잉 → E16, R55

레이블 신뢰도 등급:
  HIGH   : ADR 코드 확인 + DDI 처방 패턴 일치
  MEDIUM : ADR 코드 확인 (처방 패턴 미확인)
  LOW    : DDI 패턴만 있고 ADR 코드 없음 (Rule 레이블)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Set

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# ADR 프록시 정의
# ─────────────────────────────────────────────────────────────────────────────

# ATC prefix → DDI 그룹 매핑
_ANTICOAG_ATC    = {"B01AA", "B01AE", "B01AF"}   # 와파린, DOAC
_NSAID_ATC       = {"M01A"}
_ACEI_ARB_ATC    = {"C09AA", "C09CA"}
_KSPARING_ATC    = {"C03DA", "C03DB"}
_DIGOXIN_ATC     = {"C01AA"}
_AMIO_VERA_ATC   = {"C01BD", "C08DA"}
_SSRI_ATC        = {"N06AB"}
_MAOI_ATC        = {"N06AF", "N06AG"}
_TRIPTAN_ATC     = {"N02CC"}
_INSULIN_ATC     = {"A10A"}
_SU_ATC          = {"A10BB", "A10BC"}             # 설포닐우레아


class ADRType:
    BLEEDING          = "bleeding"
    ACUTE_KIDNEY      = "acute_kidney_injury"
    DIGOXIN_TOXICITY  = "digoxin_toxicity"
    SEROTONIN         = "serotonin_syndrome"
    HYPOGLYCEMIA      = "hypoglycemia"


# ICD-10 코드 패턴 → ADR 유형
ADR_ICD10_MAP: Dict[str, list[str]] = {
    ADRType.BLEEDING:         ["K92", "D68", "I60", "I61", "I62"],
    ADRType.ACUTE_KIDNEY:     ["N17", "E87"],
    ADRType.DIGOXIN_TOXICITY: ["I49", "R11"],
    ADRType.SEROTONIN:        ["G25", "R56"],
    ADRType.HYPOGLYCEMIA:     ["E16", "R55"],
}

# ADR 유형 → 위험도 가중치 (Red 레이블 강도)
ADR_SEVERITY_WEIGHT: Dict[str, float] = {
    ADRType.BLEEDING:         1.0,
    ADRType.ACUTE_KIDNEY:     1.0,
    ADRType.DIGOXIN_TOXICITY: 0.9,
    ADRType.SEROTONIN:        1.0,
    ADRType.HYPOGLYCEMIA:     0.8,
}

CONFIDENCE_HIGH   = "HIGH"
CONFIDENCE_MEDIUM = "MEDIUM"
CONFIDENCE_LOW    = "LOW"


# ─────────────────────────────────────────────────────────────────────────────
# 데이터 클래스
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ADREvidence:
    """단일 ADR 증거."""
    adr_type: str
    matched_icd10: list[str]        # 매칭된 ICD-10 코드
    matched_atc: list[str]          # 관련 ATC 코드
    days_after_prescription: int    # 처방 후 ADR 코드까지 경과일
    severity_weight: float


@dataclass
class LabelResult:
    """환자 레이블 결과."""
    patient_id: str
    label: int                      # 1=고위험(ADR 발생), 0=정상
    confidence: str                 # HIGH | MEDIUM | LOW
    adr_evidences: list[ADREvidence] = field(default_factory=list)
    rule_label: Optional[str] = None   # Rule-based 위험도 (Red/Yellow/Green/Normal)
    final_label: Optional[str] = None  # 최종 통합 위험도
    note: str = ""

    @property
    def adr_score(self) -> float:
        """ADR 증거 강도 합산 점수."""
        return sum(e.severity_weight for e in self.adr_evidences)

    def to_dict(self) -> dict:
        return {
            "patient_id":   self.patient_id,
            "label":        self.label,
            "confidence":   self.confidence,
            "adr_score":    round(self.adr_score, 3),
            "adr_types":    [e.adr_type for e in self.adr_evidences],
            "rule_label":   self.rule_label,
            "final_label":  self.final_label,
            "note":         self.note,
        }


# ─────────────────────────────────────────────────────────────────────────────
# ADRLabeler
# ─────────────────────────────────────────────────────────────────────────────

def _atc_matches(atc_code: str, prefixes: Set[str]) -> bool:
    """ATC 코드가 지정 prefix 집합에 속하는지 확인."""
    return any(atc_code.upper().startswith(p) for p in prefixes)


def _icd10_matches(diag_code: str, patterns: list[str]) -> bool:
    """ICD-10 코드가 패턴 리스트에 매칭되는지 확인."""
    code = diag_code.upper().replace(".", "")
    return any(code.startswith(p.replace(".", "")) for p in patterns)


class ADRLabeler:
    """ICD-10 상병코드 기반 ADR 후향적 레이블 생성기.

    Usage:
        labeler = ADRLabeler(lookback_days=90)
        result = labeler.label(
            patient_id="P001",
            atc_codes=["B01AA03", "M01AE01"],
            diagnosis_codes=[("K92.1", 45)],  # (ICD-10 코드, 처방후 경과일)
            rule_risk_level="Red",
        )
    """

    def __init__(self, lookback_days: int = 90):
        self._lookback = lookback_days

    def label(
        self,
        patient_id: str,
        atc_codes: list[str],
        diagnosis_codes: list[tuple[str, int]],  # (icd10, days_after_rx)
        rule_risk_level: Optional[str] = None,
    ) -> LabelResult:
        """단일 환자 ADR 레이블 생성."""
        evidences: list[ADREvidence] = []

        # lookback 기간 내 진단코드만 필터
        recent_diag = [(code, days) for code, days in diagnosis_codes if 0 <= days <= self._lookback]

        # ADR 유형별 검사
        for adr_type, icd_patterns in ADR_ICD10_MAP.items():
            matched_icd = [
                code for code, _ in recent_diag
                if _icd10_matches(code, icd_patterns)
            ]
            if not matched_icd:
                continue

            matched_atc = self._get_related_atc(adr_type, atc_codes)
            days = min(
                (days for code, days in recent_diag if _icd10_matches(code, icd_patterns)),
                default=self._lookback,
            )
            evidences.append(ADREvidence(
                adr_type=adr_type,
                matched_icd10=matched_icd,
                matched_atc=matched_atc,
                days_after_prescription=days,
                severity_weight=ADR_SEVERITY_WEIGHT[adr_type],
            ))

        label = 1 if evidences else 0
        confidence = self._assess_confidence(evidences, atc_codes)
        final_label = self._merge_with_rule(label, rule_risk_level)

        return LabelResult(
            patient_id=patient_id,
            label=label,
            confidence=confidence,
            adr_evidences=evidences,
            rule_label=rule_risk_level,
            final_label=final_label,
        )

    def label_batch(self, df) -> list[LabelResult]:
        """DataFrame 기반 배치 레이블링.

        df 필수 컬럼:
          patient_id, atc_codes (list), diagnosis_codes (list of (icd10, days)),
          risk_level (optional)
        """
        results = []
        for _, row in df.iterrows():
            atc_codes  = row.get("atc_codes", []) or []
            diag_codes = row.get("diagnosis_codes", []) or []
            rule_level = row.get("risk_level", None)

            result = self.label(
                patient_id=str(row["patient_id"]),
                atc_codes=list(atc_codes),
                diagnosis_codes=list(diag_codes),
                rule_risk_level=rule_level,
            )
            results.append(result)

        pos = sum(r.label for r in results)
        logger.info("ADR 레이블링 완료: %d건 중 %d건 ADR 발생 (%.1f%%)",
                    len(results), pos, pos / max(len(results), 1) * 100)
        return results

    def _get_related_atc(self, adr_type: str, atc_codes: list[str]) -> list[str]:
        """ADR 유형에 관련된 ATC 코드 반환."""
        mapping = {
            ADRType.BLEEDING:         _ANTICOAG_ATC | _NSAID_ATC,
            ADRType.ACUTE_KIDNEY:     _ACEI_ARB_ATC | _KSPARING_ATC | _NSAID_ATC,
            ADRType.DIGOXIN_TOXICITY: _DIGOXIN_ATC | _AMIO_VERA_ATC,
            ADRType.SEROTONIN:        _SSRI_ATC | _MAOI_ATC | _TRIPTAN_ATC,
            ADRType.HYPOGLYCEMIA:     _INSULIN_ATC | _SU_ATC,
        }
        relevant = mapping.get(adr_type, set())
        return [a for a in atc_codes if _atc_matches(a, relevant)]

    def _assess_confidence(self, evidences: list[ADREvidence], atc_codes: list[str]) -> str:
        """신뢰도 등급 산정."""
        if not evidences:
            return CONFIDENCE_LOW
        # 모든 증거에 관련 ATC 코드가 있으면 HIGH
        all_have_atc = all(len(e.matched_atc) > 0 for e in evidences)
        return CONFIDENCE_HIGH if all_have_atc else CONFIDENCE_MEDIUM

    def _merge_with_rule(self, adr_label: int, rule_level: Optional[str]) -> str:
        """ADR 레이블과 Rule 레이블 통합 → 최종 위험도."""
        if adr_label == 1:
            return "Red"
        if rule_level in ("Red", "Yellow", "Green"):
            return rule_level
        return "Normal"
