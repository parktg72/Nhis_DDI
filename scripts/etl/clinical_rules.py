"""임상 위험도 판정 규칙 (CLINICAL_STANDARDS_v1.0).

Red/Yellow trigger 집합 수집 공용 모듈. ETL 라벨 생성, 학습 라벨 필터,
서빙 추론 설명 3곳에서 import 되어 규칙 드리프트를 방지한다.

trigger 는 문자열 집합으로 반환한다. 판정 순서는 여기서 정하지 않는다
(호출자 책임). 규칙 변경 시 CLINICAL_STANDARDS_VERSION 을 올리고
학습 메타에 기록할 것.

Red vs Yellow 트리거 이름 비대칭 — 의도된 설계:
  Red 트리거 ("RED_*"): risk_reasons 로 임상팀에 노출되는 사유 코드.
  Yellow 트리거 ("DDI_MAJOR"/"DDI_MOD"/"DUP"/"FRAG"): 내부 토큰.
    _assign_yellow_subtype 이 set 크기·비교로 Y_TRIPLE/Y_DOUBLE/Y_DDI_MAJOR/... 판정에 사용.
    (출력 라벨은 Y_ prefix 를 별도로 가짐, 네임스페이스 분리.)
"""
from __future__ import annotations

from typing import Any

CLINICAL_STANDARDS_VERSION = "v1.0"


def collect_red_triggers(f: Any) -> set[str]:
    """Red 조건 = 절대 **금기(contraindicated DDI)만**. 즉시개입 최상위.

    2026-06-06 재설계: major3/triple_whammy/10drug+고위험/고령+장기 는 활성화 시 Red 율이
    25.8% 로 과도 → Y_TRIPLE(즉시개입이되 Red 보다 낮은 수준)로 강등.
    collect_severe_immediate_triggers 참조. f: PatientFeatures 또는 동일 attribute 객체.
    """
    triggers: set[str] = set()
    if f.ddi_contraindicated >= 1:
        triggers.add("RED_CONTRAINDICATED")
    return triggers


def collect_severe_immediate_triggers(f: Any) -> set[str]:
    """Y_TRIPLE(즉시개입, Red 바로 아래 수준) 강제 조건 — 구 Red 트리거 중 금기 외.

    즉시개입(Red 와 동일 긴급도)이 필요하나 절대 금기는 아니므로 risk_level=Yellow·
    yellow_subtype=Y_TRIPLE(action=즉시개입). 비어 있으면 일반 Yellow 차원 로직 적용.
    """
    triggers: set[str] = set()
    if f.ddi_major >= 3:
        triggers.add("SEV_MAJOR_3PLUS")
    if f.triple_whammy:
        triggers.add("SEV_TRIPLE_WHAMMY")
    if f.drug_count >= 10 and f.has_high_risk_drug:
        triggers.add("SEV_10DRUG_HIGHRISK")
    if (
        f.age is not None
        and f.age >= 75
        and f.drug_count >= 5
        and (f.has_renal_risk_drug or f.has_hepatic_risk_drug)
    ):
        triggers.add("SEV_ELDERLY_ORGAN")
    return triggers


def collect_yellow_triggers(f: Any) -> set[str]:
    """Yellow 조건 집합. 호출 전 Red trigger 가 없는지 확인은 호출자 책임.

    계수 라벨 판정(Y_DOUBLE=|triggers|==2, Y_TRIPLE=|triggers|>=3)은 외부에서 결정한다.
    """
    triggers: set[str] = set()
    if f.ddi_major >= 1:
        triggers.add("DDI_MAJOR")
    if f.ddi_moderate >= 2:
        triggers.add("DDI_MOD")
    if f.dup_same_ingredient >= 1:
        triggers.add("DUP")
    if f.institution_count >= 3:
        triggers.add("FRAG")
    return triggers
