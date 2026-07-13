"""위험 약물 판정 상수 — 단일 출처.

기준: config/drug_rules.yaml :123 의 high_risk_drugs.name_keywords (CLINICAL_STANDARDS_v1.0).

Codex 2026-05-06 ISSUE-3 (Qwen 후속): 동일 의도의 상수가 ETL / serving / rules
3곳에 따로 정의되어 있어 drift 위험. 본 모듈을 단일 출처로 두고 모든 경로에서
import 하도록 정렬.

ETL ↔ serving 은 직전까지도 정확히 동일했음 (frozenset wrapping 차이뿐) —
본 단일화는 mechanical 정리이며 동작 변경 없음. rules/safety_net.py 의 9개
list 는 본 yaml 정의(15개) 와 어긋난 결함이지만 그 정정은 임상 영향이 있어
별도 단계로 분리.
"""
from __future__ import annotations

# ── 고위험 약물 (CLINICAL_STANDARDS_v1.0 — drug_rules.yaml :123) ──────────────
HIGH_RISK_KEYWORDS: frozenset[str] = frozenset({
    "warfarin", "methotrexate", "lithium", "digoxin", "amiodarone",
    "phenytoin", "cyclosporine", "tacrolimus", "sirolimus", "theophylline",
    "insulin", "clozapine", "carbamazepine", "valproate", "phenobarbital",
})
HIGH_RISK_ATC_PREFIXES: tuple[str, ...] = (
    "B01AA03", "L01BA01", "N05AN01", "C01AA05", "C01BD01",
    "N03AB02", "L04AD01", "L04AD02", "L04AA18", "R03DA04",
)

# ── 신기능저하 위험 약물 (NSAIDs / aminoglycosides / calcineurin inhibitors 등) ──
RENAL_RISK_KEYWORDS: frozenset[str] = frozenset({
    "ibuprofen", "naproxen", "diclofenac", "celecoxib", "ketorolac",
    "indomethacin", "meloxicam", "piroxicam",       # NSAIDs
    "gentamicin", "tobramycin", "amikacin", "vancomycin",  # 신독성 항생제
    "lithium", "cisplatin", "acyclovir", "tenofovir",
    "cyclosporine", "tacrolimus",                    # calcineurin inhibitors
})
RENAL_RISK_ATC_PREFIXES: tuple[str, ...] = ("M01A", "N05AN01", "J01GB", "L04AD")

# ── 간기능저하 위험 약물 ──────────────────────────────────────────────────────
HEPATIC_RISK_KEYWORDS: frozenset[str] = frozenset({
    "methotrexate", "valproate", "valproic", "isoniazid", "amiodarone",
    "phenytoin", "carbamazepine", "ketoconazole", "itraconazole",
    "acetaminophen", "paracetamol",
})
HEPATIC_RISK_ATC_PREFIXES: tuple[str, ...] = (
    "L01BA01", "N03AG01", "J04AC01", "C01BD01", "N03AB02",
)
