#!/usr/bin/env python3
"""
Rule-based Safety Net

환자의 처방 약물 목록을 입력받아 DDI 위험도 등급을 평가.
최종 등급 = max(Rule 등급, ML 등급) 구조에서 Rule 부분 담당.

핵심 보장:
  - Top 10 DDI 100% 탐지율
  - Contraindicated DDI Zero Miss
  - Triple Whammy 탐지
  - Critical Error 0건

사용 예시:
  from rules.safety_net import SafetyNet

  sn = SafetyNet()
  result = sn.assess(drugs=["warfarin", "ibuprofen", "aspirin"])
  print(result.risk_grade)       # "Red"
  print(result.triggered_rules)  # ["TOP01: Anticoagulant_NSAIDs"]
  print(result.explanation)      # "와파린/DOAC + NSAIDs → 출혈 위험..."
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml


SEVERITY_RANK = {"Contraindicated": 0, "Major": 1, "Moderate": 2, "Minor": 3, "Unknown": 4}
RISK_GRADE_RANK = {"Red": 0, "Yellow": 1, "Green": 2, "Normal": 3}

DEFAULT_RULES_PATH = Path(__file__).parent.parent / "config" / "drug_rules.yaml"
DEFAULT_DDI_MATRIX_PATH = Path(__file__).parent.parent / "data" / "processed" / "ddi_matrix_final.parquet"
DEFAULT_DRUG_INDEX_PATH = Path(__file__).parent.parent / "data" / "processed" / "drug_name_index.parquet"


@dataclass
class DDIPair:
    drug_a: str
    drug_b: str
    severity: str
    description: str
    source: str
    rule_id: Optional[str] = None


@dataclass
class RiskAssessment:
    """SafetyNet 평가 결과."""
    risk_grade: str                            # Red / Yellow / Green / Normal
    ddi_contraindicated_count: int = 0
    ddi_major_count: int = 0
    ddi_moderate_count: int = 0
    ddi_minor_count: int = 0
    triple_whammy_flag: bool = False
    qt_drug_count: int = 0
    triggered_rules: list[str] = field(default_factory=list)   # ["TOP01: Anticoagulant_NSAIDs", ...]
    ddi_pairs: list[DDIPair] = field(default_factory=list)
    explanation: str = ""
    input_drugs: list[str] = field(default_factory=list)
    matched_drugs: list[str] = field(default_factory=list)      # 실제 매핑된 약물

    @property
    def has_critical_risk(self) -> bool:
        return self.risk_grade == "Red"

    @property
    def summary(self) -> str:
        lines = [
            f"위험도: {self.risk_grade}",
            f"DDI: Contraindicated={self.ddi_contraindicated_count}, Major={self.ddi_major_count}, "
            f"Moderate={self.ddi_moderate_count}, Minor={self.ddi_minor_count}",
            f"Triple Whammy: {'예' if self.triple_whammy_flag else '아니오'}",
            f"QT 연장 약물 수: {self.qt_drug_count}",
            f"발동 규칙: {', '.join(self.triggered_rules) if self.triggered_rules else '없음'}",
        ]
        if self.explanation:
            lines.append(f"설명: {self.explanation}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "risk_grade": self.risk_grade,
            "ddi_contraindicated_count": self.ddi_contraindicated_count,
            "ddi_major_count": self.ddi_major_count,
            "ddi_moderate_count": self.ddi_moderate_count,
            "ddi_minor_count": self.ddi_minor_count,
            "triple_whammy_flag": self.triple_whammy_flag,
            "qt_drug_count": self.qt_drug_count,
            "triggered_rules": self.triggered_rules,
            "top_risk_factors": [
                {"rule": p.rule_id or "DDI", "drug_a": p.drug_a, "drug_b": p.drug_b, "severity": p.severity}
                for p in sorted(self.ddi_pairs, key=lambda x: SEVERITY_RANK.get(x.severity, 4))[:5]
            ],
        }


class DrugMatcher:
    """약물명 매칭 엔진 (대소문자 무관, 부분 매칭 지원)."""

    def __init__(self, drug_index: pd.DataFrame, drug_groups: dict):
        self._index = drug_index   # drug_name_lower → drugbank_id, atc_codes
        self._groups = drug_groups  # group_name → {name_keywords, atc_prefixes}
        self._name_map = {}        # normalized_name → drugbank_id
        self._atc_map = {}         # atc_code → [drugbank_id, ...]

        if not drug_index.empty:
            for _, row in drug_index.iterrows():
                name_lower = str(row.get("drug_name_lower", "")).strip()
                did = str(row.get("drugbank_id", "")).strip()
                atc = str(row.get("atc_codes", "")).strip()
                if name_lower:
                    self._name_map[name_lower] = did
                if atc:
                    for code in atc.split("|"):
                        code = code.strip()
                        if code:
                            self._atc_map.setdefault(code, []).append(did)

    def normalize(self, name: str) -> str:
        return name.lower().strip()

    def match_drug_to_group(self, drug_name: str, group_def: dict) -> bool:
        """약물명이 그룹 정의에 매칭되는지 확인."""
        name_lower = self.normalize(drug_name)

        # 이름 키워드 매칭
        for kw in group_def.get("name_keywords", []):
            if kw.lower() in name_lower:
                return True

        # ATC 코드 매칭 (drug_index 에서 ATC 조회 후 prefix 비교)
        did = self._name_map.get(name_lower, "")
        if did:
            atc_str = ""
            # 인덱스에서 ATC 코드 조회
            for _, row in self._index.iterrows():
                if row.get("drugbank_id") == did:
                    atc_str = str(row.get("atc_codes", ""))
                    break
            for atc_code in atc_str.split("|"):
                atc_code = atc_code.strip()
                for prefix in group_def.get("atc_prefixes", []):
                    if atc_code.startswith(str(prefix)):
                        return True

        return False

    def drugs_in_group(self, drug_list: list[str], group_name: str) -> list[str]:
        """약물 목록에서 그룹에 속하는 약물 반환."""
        group_def = self._groups.get(group_name, {})
        # 다중 그룹 지원 (group_name 이 리스트인 경우)
        if isinstance(group_name, list):
            result = []
            for gn in group_name:
                result.extend(self.drugs_in_group(drug_list, gn))
            return list(set(result))
        return [d for d in drug_list if self.match_drug_to_group(d, group_def)]

    def count_in_group(self, drug_list: list[str], group_name) -> int:
        return len(self.drugs_in_group(drug_list, group_name))


class SafetyNet:
    """
    Rule-based Safety Net 엔진.

    Phase 1 핵심 컴포넌트: Top 10 DDI + Contraindicated DDI + Triple Whammy 보장.
    """

    def __init__(
        self,
        rules_path: Path = DEFAULT_RULES_PATH,
        ddi_matrix_path: Path = DEFAULT_DDI_MATRIX_PATH,
        drug_index_path: Path = DEFAULT_DRUG_INDEX_PATH,
    ):
        self._rules = self._load_rules(rules_path)
        self._ddi_matrix = self._load_ddi_matrix(ddi_matrix_path)
        self._drug_index = self._load_drug_index(drug_index_path)
        self._drug_groups = self._rules.get("drug_groups", {})
        self._matcher = DrugMatcher(self._drug_index, self._drug_groups)
        self._top10_rules = self._rules.get("top10_ddi_rules", [])
        self._risk_rules = self._rules.get("risk_grade_rules", {})

    # ─── 로더 ──────────────────────────────────────────────────────────────

    @staticmethod
    def _load_rules(path: Path) -> dict:
        if not path.exists():
            raise FileNotFoundError(f"drug_rules.yaml 없음: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @staticmethod
    def _load_ddi_matrix(path: Path) -> pd.DataFrame:
        if path.exists():
            df = pd.read_parquet(path)
            # 이름 소문자 인덱스 컬럼 추가
            df["_a_lower"] = df["drug_a_name"].str.lower().str.strip()
            df["_b_lower"] = df["drug_b_name"].str.lower().str.strip()
            return df
        print(f"[경고] DDI 매트릭스 없음: {path}. Top 10 규칙만 사용합니다.")
        return pd.DataFrame()

    @staticmethod
    def _load_drug_index(path: Path) -> pd.DataFrame:
        if path.exists():
            return pd.read_parquet(path)
        return pd.DataFrame()

    # ─── 핵심 평가 ─────────────────────────────────────────────────────────

    def assess(
        self,
        drugs: list[str],
        patient_age: Optional[int] = None,
        concurrent_drug_count: Optional[int] = None,
        has_renal_risk: bool = False,
        has_hepatic_risk: bool = False,
    ) -> RiskAssessment:
        """
        환자 처방 약물 목록으로 위험도 평가.

        Args:
            drugs: 약물명 목록 (대소문자 무관)
            patient_age: 환자 나이 (75세 이상 고위험 기준 적용)
            concurrent_drug_count: 90일 내 동시 복용 약물 수 (미입력 시 len(drugs))
            has_renal_risk: 신기능저하 약물 포함 여부
            has_hepatic_risk: 간기능저하 약물 포함 여부

        Returns:
            RiskAssessment
        """
        if not drugs:
            return RiskAssessment(risk_grade="Normal", input_drugs=[])

        drug_count = concurrent_drug_count or len(drugs)
        result = RiskAssessment(risk_grade="Normal", input_drugs=list(drugs))

        # 1. Top 10 DDI 규칙 적용 (100% 탐지 보장)
        self._apply_top10_rules(drugs, result)

        # 2. DDI 매트릭스 기반 전체 DDI 탐지
        self._apply_matrix_ddi(drugs, result)

        # 3. Triple Whammy 탐지
        self._check_triple_whammy(drugs, result)

        # 4. QT 연장 다중 병용
        self._check_qt_prolongation(drugs, result)

        # 5. 위험도 등급 산출
        self._determine_risk_grade(
            result, drug_count, patient_age, has_renal_risk, has_hepatic_risk
        )

        # 6. 설명 생성
        result.explanation = self._build_explanation(result)

        return result

    # ─── Top 10 DDI 규칙 ──────────────────────────────────────────────────

    def _apply_top10_rules(self, drugs: list[str], result: RiskAssessment):
        for rule in self._top10_rules:
            rule_id = rule["id"]

            # QT 연장 다중 병용 (requires_count)
            if rule.get("requires_count"):
                qt_drugs = self._matcher.drugs_in_group(drugs, rule["drug_group_a"])
                if len(qt_drugs) >= rule["requires_count"]:
                    pair = DDIPair(
                        drug_a=qt_drugs[0],
                        drug_b=f"(총 {len(qt_drugs)}종)",
                        severity=rule["severity"],
                        description=rule["description"],
                        source="TOP10_RULE",
                        rule_id=rule_id,
                    )
                    self._add_ddi(result, pair)
                    result.triggered_rules.append(f"{rule_id}: {rule['name']}")
                continue

            # Triple Whammy (requires_triple)
            if rule.get("requires_triple"):
                self._check_triple_whammy(drugs, result, rule)
                continue

            # 일반 A + B 규칙
            group_a = rule.get("drug_group_a", "")
            group_b = rule.get("drug_group_b", "")

            drugs_a = self._matcher.drugs_in_group(drugs, group_a) if isinstance(group_a, str) else \
                      [d for g in group_a for d in self._matcher.drugs_in_group(drugs, g)]
            drugs_b = self._matcher.drugs_in_group(drugs, group_b) if isinstance(group_b, str) else \
                      [d for g in group_b for d in self._matcher.drugs_in_group(drugs, g)]

            if drugs_a and drugs_b:
                pair = DDIPair(
                    drug_a=drugs_a[0],
                    drug_b=drugs_b[0],
                    severity=rule["severity"],
                    description=rule["description"],
                    source="TOP10_RULE",
                    rule_id=rule_id,
                )
                self._add_ddi(result, pair)
                result.triggered_rules.append(f"{rule_id}: {rule['name']}")

    # ─── 매트릭스 기반 DDI 탐지 ───────────────────────────────────────────

    def _apply_matrix_ddi(self, drugs: list[str], result: RiskAssessment):
        """DDI 매트릭스에서 약물 쌍 검색."""
        if self._ddi_matrix.empty:
            return

        drug_lower = [d.lower().strip() for d in drugs]

        for name_a in drug_lower:
            # 이름 기반 매칭 (완전 일치 + 포함 매칭)
            matches_a = self._ddi_matrix[
                self._ddi_matrix["_a_lower"].str.contains(name_a, regex=False, na=False) |
                self._ddi_matrix["_b_lower"].str.contains(name_a, regex=False, na=False)
            ]

            for _, row in matches_a.iterrows():
                # 파트너 약물 확인
                if name_a in str(row["_a_lower"]):
                    partner_lower = str(row["_b_lower"])
                    partner_orig = str(row["drug_b_name"])
                    drug_a_name = str(row["drug_a_name"])
                else:
                    partner_lower = str(row["_a_lower"])
                    partner_orig = str(row["drug_a_name"])
                    drug_a_name = str(row["drug_b_name"])

                # 파트너가 처방 목록에 있는지 확인
                for name_b in drug_lower:
                    if name_b != name_a and name_b in partner_lower:
                        pair = DDIPair(
                            drug_a=drug_a_name,
                            drug_b=partner_orig,
                            severity=str(row.get("severity", "Unknown")),
                            description=str(row.get("description", ""))[:200],
                            source=str(row.get("source", "DDI_Matrix")),
                            rule_id=None,
                        )
                        self._add_ddi(result, pair)
                        break

    # ─── Triple Whammy ────────────────────────────────────────────────────

    def _check_triple_whammy(
        self,
        drugs: list[str],
        result: RiskAssessment,
        rule: Optional[dict] = None,
    ):
        """Triple Whammy: ACEi/ARB + K보존이뇨제 + NSAIDs."""
        has_rasi = (
            self._matcher.count_in_group(drugs, "acei") > 0 or
            self._matcher.count_in_group(drugs, "arb") > 0
        )
        has_k_diuretic = self._matcher.count_in_group(drugs, "k_sparing_diuretics") > 0
        has_nsaid = self._matcher.count_in_group(drugs, "nsaids") > 0

        if has_rasi and has_k_diuretic and has_nsaid:
            result.triple_whammy_flag = True
            rasi_drugs = self._matcher.drugs_in_group(drugs, "acei") + self._matcher.drugs_in_group(drugs, "arb")
            k_drugs = self._matcher.drugs_in_group(drugs, "k_sparing_diuretics")
            nsaid_drugs = self._matcher.drugs_in_group(drugs, "nsaids")

            pair = DDIPair(
                drug_a=f"{rasi_drugs[0]}+{k_drugs[0]}",
                drug_b=nsaid_drugs[0],
                severity="Major",
                description="Triple Whammy: ACEi/ARB + K보존이뇨제 + NSAIDs → 급성신부전 위험",
                source="TRIPLE_WHAMMY_RULE",
                rule_id="TOP03",
            )
            self._add_ddi(result, pair)
            if "TOP03: Triple_Whammy" not in result.triggered_rules:
                result.triggered_rules.append("TOP03: Triple_Whammy")

    # ─── QT 연장 ─────────────────────────────────────────────────────────

    def _check_qt_prolongation(self, drugs: list[str], result: RiskAssessment):
        """QT 연장 약물 3종 이상 병용 탐지."""
        qt_drugs = self._matcher.drugs_in_group(drugs, "qt_prolonging")
        result.qt_drug_count = len(qt_drugs)

        if len(qt_drugs) >= 3:
            pair = DDIPair(
                drug_a=qt_drugs[0],
                drug_b=f"(+{len(qt_drugs)-1}종 QT 연장 약물)",
                severity="Major",
                description=f"QT 연장 약물 {len(qt_drugs)}종 병용 → Torsades de Pointes 위험",
                source="QT_RULE",
                rule_id="TOP09",
            )
            self._add_ddi(result, pair)
            if "TOP09: QT_Prolongation_Multiple" not in result.triggered_rules:
                result.triggered_rules.append("TOP09: QT_Prolongation_Multiple")

    # ─── DDI 추가 (중복 방지) ─────────────────────────────────────────────

    def _add_ddi(self, result: RiskAssessment, pair: DDIPair):
        """DDI 쌍 추가 + 심각도 카운터 업데이트."""
        # 중복 체크 (동일 쌍 + 동일 심각도)
        key = frozenset([pair.drug_a.lower(), pair.drug_b.lower()])
        for existing in result.ddi_pairs:
            if frozenset([existing.drug_a.lower(), existing.drug_b.lower()]) == key:
                # 더 높은 심각도로 업데이트
                if SEVERITY_RANK.get(pair.severity, 4) < SEVERITY_RANK.get(existing.severity, 4):
                    result.ddi_pairs.remove(existing)
                    self._decrement_counter(result, existing.severity)
                    break
                else:
                    return   # 기존이 더 높거나 같음

        result.ddi_pairs.append(pair)
        self._increment_counter(result, pair.severity)

    @staticmethod
    def _increment_counter(result: RiskAssessment, severity: str):
        if severity == "Contraindicated":
            result.ddi_contraindicated_count += 1
        elif severity == "Major":
            result.ddi_major_count += 1
        elif severity == "Moderate":
            result.ddi_moderate_count += 1
        elif severity == "Minor":
            result.ddi_minor_count += 1

    @staticmethod
    def _decrement_counter(result: RiskAssessment, severity: str):
        if severity == "Contraindicated":
            result.ddi_contraindicated_count = max(0, result.ddi_contraindicated_count - 1)
        elif severity == "Major":
            result.ddi_major_count = max(0, result.ddi_major_count - 1)
        elif severity == "Moderate":
            result.ddi_moderate_count = max(0, result.ddi_moderate_count - 1)
        elif severity == "Minor":
            result.ddi_minor_count = max(0, result.ddi_minor_count - 1)

    # ─── 위험도 등급 산출 ─────────────────────────────────────────────────

    def _determine_risk_grade(
        self,
        result: RiskAssessment,
        drug_count: int,
        patient_age: Optional[int],
        has_renal_risk: bool,
        has_hepatic_risk: bool,
    ):
        """CLINICAL_STANDARDS_v1.0.md 기준 위험도 등급 산출."""
        grade = "Normal"

        # 🔴 Red 조건
        red = False
        if result.ddi_contraindicated_count >= 1:
            red = True
        if result.ddi_major_count >= 3:
            red = True
        if result.triple_whammy_flag:
            red = True
        if result.qt_drug_count >= 3:
            red = True
        if drug_count >= 10 and self._has_high_risk_drug(result):
            red = True
        if (patient_age is not None and patient_age >= 75 and
                drug_count >= 5 and (has_renal_risk or has_hepatic_risk)):
            red = True

        if red:
            grade = "Red"
        elif (result.ddi_major_count >= 1 or
              result.ddi_moderate_count >= 2 or
              result.triple_whammy_flag):
            grade = "Yellow"
        elif (result.ddi_minor_count > 0 or
              (drug_count >= 5 and result.ddi_contraindicated_count == 0 and result.ddi_major_count == 0)):
            grade = "Green"
        else:
            grade = "Normal"

        result.risk_grade = grade

    @staticmethod
    def _has_high_risk_drug(result: RiskAssessment) -> bool:
        """고위험 약물 포함 여부 (DDI 쌍에서 확인)."""
        high_risk_keywords = [
            "warfarin", "methotrexate", "lithium", "digoxin", "amiodarone",
            "phenytoin", "cyclosporine", "tacrolimus", "theophylline",
        ]
        for pair in result.ddi_pairs:
            for kw in high_risk_keywords:
                if kw in pair.drug_a.lower() or kw in pair.drug_b.lower():
                    return True
        return False

    # ─── 설명 생성 ────────────────────────────────────────────────────────

    def _build_explanation(self, result: RiskAssessment) -> str:
        parts = []
        grade_map = {
            "Red": "🔴 고위험 - 즉각 개입 필요",
            "Yellow": "🟡 중위험 - 월 1회 모니터링",
            "Green": "🟢 저위험 - 분기 1회 안내",
            "Normal": "⚪ 정상",
        }
        parts.append(grade_map.get(result.risk_grade, result.risk_grade))

        if result.ddi_contraindicated_count:
            parts.append(f"절대금기 DDI {result.ddi_contraindicated_count}건")
        if result.ddi_major_count:
            parts.append(f"주요 DDI {result.ddi_major_count}건")
        if result.triple_whammy_flag:
            parts.append("Triple Whammy 해당")
        if result.qt_drug_count >= 3:
            parts.append(f"QT 연장 약물 {result.qt_drug_count}종")

        # 주요 DDI 쌍 설명
        critical_pairs = [p for p in result.ddi_pairs if p.severity in ("Contraindicated", "Major")][:3]
        for pair in critical_pairs:
            parts.append(f"{pair.drug_a} + {pair.drug_b} ({pair.severity})")

        return "; ".join(parts)

    # ─── 유틸리티 ────────────────────────────────────────────────────────

    def info(self) -> str:
        n_ddi = len(self._ddi_matrix) if not self._ddi_matrix.empty else 0
        n_drugs = len(self._drug_index) if not self._drug_index.empty else 0
        return (
            f"SafetyNet 로드 완료\n"
            f"  DDI 매트릭스: {n_ddi:,} 쌍\n"
            f"  약물 인덱스: {n_drugs:,} 약물\n"
            f"  Top 10 DDI 규칙: {len(self._top10_rules)}개\n"
            f"  약물 그룹: {len(self._drug_groups)}개"
        )

    def get_ddi_severity(self, drug_a: str, drug_b: str) -> str:
        """두 약물 간 DDI 심각도 조회."""
        if self._ddi_matrix.empty:
            return "Unknown"
        a_lower = drug_a.lower().strip()
        b_lower = drug_b.lower().strip()
        mask = (
            (self._ddi_matrix["_a_lower"].str.contains(a_lower, na=False) &
             self._ddi_matrix["_b_lower"].str.contains(b_lower, na=False)) |
            (self._ddi_matrix["_a_lower"].str.contains(b_lower, na=False) &
             self._ddi_matrix["_b_lower"].str.contains(a_lower, na=False))
        )
        matches = self._ddi_matrix[mask]
        if matches.empty:
            return "None"
        # 가장 높은 심각도 반환
        severities = matches["severity"].tolist()
        severities.sort(key=lambda s: SEVERITY_RANK.get(s, 4))
        return severities[0]


# ─── CLI 인터페이스 ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SafetyNet 약물 DDI 평가")
    parser.add_argument("drugs", nargs="+", help="평가할 약물명 목록 (공백 구분)")
    parser.add_argument("--age", type=int, default=None, help="환자 나이")
    parser.add_argument("--renal-risk", action="store_true", help="신기능저하 약물 포함")
    args = parser.parse_args()

    sn = SafetyNet()
    print(sn.info())
    print("\n" + "=" * 60)
    print(f"입력 약물: {args.drugs}")
    print("=" * 60)

    result = sn.assess(
        drugs=args.drugs,
        patient_age=args.age,
        has_renal_risk=args.renal_risk,
    )
    print(result.summary)
