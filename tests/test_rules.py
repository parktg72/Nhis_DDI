#!/usr/bin/env python3
"""
Rule-based Safety Net 단위 테스트

QA_PLAN_v1.0.md 기준:
  - Top 10 DDI 탐지율 100%
  - Critical Error 0건 (고위험 미탐지 절대 금지)
  - Contraindicated DDI 100% 탐지

실행:
  pytest tests/test_rules.py -v
  pytest tests/test_rules.py -v --tb=short -k "top10"
"""
import sys
from pathlib import Path

import pytest
import yaml

# 프로젝트 루트를 sys.path에 추가
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from rules.safety_net import SafetyNet, RiskAssessment
from rules.duplicate_detector import DuplicateDetector


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def safety_net():
    """SafetyNet 인스턴스 (DDI 매트릭스 없이 규칙만 사용)."""
    return SafetyNet(
        ddi_matrix_path=Path("nonexistent_path_for_test"),  # 규칙만 테스트
        drug_index_path=Path("nonexistent_path_for_test"),
    )


@pytest.fixture(scope="module")
def duplicate_detector():
    return DuplicateDetector()


@pytest.fixture(scope="module")
def drug_rules():
    rules_path = ROOT / "config" / "drug_rules.yaml"
    with open(rules_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ─── Top 10 DDI 탐지 테스트 ───────────────────────────────────────────────────

class TestTop10DDI:
    """TOP01~TOP10 100% 탐지 보장 테스트."""

    def test_top01_warfarin_nsaid_detected(self, safety_net):
        """TOP01: 와파린 + NSAIDs → 탐지 필수."""
        result = safety_net.assess(["warfarin", "ibuprofen"])
        assert any("TOP01" in r for r in result.triggered_rules), \
            f"TOP01 미탐지. triggered_rules={result.triggered_rules}"
        assert result.risk_grade in ("Red", "Yellow"), \
            f"고위험/중위험 예상. 실제={result.risk_grade}"

    def test_top01_doac_nsaid_detected(self, safety_net):
        """TOP01: DOAC (rivaroxaban) + NSAIDs."""
        result = safety_net.assess(["rivaroxaban", "naproxen"])
        assert any("TOP01" in r for r in result.triggered_rules)

    def test_top02_clopidogrel_omeprazole(self, safety_net):
        """TOP02: 클로피도그렐 + 오메프라졸."""
        result = safety_net.assess(["clopidogrel", "omeprazole"])
        assert any("TOP02" in r for r in result.triggered_rules), \
            f"TOP02 미탐지. triggered_rules={result.triggered_rules}"

    def test_top03_triple_whammy(self, safety_net):
        """TOP03: Triple Whammy - ACEi + K보존이뇨제 + NSAIDs."""
        result = safety_net.assess(["enalapril", "spironolactone", "ibuprofen"])
        assert result.triple_whammy_flag, "Triple Whammy 미탐지"
        assert result.risk_grade == "Red", f"Triple Whammy는 Red여야 함. 실제={result.risk_grade}"

    def test_top03_arb_variant(self, safety_net):
        """TOP03: ARB 변형 - losartan + spironolactone + naproxen."""
        result = safety_net.assess(["losartan", "spironolactone", "naproxen"])
        assert result.triple_whammy_flag

    def test_top04_digoxin_amiodarone(self, safety_net):
        """TOP04: 디곡신 + 아미오다론."""
        result = safety_net.assess(["digoxin", "amiodarone"])
        assert any("TOP04" in r for r in result.triggered_rules), \
            f"TOP04 미탐지. triggered_rules={result.triggered_rules}"

    def test_top04_digoxin_verapamil(self, safety_net):
        """TOP04: 디곡신 + 베라파밀."""
        result = safety_net.assess(["digoxin", "verapamil"])
        assert any("TOP04" in r for r in result.triggered_rules)

    def test_top05_methotrexate_trimethoprim(self, safety_net):
        """TOP05: 메토트렉세이트 + 트리메토프림 (Contraindicated)."""
        result = safety_net.assess(["methotrexate", "trimethoprim"])
        assert any("TOP05" in r for r in result.triggered_rules)
        assert result.ddi_contraindicated_count >= 1, "Contraindicated 미탐지"
        assert result.risk_grade == "Red", f"Contraindicated는 Red여야 함. 실제={result.risk_grade}"

    def test_top06_ssri_maoi_contraindicated(self, safety_net):
        """TOP06: SSRI + MAOi → 세로토닌 증후군 (Contraindicated)."""
        result = safety_net.assess(["fluoxetine", "phenelzine"])
        assert any("TOP06" in r for r in result.triggered_rules)
        assert result.ddi_contraindicated_count >= 1
        assert result.risk_grade == "Red"

    def test_top07_ssri_triptan(self, safety_net):
        """TOP07: SSRI + Triptan."""
        result = safety_net.assess(["sertraline", "sumatriptan"])
        assert any("TOP07" in r for r in result.triggered_rules), \
            f"TOP07 미탐지. triggered_rules={result.triggered_rules}"

    def test_top08_lithium_nsaid(self, safety_net):
        """TOP08: 리튬 + NSAIDs."""
        result = safety_net.assess(["lithium", "ibuprofen"])
        assert any("TOP08" in r for r in result.triggered_rules)

    def test_top08_lithium_furosemide(self, safety_net):
        """TOP08: 리튬 + 이뇨제."""
        result = safety_net.assess(["lithium", "furosemide"])
        assert any("TOP08" in r for r in result.triggered_rules)

    def test_top09_qt_prolongation_3drugs(self, safety_net):
        """TOP09: QT 연장 약물 3종 이상."""
        result = safety_net.assess(["amiodarone", "haloperidol", "azithromycin"])
        assert any("TOP09" in r for r in result.triggered_rules), \
            f"TOP09 미탐지. triggered_rules={result.triggered_rules}"
        assert result.qt_drug_count >= 3

    def test_top09_qt_2drugs_no_trigger(self, safety_net):
        """TOP09: QT 연장 약물 2종은 발동 안 됨."""
        result = safety_net.assess(["amiodarone", "haloperidol"])
        assert not any("TOP09" in r for r in result.triggered_rules)

    def test_top10_statin_clarithromycin(self, safety_net):
        """TOP10: 스타틴 + 클래리스로마이신."""
        result = safety_net.assess(["atorvastatin", "clarithromycin"])
        assert any("TOP10" in r for r in result.triggered_rules), \
            f"TOP10 미탐지. triggered_rules={result.triggered_rules}"


# ─── 위험도 등급 산출 테스트 ─────────────────────────────────────────────────

class TestRiskGrading:

    def test_contraindicated_always_red(self, safety_net):
        """Contraindicated DDI → 무조건 Red."""
        result = safety_net.assess(["methotrexate", "trimethoprim"])
        assert result.risk_grade == "Red", \
            f"Contraindicated는 반드시 Red. 실제={result.risk_grade}"

    def test_triple_whammy_always_red(self, safety_net):
        """Triple Whammy → 무조건 Red."""
        result = safety_net.assess(["lisinopril", "spironolactone", "diclofenac"])
        assert result.risk_grade == "Red"

    def test_major_ddi_3plus_red(self, safety_net):
        """Major DDI 3건 이상 → Red."""
        # warfarin+ibuprofen (TOP01), digoxin+amiodarone (TOP04), lithium+ibuprofen (TOP08)
        result = safety_net.assess(["warfarin", "ibuprofen", "digoxin", "amiodarone", "lithium"])
        assert result.risk_grade == "Red", \
            f"Major DDI 3건 이상은 Red. major_count={result.ddi_major_count}"

    def test_elderly_polypharmacy_risk(self, safety_net):
        """75세 이상 + 5종 이상 + 신기능저하 → Red."""
        drugs = ["amlodipine", "metformin", "atorvastatin", "aspirin", "omeprazole"]
        result = safety_net.assess(drugs, patient_age=78, has_renal_risk=True)
        assert result.risk_grade == "Red"

    def test_no_ddi_normal(self, safety_net):
        """DDI 없는 단순 처방 → Normal 또는 Green."""
        result = safety_net.assess(["amlodipine"])
        assert result.risk_grade in ("Normal", "Green")

    def test_empty_drugs_normal(self, safety_net):
        """빈 약물 목록 → Normal."""
        result = safety_net.assess([])
        assert result.risk_grade == "Normal"


# ─── Critical Error 방지 테스트 ───────────────────────────────────────────────

class TestCriticalErrorPrevention:
    """QA_PLAN_v1.0.md Critical Error 기준: 절대 미탐지 금지."""

    CRITICAL_CASES = [
        # (케이스 설명, 약물 목록, 예상_최소_등급)
        ("SSRI+MAOi 세로토닌증후군", ["fluoxetine", "tranylcypromine"], "Red"),
        ("메토트렉세이트+트리메토프림 골수억제", ["methotrexate", "co-trimoxazole"], "Red"),
        ("와파린+NSAIDs 출혈", ["warfarin", "naproxen"], "Yellow"),
        ("Triple Whammy", ["ramipril", "amiloride", "celecoxib"], "Red"),
        ("디곡신+아미오다론 독성", ["digoxin", "amiodarone"], "Yellow"),
    ]

    @pytest.mark.parametrize("desc,drugs,min_grade", CRITICAL_CASES)
    def test_no_critical_miss(self, safety_net, desc, drugs, min_grade):
        """Critical 케이스가 Normal로 분류되지 않아야 함."""
        result = safety_net.assess(drugs)
        grade_rank = {"Red": 0, "Yellow": 1, "Green": 2, "Normal": 3}
        actual_rank = grade_rank[result.risk_grade]
        expected_rank = grade_rank[min_grade]
        assert actual_rank <= expected_rank, (
            f"Critical Error! [{desc}] "
            f"약물={drugs} → 예상={min_grade}, 실제={result.risk_grade}"
        )


# ─── 중복약물 탐지 테스트 ────────────────────────────────────────────────────

class TestDuplicateDetector:

    def test_level1_same_ingredient(self, duplicate_detector):
        """Level 1: 완전 동일 성분 탐지."""
        drugs = [
            {"name": "amlodipine_A", "atc": "C08CA01"},
            {"name": "amlodipine_B", "atc": "C08CA01"},
        ]
        result = duplicate_detector.detect(drugs)
        assert result.duplicate_level1_count >= 1, "Level 1 중복 미탐지"

    def test_level2_same_pharmacological_subgroup(self, duplicate_detector):
        """Level 2: ATC 4단계 동일."""
        drugs = [
            {"name": "amlodipine", "atc": "C08CA01"},
            {"name": "nifedipine",  "atc": "C08CA05"},
        ]
        result = duplicate_detector.detect(drugs)
        assert result.duplicate_level2_count >= 1, "Level 2 중복 미탐지"

    def test_level3_same_therapeutic_group(self, duplicate_detector):
        """Level 3: ATC 3단계 동일 (4자리 prefix 동일, 5자리 다름)."""
        drugs = [
            {"name": "amoxicillin", "atc": "J01CA04"},  # 광범위 페니실린 (J01C + A)
            {"name": "cloxacillin", "atc": "J01CF02"},  # 베타락탐분해효소저항 페니실린 (J01C + F)
        ]
        result = duplicate_detector.detect(drugs)
        # J01CA vs J01CF → ATC 3단계(4자리) J01C 동일, 4단계(5자리) 다름 → Level 3
        assert result.duplicate_level3_count >= 1, \
            f"Level 3 중복 미탐지. duplicates={result.duplicates}"

    def test_antihypertensive_exception_e1(self, duplicate_detector):
        """E1 예외: 항고혈압제 다제병용은 허용."""
        drugs = [
            {"name": "amlodipine",             "atc": "C08CA01"},
            {"name": "ramipril",               "atc": "C09AA05"},
            {"name": "hydrochlorothiazide",    "atc": "C03AA03"},
        ]
        result = duplicate_detector.detect(drugs)
        # E1 예외가 적용되어 Level 3 미계상
        level3_not_allowed = [d for d in result.duplicates if d.level == 3 and not d.is_allowed]
        assert len(level3_not_allowed) == 0 or "E1" in result.exception_applied, \
            "E1 예외 미적용"

    def test_no_duplicate_different_classes(self, duplicate_detector):
        """완전히 다른 약물군 → 중복 없음."""
        drugs = [
            {"name": "metformin",   "atc": "A10BA02"},
            {"name": "atorvastatin", "atc": "C10AA05"},
            {"name": "amlodipine",  "atc": "C08CA01"},
        ]
        result = duplicate_detector.detect(drugs)
        # Level 1 중복 없어야 함
        assert result.duplicate_level1_count == 0


# ─── 규칙 설정 파일 유효성 테스트 ────────────────────────────────────────────

class TestRulesConfig:

    def test_all_top10_rules_present(self, drug_rules):
        """Top 10 DDI 규칙이 모두 존재."""
        rules = drug_rules.get("top10_ddi_rules", [])
        rule_ids = {r["id"] for r in rules}
        expected = {f"TOP{i:02d}" for i in range(1, 11)}
        missing = expected - rule_ids
        assert not missing, f"누락된 Top 10 규칙: {missing}"

    def test_all_top10_have_required_fields(self, drug_rules):
        """Top 10 규칙에 필수 필드 존재."""
        required_fields = {"id", "name", "description", "severity", "mechanism", "clinical_risk"}
        for rule in drug_rules.get("top10_ddi_rules", []):
            missing = required_fields - set(rule.keys())
            assert not missing, f"규칙 {rule.get('id')} 누락 필드: {missing}"

    def test_drug_groups_defined(self, drug_rules):
        """필수 약물 그룹이 정의되어 있음."""
        groups = drug_rules.get("drug_groups", {})
        required_groups = [
            "warfarin", "doac", "anticoagulants_all", "nsaids",
            "clopidogrel", "ppi_cyp2c19", "acei", "arb",
            "k_sparing_diuretics", "digoxin", "ssri", "maoi",
            "methotrexate", "lithium", "qt_prolonging", "statin",
        ]
        for group in required_groups:
            assert group in groups, f"약물 그룹 '{group}' 미정의"

    def test_exception_rules_e1_to_e9(self, drug_rules):
        """중복약물 예외 E1~E9 정의 확인."""
        exceptions = drug_rules.get("duplicate_exceptions", {})
        for i in range(1, 10):
            exc_code = f"E{i}"
            assert exc_code in exceptions, f"예외 규칙 '{exc_code}' 미정의"

    def test_severity_values_valid(self, drug_rules):
        """모든 DDI 규칙의 severity 값이 유효."""
        valid = {"Contraindicated", "Major", "Moderate", "Minor"}
        for rule in drug_rules.get("top10_ddi_rules", []):
            sev = rule.get("severity")
            assert sev in valid, f"규칙 {rule['id']} 유효하지 않은 severity: {sev}"


# ─── 성능 테스트 ─────────────────────────────────────────────────────────────

class TestPerformance:

    def test_assessment_completes_quickly(self, safety_net):
        """100개 약물 평가가 1초 내 완료."""
        import time
        drugs = [f"drug_{i}" for i in range(100)]
        start = time.time()
        result = safety_net.assess(drugs)
        elapsed = time.time() - start
        assert elapsed < 1.0, f"평가 시간 초과: {elapsed:.2f}초"

    def test_assessment_is_deterministic(self, safety_net):
        """동일 입력에 동일 결과."""
        drugs = ["warfarin", "ibuprofen", "fluoxetine"]
        result1 = safety_net.assess(drugs)
        result2 = safety_net.assess(drugs)
        assert result1.risk_grade == result2.risk_grade
        assert set(result1.triggered_rules) == set(result2.triggered_rules)


# ─── _has_high_risk_drug yaml 동기 회귀 (Codex 2026-05-06 ISSUE-3b) ───────────

class TestHasHighRiskDrugYamlAlignment:
    """drug_rules.yaml :123 의 15개 keyword 모두 _has_high_risk_drug 가 인식해야 함.

    직전까지 hardcoded 9개 list 였음 — sirolimus / insulin / clozapine /
    carbamazepine / valproate / phenobarbital 6개 누락. drug_count >= 10 AND
    has_high_risk_drug (drug_rules.yaml :255) 분기에서 해당 환자가 잘못 grade 됨.
    yaml 1차 자료 동기 후 회귀 가드.
    """

    @pytest.mark.parametrize("high_risk_drug", [
        # 직전까지 누락되어 있던 6개 — yaml :123 에 따르면 모두 high-risk
        "sirolimus", "insulin", "clozapine",
        "carbamazepine", "valproate", "phenobarbital",
    ])
    def test_newly_aligned_drug_detected(self, safety_net, high_risk_drug):
        """yaml 정의된 약물이 _has_high_risk_drug 에서 인식되는지."""
        from rules.safety_net import RiskAssessment
        result = RiskAssessment(risk_grade="Normal")
        result.input_drugs = [high_risk_drug]
        assert safety_net._has_high_risk_drug(result), (
            f"yaml :123 정의 약물 '{high_risk_drug}' 이 _has_high_risk_drug 에서 "
            f"미인식 — drug_rules.yaml 동기 깨짐"
        )

    @pytest.mark.parametrize("legacy_drug", [
        # 직전부터 인식되던 9개 — 회귀 방지
        "warfarin", "methotrexate", "lithium", "digoxin", "amiodarone",
        "phenytoin", "cyclosporine", "tacrolimus", "theophylline",
    ])
    def test_legacy_drug_still_detected(self, safety_net, legacy_drug):
        """기존 인식 약물 회귀 방지."""
        from rules.safety_net import RiskAssessment
        result = RiskAssessment(risk_grade="Normal")
        result.input_drugs = [legacy_drug]
        assert safety_net._has_high_risk_drug(result), (
            f"기존 인식 약물 '{legacy_drug}' 이 _has_high_risk_drug 에서 회귀"
        )


# ─── get_ddi_severity regex 특수문자 회귀 가드 (Codex 2026-05-06 ISSUE-6) ─────

class TestGetDDISeverityRegexEscape:
    """get_ddi_severity 의 str.contains 가 약물명을 literal 로 매칭해야 함.

    배경: _apply_matrix_ddi (line 325-326) 는 이미 regex=False 사용.
    하지만 get_ddi_severity (line 560-563) 는 regex=False 누락 → 약물명에
    +, (, ), ., * 등이 포함되면 regex 메타문자로 해석되어 오탐/누락 발생.
    Qwen 2026-05-06 후속 검토에서 식별, schema-grade 결함은 아니지만 cross-
    family 동일 가드 (학습-서빙 parity 와 같은 정신).
    """

    @staticmethod
    def _make_sn_with_matrix(matrix_df):
        """SafetyNet 인스턴스에 mini DDI matrix 직접 주입."""
        import pandas as pd  # noqa
        sn = SafetyNet(
            ddi_matrix_path=Path("nonexistent_path_for_test"),
            drug_index_path=Path("nonexistent_path_for_test"),
        )
        sn._ddi_matrix = matrix_df
        return sn

    def test_period_no_false_positive(self):
        """검색 입력 'drug.name' 의 . 가 wildcard 로 해석되어 False positive 발생 X.

        str.contains(pattern, target) — pattern 이 user 입력 'drug.name'.
        regex=True: . 가 wildcard 라 target 'drugxname' 매치 → False pos "Major".
        regex=False: literal '.' → 매치 X → "None" (정답).
        """
        import pandas as pd
        df = pd.DataFrame({
            "drug_a_name": ["drugxname"],
            "drug_b_name": ["partner_b"],
            "_a_lower":    ["drugxname"],   # innocent literal string (no .)
            "_b_lower":    ["partner_b"],
            "severity":    ["Major"],
        })
        sn = self._make_sn_with_matrix(df)
        # 검색 입력에만 메타문자 . 포함
        result = sn.get_ddi_severity("drug.name", "partner_b")
        assert result == "None", (
            f"회귀: regex 메타 . 가 wildcard 로 해석되어 'drugxname' 에 False "
            f"positive 매치. severity={result}"
        )

    def test_plus_self_match(self):
        """약물명 'drug+plus' 의 + 가 regex 메타로 해석되지 않아야 함.

        regex=True 면 'drug+plus' 의 + 가 'g 1회 이상' 의미라 자기 자신 미매치.
        """
        import pandas as pd
        df = pd.DataFrame({
            "drug_a_name": ["drug+plus"],
            "drug_b_name": ["partner_a"],
            "_a_lower":    ["drug+plus"],
            "_b_lower":    ["partner_a"],
            "severity":    ["Major"],
        })
        sn = self._make_sn_with_matrix(df)
        result = sn.get_ddi_severity("drug+plus", "partner_a")
        assert result == "Major", (
            f"회귀: 'drug+plus' 자기-매치 실패 — + 가 regex 메타로 해석됨. "
            f"severity={result}"
        )

    def test_paren_no_regex_error(self):
        """약물명 'drug(brand)' 의 () 가 regex group 으로 해석되지 않아야 함.

        regex=True 면 () 가 group capture 라 매치 결과 다르거나 해석 모호.
        """
        import pandas as pd
        df = pd.DataFrame({
            "drug_a_name": ["drug(brand)"],
            "drug_b_name": ["partner_c"],
            "_a_lower":    ["drug(brand)"],
            "_b_lower":    ["partner_c"],
            "severity":    ["Major"],
        })
        sn = self._make_sn_with_matrix(df)
        result = sn.get_ddi_severity("drug(brand)", "partner_c")
        assert result == "Major", (
            f"회귀: 'drug(brand)' 처리 실패 — () 가 regex 메타로 해석됨. "
            f"severity={result}"
        )
