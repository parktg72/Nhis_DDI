"""
prescription_aggregator 단위 테스트

핵심 검증:
  - _assign_risk_level: CLINICAL_STANDARDS_v1.0.md 기준 일치
  - _fill_risk_drug_flags: 고위험/신기능/간기능 약물 탐지
  - aggregate_patient_features: 피처 집계 정합성

실행:
  pytest tests/test_etl/test_prescription_aggregator.py -v
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from scripts.etl.drug_ontology import SEVERITY_ORDER
from scripts.etl.models import PatientFeatures, PrescriptionRecord
from scripts.etl.prescription_aggregator import (
    _HEPATIC_RISK_ATC_PREFIXES,
    _HEPATIC_RISK_KEYWORDS,
    _HIGH_RISK_ATC_PREFIXES,
    _HIGH_RISK_KEYWORDS,
    _RENAL_RISK_ATC_PREFIXES,
    _RENAL_RISK_KEYWORDS,
    _SEVERITY_ORDER,
    _assign_risk_level,
    _assign_yellow_subtype,
    _check_risk_drugs,
    _ddi_lookup_cache,
    _fill_dup_features,
    _fill_risk_drug_flags,
    aggregate_patient_features,
    count_ddi_severities,
    ddi_pair_severities,
)

# ─── 헬퍼 ───────────────────────────────────────────────────────────────────

def _make_features(**kwargs) -> PatientFeatures:
    """기본 PatientFeatures 생성 (필수 필드 채움)."""
    defaults = dict(
        patient_id="P000001",
        window_start=date(2024, 1, 1),
        window_end=date(2024, 3, 31),
    )
    defaults.update(kwargs)
    return PatientFeatures(**defaults)


def _make_rx(
    drug_name: str = "",
    atc_code: str = "",
    wk_compn_cd: str = "100000001",
    start: date | None = None,
    days: int = 30,
) -> PrescriptionRecord:
    start = start or date(2024, 1, 1)
    return PrescriptionRecord(
        patient_id="P000001",
        institution_id="INST001",
        bill_no="BILL001",
        wk_compn_cd=wk_compn_cd,
        drug_name=drug_name,
        atc_code=atc_code,
        start_date=start,
        end_date=start + timedelta(days=days - 1),
        total_days=days,
    )


def _make_pair(a: str, b: str):
    from scripts.etl.models import DrugOverlapPair

    return DrugOverlapPair(
        patient_id="P000001",
        drug_a_wk_compn=a,
        drug_a_edi=None,
        drug_a_atc=None,
        drug_a_name=None,
        drug_b_wk_compn=b,
        drug_b_edi=None,
        drug_b_atc=None,
        drug_b_name=None,
        overlap_start=date(2024, 1, 1),
        overlap_end=date(2024, 1, 5),
        overlap_days=5,
        window_start=date(2024, 1, 1),
        window_end=date(2024, 3, 30),
    )


class _FakeDrugMaster:
    def __init__(self, mapping: dict[str, list[str]]) -> None:
        self.mapping = mapping

    def get_ddi_ids(self, wk_compn_cd: str) -> list[str]:
        return self.mapping.get(str(wk_compn_cd), [])


class TestDDIOntologyCompatibility:
    def setup_method(self) -> None:
        _ddi_lookup_cache.clear()

    def test_private_severity_order_aliases_new_single_source(self):
        assert _SEVERITY_ORDER == SEVERITY_ORDER

    def test_ddi_lookup_cache_clear_remains_available(self):
        ddi_matrix = pd.DataFrame([
            {"drug_a_id": "D_A", "drug_b_id": "D_B", "severity": "Major"},
        ])
        pairs = [_make_pair("WK_A", "WK_B")]
        master = _FakeDrugMaster({"WK_A": ["D_A"], "WK_B": ["D_B"]})

        assert ddi_pair_severities(pairs, ddi_matrix, master) == [(pairs[0], "Major")]
        assert _ddi_lookup_cache

        _ddi_lookup_cache.clear()
        assert _ddi_lookup_cache == {}

    def test_ddi_pair_severities_and_counts_cover_all_severities(self):
        ddi_matrix = pd.DataFrame([
            {"drug_a_id": "D_CON_A", "drug_b_id": "D_CON_B", "severity": "Contraindicated"},
            {"drug_a_id": "D_MAJ_A", "drug_b_id": "D_MAJ_B", "severity": "Major"},
            {"drug_a_id": "D_MOD_A", "drug_b_id": "D_MOD_B", "severity": "Moderate"},
            {"drug_a_id": "D_MIN_A", "drug_b_id": "D_MIN_B", "severity": "Minor"},
        ])
        pairs = [
            _make_pair("WK_CON_A", "WK_CON_B"),
            _make_pair("WK_MAJ_A", "WK_MAJ_B"),
            _make_pair("WK_MOD_A", "WK_MOD_B"),
            _make_pair("WK_MIN_A", "WK_MIN_B"),
        ]
        master = _FakeDrugMaster({
            "WK_CON_A": ["D_CON_A"],
            "WK_CON_B": ["D_CON_B"],
            "WK_MAJ_A": ["D_MAJ_A"],
            "WK_MAJ_B": ["D_MAJ_B"],
            "WK_MOD_A": ["D_MOD_A"],
            "WK_MOD_B": ["D_MOD_B"],
            "WK_MIN_A": ["D_MIN_A"],
            "WK_MIN_B": ["D_MIN_B"],
        })

        assert ddi_pair_severities(pairs, ddi_matrix, master) == [
            (pairs[0], "Contraindicated"),
            (pairs[1], "Major"),
            (pairs[2], "Moderate"),
            (pairs[3], "Minor"),
        ]
        assert count_ddi_severities(pairs, ddi_matrix, master) == {
            "Contraindicated": 1,
            "Major": 1,
            "Moderate": 1,
            "Minor": 1,
        }

    def test_combo_cross_product_and_unmapped_pairs_match_existing_behavior(self):
        ddi_matrix = pd.DataFrame([
            {"drug_a_id": "D_A", "drug_b_id": "D_B", "severity": "Minor"},
            {"drug_a_id": "D_C", "drug_b_id": "D_B", "severity": "Major"},
        ])
        combo = _make_pair("WK_COMBO", "WK_B")
        unmapped = _make_pair("WK_UNKNOWN", "WK_B")
        master = _FakeDrugMaster({
            "WK_COMBO": ["D_A", "D_C"],
            "WK_B": ["D_B"],
        })

        result = ddi_pair_severities([combo, unmapped], ddi_matrix, master)

        assert result == [(combo, "Major")]
        assert result[0][0] is combo
        assert count_ddi_severities([combo, unmapped], ddi_matrix, master) == {
            "Contraindicated": 0,
            "Major": 1,
            "Moderate": 0,
            "Minor": 0,
        }


# ─── _assign_risk_level 테스트 ───────────────────────────────────────────────

class TestAssignRiskLevel:
    """CLINICAL_STANDARDS_v1.0.md 위험도 기준 일치 검증."""

    def test_contraindicated_ddi_is_red(self):
        feat = _make_features(ddi_contraindicated=1)
        _assign_risk_level(feat)
        assert feat.risk_level == "Red"

    def test_major_ddi_is_ddi_major_pharmacist(self):
        """2026-06-07: major DDI(≥1, ≥3 포함) → Y_DDI_MAJOR(약사전화), Red/Y_TRIPLE 아님."""
        feat = _make_features(ddi_major=3)
        _assign_risk_level(feat)
        assert feat.risk_level == "Yellow"
        _assign_yellow_subtype(feat)
        assert feat.yellow_subtype == "Y_DDI_MAJOR"

    def test_major_ddi_2_is_yellow(self):
        feat = _make_features(ddi_major=2)
        _assign_risk_level(feat)
        assert feat.risk_level == "Yellow"

    def test_triple_whammy_is_severe_ytriple(self):
        feat = _make_features(triple_whammy=True)
        _assign_risk_level(feat)
        assert feat.risk_level == "Yellow"
        _assign_yellow_subtype(feat)
        assert feat.yellow_subtype == "Y_TRIPLE"

    def test_10drugs_with_high_risk_is_severe_ytriple(self):
        """10종 이상 + 고위험 약물 → Yellow/Y_TRIPLE(즉시개입), Red 아님."""
        feat = _make_features(drug_count=12, has_high_risk_drug=True)
        _assign_risk_level(feat)
        assert feat.risk_level == "Yellow"
        assert any("SEV_10DRUG_HIGHRISK" in r for r in feat.risk_reasons)
        _assign_yellow_subtype(feat)
        assert feat.yellow_subtype == "Y_TRIPLE"

    def test_10drugs_without_high_risk_not_red(self):
        """10종 이상이지만 고위험 약물 없으면 Red 아님."""
        feat = _make_features(drug_count=12, has_high_risk_drug=False)
        _assign_risk_level(feat)
        assert feat.risk_level != "Red"

    def test_elderly_with_renal_risk_is_severe_ytriple(self):
        """75세 이상 + 5종 이상 + 신기능 저하 약물 → Yellow/Y_TRIPLE(즉시개입)."""
        feat = _make_features(age=78, drug_count=6, has_renal_risk_drug=True)
        _assign_risk_level(feat)
        assert feat.risk_level == "Yellow"
        assert any("SEV_ELDERLY_ORGAN" in r for r in feat.risk_reasons)
        _assign_yellow_subtype(feat)
        assert feat.yellow_subtype == "Y_TRIPLE"

    def test_elderly_with_hepatic_risk_is_severe_ytriple(self):
        """75세 이상 + 5종 이상 + 간기능 저하 약물 → Yellow/Y_TRIPLE(즉시개입)."""
        feat = _make_features(age=80, drug_count=7, has_hepatic_risk_drug=True)
        _assign_risk_level(feat)
        assert feat.risk_level == "Yellow"
        _assign_yellow_subtype(feat)
        assert feat.yellow_subtype == "Y_TRIPLE"

    def test_elderly_without_organ_risk_not_red(self):
        """75세 이상 + 5종 이상이지만 신기능/간기능 약물 없으면 Red 아님."""
        feat = _make_features(
            age=80, drug_count=7,
            has_renal_risk_drug=False, has_hepatic_risk_drug=False,
        )
        _assign_risk_level(feat)
        assert feat.risk_level != "Red"

    def test_young_with_renal_risk_not_red(self):
        """75세 미만이면 신기능 약물 있어도 이 조건으로 Red 아님."""
        feat = _make_features(age=60, drug_count=7, has_renal_risk_drug=True)
        _assign_risk_level(feat)
        # Red 아닌 다른 등급 (다른 Red 조건 해당 안되면)
        assert feat.risk_level != "Red"

    def test_major_ddi_1_is_yellow(self):
        feat = _make_features(ddi_major=1)
        _assign_risk_level(feat)
        assert feat.risk_level == "Yellow"

    def test_moderate_ddi_2_is_yellow(self):
        feat = _make_features(ddi_moderate=2)
        _assign_risk_level(feat)
        assert feat.risk_level == "Yellow"

    def test_dup_same_ingredient_is_yellow(self):
        feat = _make_features(dup_same_ingredient=1)
        _assign_risk_level(feat)
        assert feat.risk_level == "Yellow"

    def test_3_institutions_is_yellow(self):
        feat = _make_features(institution_count=3)
        _assign_risk_level(feat)
        assert feat.risk_level == "Yellow"

    def test_minor_ddi_only_is_green(self):
        feat = _make_features(ddi_minor=1)
        _assign_risk_level(feat)
        assert feat.risk_level == "Green"

    def test_5drugs_no_ddi_is_green(self):
        feat = _make_features(drug_count=5)
        _assign_risk_level(feat)
        assert feat.risk_level == "Green"

    def test_no_risk_is_normal(self):
        feat = _make_features(drug_count=2)
        _assign_risk_level(feat)
        assert feat.risk_level == "Normal"

    def test_reasons_populated(self):
        feat = _make_features(ddi_contraindicated=2)
        _assign_risk_level(feat)
        assert len(feat.risk_reasons) > 0


# ─── _check_risk_drugs / _fill_risk_drug_flags 테스트 ────────────────────────

class TestRiskDrugFlags:

    def test_high_risk_by_name(self):
        rx = [_make_rx(drug_name="Warfarin 5mg")]
        assert _check_risk_drugs(rx, _HIGH_RISK_KEYWORDS, _HIGH_RISK_ATC_PREFIXES)

    def test_high_risk_by_atc(self):
        rx = [_make_rx(atc_code="B01AA03")]
        assert _check_risk_drugs(rx, _HIGH_RISK_KEYWORDS, _HIGH_RISK_ATC_PREFIXES)

    def test_no_high_risk(self):
        rx = [_make_rx(drug_name="amlodipine", atc_code="C08CA01")]
        assert not _check_risk_drugs(rx, _HIGH_RISK_KEYWORDS, _HIGH_RISK_ATC_PREFIXES)

    def test_renal_risk_nsaid(self):
        rx = [_make_rx(drug_name="Ibuprofen 400mg")]
        assert _check_risk_drugs(rx, _RENAL_RISK_KEYWORDS, _RENAL_RISK_ATC_PREFIXES)

    def test_renal_risk_by_atc(self):
        rx = [_make_rx(atc_code="M01AE01")]  # NSAIDs ATC prefix M01A
        assert _check_risk_drugs(rx, _RENAL_RISK_KEYWORDS, _RENAL_RISK_ATC_PREFIXES)

    def test_hepatic_risk(self):
        rx = [_make_rx(drug_name="Methotrexate 2.5mg")]
        assert _check_risk_drugs(rx, _HEPATIC_RISK_KEYWORDS, _HEPATIC_RISK_ATC_PREFIXES)

    def test_fill_flags_all_true(self):
        rx = [
            _make_rx(drug_name="warfarin"),       # high risk
            _make_rx(drug_name="ibuprofen"),       # renal risk
            _make_rx(drug_name="methotrexate"),    # hepatic risk + high risk
        ]
        feat = _make_features()
        _fill_risk_drug_flags(feat, rx)
        assert feat.has_high_risk_drug is True
        assert feat.has_renal_risk_drug is True
        assert feat.has_hepatic_risk_drug is True

    def test_fill_flags_all_false(self):
        rx = [_make_rx(drug_name="amlodipine", atc_code="C08CA01")]
        feat = _make_features()
        _fill_risk_drug_flags(feat, rx)
        assert feat.has_high_risk_drug is False
        assert feat.has_renal_risk_drug is False
        assert feat.has_hepatic_risk_drug is False


# ─── aggregate_patient_features 통합 테스트 ──────────────────────────────────

class TestAggregatePatientFeatures:

    def test_basic_aggregation(self):
        """기본 집계: 약물 수, 기관 수."""
        rxs = [
            PrescriptionRecord(
                patient_id="P001", institution_id="I001", bill_no="B1",
                wk_compn_cd="100000001",
                start_date=date(2024, 1, 1), end_date=date(2024, 1, 30), total_days=30,
            ),
            PrescriptionRecord(
                patient_id="P001", institution_id="I002", bill_no="B2",
                wk_compn_cd="200000002",
                start_date=date(2024, 1, 5), end_date=date(2024, 2, 3), total_days=30,
            ),
        ]
        feat = aggregate_patient_features(
            patient_id="P001",
            prescriptions=rxs,
            overlap_pairs=[],
            ddi_matrix=None,
            dup_groups=None,
            age=65,
        )
        assert feat.drug_count == 2
        assert feat.institution_count == 2
        assert feat.age == 65

    def test_empty_prescriptions(self):
        feat = aggregate_patient_features(
            patient_id="P001", prescriptions=[], overlap_pairs=[],
            ddi_matrix=None, dup_groups=None,
        )
        assert feat.drug_count == 0
        assert feat.risk_level == "Normal"

    def test_age_passed_through(self):
        """나이가 올바르게 전달되어 위험도 판정에 사용됨."""
        rxs = [
            PrescriptionRecord(
                patient_id="P001", institution_id="I001", bill_no=f"B{i}",
                wk_compn_cd=f"{i}0000000{i}",
                drug_name="ibuprofen" if i == 1 else f"drug_{i}",
                start_date=date(2024, 1, 1), end_date=date(2024, 3, 31), total_days=90,
            )
            for i in range(1, 7)  # 6종 약물
        ]
        feat = aggregate_patient_features(
            patient_id="P001",
            prescriptions=rxs,
            overlap_pairs=[],
            ddi_matrix=None,
            dup_groups=None,
            age=78,  # 75세 이상
        )
        # ibuprofen → has_renal_risk_drug=True, 75세+5종+ → 중증 Yellow/Y_TRIPLE(즉시개입, 재설계)
        assert feat.age == 78
        assert feat.has_renal_risk_drug is True
        assert feat.risk_level == "Yellow"
        assert feat.yellow_subtype == "Y_TRIPLE"


# ─── ATC 계층 피처 계산 정확성 테스트 ────────────────────────────────────────

class TestATCHierarchyFeatures:
    """
    ATC 계층별 중복 피처가 올바른 필드에 할당되는지 검증.

    ATC 계층:
      5단계(7자리) → dup_atc5 / dup_same_ingredient
      4단계(5자리 prefix) → dup_atc4
      3단계(4자리 prefix) → dup_atc3
    """

    def _make_feat(self) -> PatientFeatures:
        return PatientFeatures(
            patient_id="P001",
            window_start=date(2024, 1, 1),
            window_end=date(2024, 3, 31),
        )

    def _rx(self, atc: str) -> PrescriptionRecord:
        return PrescriptionRecord(
            patient_id="P001", institution_id="I001", bill_no="B1",
            wk_compn_cd="900000000",
            atc_code=atc,
            start_date=date(2024, 1, 1), end_date=date(2024, 1, 30), total_days=30,
        )

    def test_dup_atc5_exact_match(self):
        """동일 ATC 5단계(7자리) 2개 → dup_atc5 = 1."""
        rxs = [self._rx("A10BA02"), self._rx("A10BA02")]
        feat = self._make_feat()
        _fill_dup_features(feat, rxs, dup_groups=None)
        assert feat.dup_atc5 == 1, f"dup_atc5 expected 1, got {feat.dup_atc5}"

    def test_dup_atc4_same_prefix5(self):
        """5자리 prefix 동일, 전체 코드 다름 → dup_atc4 = 1, dup_atc5 = 0."""
        # A10BA02 vs A10BA03 — prefix 5자리 'A10BA' 동일
        rxs = [self._rx("A10BA02"), self._rx("A10BA03")]
        feat = self._make_feat()
        _fill_dup_features(feat, rxs, dup_groups=None)
        assert feat.dup_atc5 == 0, f"dup_atc5 should be 0, got {feat.dup_atc5}"
        assert feat.dup_atc4 == 1, f"dup_atc4 expected 1, got {feat.dup_atc4}"

    def test_dup_atc3_same_prefix4(self):
        """4자리 prefix 동일, 5자리 다름 → dup_atc3 = 1, dup_atc4 = 0."""
        # A10BA02 vs A10BB01 — prefix 4자리 'A10B' 동일, 5자리 다름
        rxs = [self._rx("A10BA02"), self._rx("A10BB01")]
        feat = self._make_feat()
        _fill_dup_features(feat, rxs, dup_groups=None)
        assert feat.dup_atc4 == 0, f"dup_atc4 should be 0, got {feat.dup_atc4}"
        assert feat.dup_atc3 == 1, f"dup_atc3 expected 1, got {feat.dup_atc3}"

    def test_completely_different_atc_no_dup(self):
        """ATC가 완전히 다른 두 약물 → 모든 dup_atcN = 0."""
        rxs = [self._rx("A10BA02"), self._rx("C09AA01")]
        feat = self._make_feat()
        _fill_dup_features(feat, rxs, dup_groups=None)
        assert feat.dup_atc5 == 0
        assert feat.dup_atc4 == 0
        assert feat.dup_atc3 == 0

    def test_no_label_crossover(self):
        """4단계 중복이 dup_atc5로 잘못 올라가지 않음 (레이블 오기입 회귀)."""
        # A10BA02 vs A10BA03 — ATC 4단계 중복, 5단계는 아님
        rxs = [self._rx("A10BA02"), self._rx("A10BA03")]
        feat = self._make_feat()
        _fill_dup_features(feat, rxs, dup_groups=None)
        # 이전 버그: dup_atc5 = 1 (cnt4 결과가 dup_atc5에 잘못 할당됨)
        assert feat.dup_atc5 == 0, "회귀: ATC 4단계 중복이 dup_atc5에 잘못 할당됨"
        assert feat.dup_atc4 == 1


# ─── _assign_risk_level 리팩터 후 라벨 동일성 회귀 테스트 ──────────────────────

class TestAssignRiskLevelBackwardCompat:
    """리팩터 전후 라벨 동일성 — 기존 elif cascade 규칙이 그대로 적용되는지."""

    def _make(self, **kwargs) -> PatientFeatures:
        base = dict(
            patient_id="P001",
            window_start=date(2026, 1, 1),
            window_end=date(2026, 3, 31),
        )
        base.update(kwargs)
        return PatientFeatures(**base)

    def test_contraindicated_red(self):
        f = self._make(ddi_contraindicated=1)
        _assign_risk_level(f)
        assert f.risk_level == "Red"
        assert any("Contraindicated" in r or "RED_CONTRAINDICATED" in r for r in f.risk_reasons)

    def test_major_ge_3_severe_yellow(self):
        """재설계: major≥3 → Red 아닌 Yellow(Y_TRIPLE 즉시개입)."""
        f = self._make(ddi_major=3)
        _assign_risk_level(f)
        assert f.risk_level == "Yellow"

    def test_triple_whammy_severe_yellow(self):
        f = self._make(triple_whammy=True)
        _assign_risk_level(f)
        assert f.risk_level == "Yellow"

    def test_major_1_yellow(self):
        f = self._make(ddi_major=1)
        _assign_risk_level(f)
        assert f.risk_level == "Yellow"

    def test_moderate_2_yellow(self):
        f = self._make(ddi_moderate=2)
        _assign_risk_level(f)
        assert f.risk_level == "Yellow"

    def test_dup_yellow(self):
        f = self._make(dup_same_ingredient=1)
        _assign_risk_level(f)
        assert f.risk_level == "Yellow"

    def test_institution_ge_3_yellow(self):
        f = self._make(institution_count=3)
        _assign_risk_level(f)
        assert f.risk_level == "Yellow"

    def test_minor_green(self):
        f = self._make(ddi_minor=1)
        _assign_risk_level(f)
        assert f.risk_level == "Green"

    def test_5drug_green(self):
        f = self._make(drug_count=5)
        _assign_risk_level(f)
        assert f.risk_level == "Green"

    def test_normal(self):
        f = self._make()
        _assign_risk_level(f)
        assert f.risk_level == "Normal"

    def test_red_takes_priority_over_yellow(self):
        """Red + Yellow trigger 동시 존재 시 Red 우선 (기존 elif cascade 동작 보존)."""
        f = self._make(ddi_contraindicated=1, ddi_major=1, dup_same_ingredient=1)
        _assign_risk_level(f)
        assert f.risk_level == "Red"
