"""clinical_rules: Red/Yellow trigger 집합 수집 공용 모듈 테스트."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.etl.clinical_rules import (
    CLINICAL_STANDARDS_VERSION,
    collect_red_triggers,
    collect_severe_immediate_triggers,
    collect_yellow_triggers,
)


def _features(**kwargs):
    """테스트용 feature-like 객체 — PatientFeatures 와 동일 attribute."""
    base = dict(
        ddi_contraindicated=0, ddi_major=0, ddi_moderate=0, ddi_minor=0,
        triple_whammy=False, drug_count=0, has_high_risk_drug=False,
        has_renal_risk_drug=False, has_hepatic_risk_drug=False,
        dup_same_ingredient=0, institution_count=0, age=None,
    )
    base.update(kwargs)
    return SimpleNamespace(**base)


def test_version_constant():
    assert CLINICAL_STANDARDS_VERSION == "v1.0"


class TestCollectRedTriggers:
    def test_empty_on_normal(self):
        assert collect_red_triggers(_features()) == set()

    def test_contraindicated(self):
        assert collect_red_triggers(_features(ddi_contraindicated=1)) == {"RED_CONTRAINDICATED"}

    def test_demoted_triggers_not_red(self):
        """2026-06-06 재설계: major3/triple/10drug/elderly 는 Red 아님(금기만 Red)."""
        assert collect_red_triggers(_features(ddi_major=3)) == set()
        assert collect_red_triggers(_features(triple_whammy=True)) == set()
        assert collect_red_triggers(_features(drug_count=10, has_high_risk_drug=True)) == set()
        assert collect_red_triggers(_features(age=80, drug_count=6, has_renal_risk_drug=True)) == set()

    def test_only_contraindicated_is_red(self):
        trg = collect_red_triggers(_features(ddi_contraindicated=1, triple_whammy=True, ddi_major=3))
        assert trg == {"RED_CONTRAINDICATED"}  # 금기만 Red, 나머지는 severe(Y_TRIPLE)


class TestCollectSevereImmediateTriggers:
    """구 Red 트리거(금기 외) → 즉시개입 Y_TRIPLE 강제 조건."""
    def test_empty_on_normal(self):
        assert collect_severe_immediate_triggers(_features()) == set()

    def test_major_ge_3(self):
        assert collect_severe_immediate_triggers(_features(ddi_major=3)) == {"SEV_MAJOR_3PLUS"}
        assert collect_severe_immediate_triggers(_features(ddi_major=2)) == set()

    def test_triple_whammy(self):
        assert collect_severe_immediate_triggers(_features(triple_whammy=True)) == {"SEV_TRIPLE_WHAMMY"}

    def test_10drug_high_risk(self):
        assert collect_severe_immediate_triggers(_features(drug_count=10, has_high_risk_drug=True)) == {"SEV_10DRUG_HIGHRISK"}
        assert collect_severe_immediate_triggers(_features(drug_count=10, has_high_risk_drug=False)) == set()

    def test_elderly_organ(self):
        assert collect_severe_immediate_triggers(_features(age=75, drug_count=5, has_renal_risk_drug=True)) == {"SEV_ELDERLY_ORGAN"}
        assert collect_severe_immediate_triggers(_features(age=74, drug_count=5, has_renal_risk_drug=True)) == set()
        assert collect_severe_immediate_triggers(_features(age=80, drug_count=6, has_hepatic_risk_drug=True)) == {"SEV_ELDERLY_ORGAN"}

    def test_age_none_no_elderly(self):
        assert "SEV_ELDERLY_ORGAN" not in collect_severe_immediate_triggers(
            _features(age=None, drug_count=5, has_renal_risk_drug=True))

    def test_drug_count_9_not_triggered(self):
        assert "SEV_10DRUG_HIGHRISK" not in collect_severe_immediate_triggers(
            _features(drug_count=9, has_high_risk_drug=True))


class TestCollectYellowTriggers:
    def test_empty_on_normal(self):
        assert collect_yellow_triggers(_features()) == set()

    def test_ddi_major_single_or_double(self):
        assert collect_yellow_triggers(_features(ddi_major=1)) == {"DDI_MAJOR"}
        assert collect_yellow_triggers(_features(ddi_major=2)) == {"DDI_MAJOR"}

    def test_ddi_moderate_ge_2(self):
        assert collect_yellow_triggers(_features(ddi_moderate=2)) == {"DDI_MOD"}
        assert collect_yellow_triggers(_features(ddi_moderate=1)) == set()

    def test_dup_same_ingredient(self):
        assert collect_yellow_triggers(_features(dup_same_ingredient=1)) == {"DUP"}

    def test_institution_count_ge_3(self):
        assert collect_yellow_triggers(_features(institution_count=3)) == {"FRAG"}
        assert collect_yellow_triggers(_features(institution_count=2)) == set()

    def test_multiple_yellow_triggers(self):
        trg = collect_yellow_triggers(_features(ddi_major=1, dup_same_ingredient=1))
        assert trg == {"DDI_MAJOR", "DUP"}

    def test_three_triggers_all_returned(self):
        """3개 이상 Yellow trigger 가 동시에 있어도 모두 반환 (short-circuit 방지)."""
        trg = collect_yellow_triggers(_features(
            ddi_major=1, ddi_moderate=2, dup_same_ingredient=1, institution_count=3,
        ))
        assert trg == {"DDI_MAJOR", "DDI_MOD", "DUP", "FRAG"}
