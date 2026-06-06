"""_check_risk_drugs 성분 키워드 경로 (Phase 2-3, has_high_risk/renal/hepatic).

학습 records 는 drug_name/atc_code 가 없어(df_row_to_record) 기존 이름·ATC 경로는 dead.
→ DrugMaster.get_components(wk) 성분명 키워드 매칭으로 활성화. 키워드는 기존
risk_drug_constants 단일출처(새 정의 아님 — 식별자만 수정). 학습·서빙(향후 edi→wk) 공용.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.etl.models import PrescriptionRecord
from scripts.etl.prescription_aggregator import (
    _check_risk_drugs,
    _fill_risk_drug_flags,
    _HIGH_RISK_KEYWORDS,
    _RENAL_RISK_KEYWORDS,
    _HEPATIC_RISK_KEYWORDS,
)


class _DM:
    def __init__(self, mapping: dict):
        self._m = mapping

    def get_components(self, wk):
        return list(self._m.get(wk, []))


def _rec(wk):
    return PrescriptionRecord(patient_id="P", institution_id="I", bill_no="B",
                              wk_compn_cd=wk, start_date=date(2024, 7, 1),
                              end_date=date(2024, 7, 30), total_days=30, source="T30")


def test_component_keyword_detects_high_risk():
    # digoxin = HIGH_RISK 키워드, records 엔 이름/ATC 없음 → 성분 경로로만 탐지
    dm = _DM({"W1": ["digoxin"]})
    assert _check_risk_drugs([_rec("W1")], _HIGH_RISK_KEYWORDS, (), dm) is True


def test_renal_and_hepatic_via_components():
    dm = _DM({"R": ["ibuprofen"], "H": ["acetaminophen"]})
    assert _check_risk_drugs([_rec("R")], _RENAL_RISK_KEYWORDS, (), dm) is True
    assert _check_risk_drugs([_rec("H")], _HEPATIC_RISK_KEYWORDS, (), dm) is True


def test_no_match_false():
    dm = _DM({"X": ["metformin"]})  # 위험 키워드 아님
    assert _check_risk_drugs([_rec("X")], _HIGH_RISK_KEYWORDS, (), dm) is False


def test_no_drug_master_records_have_no_name_atc_false():
    """drug_master 없고 record 에 name/atc 없으면 (학습 dead 경로) False."""
    assert _check_risk_drugs([_rec("W1")], _HIGH_RISK_KEYWORDS, (), None) is False


def test_name_path_still_works_when_present():
    """drug_name 이 있으면(예: 서빙) 이름 경로도 유효."""
    r = _rec("W1"); r.drug_name = "Digoxin 0.25mg"
    assert _check_risk_drugs([r], _HIGH_RISK_KEYWORDS, (), None) is True


def test_fill_sets_three_flags_via_components():
    from scripts.etl.models import PatientFeatures
    dm = _DM({"hi": ["warfarin"], "re": ["gentamicin"], "he": ["isoniazid"]})
    feat = PatientFeatures(patient_id="P", window_start=date(2024, 7, 1),
                           window_end=date(2024, 7, 30))
    _fill_risk_drug_flags(feat, [_rec("hi"), _rec("re"), _rec("he")], dm)
    assert feat.has_high_risk_drug is True
    assert feat.has_renal_risk_drug is True
    assert feat.has_hepatic_risk_drug is True


@pytest.mark.skipif(
    not (ROOT / "data" / "processed" / "hira_drug_master.parquet").exists(),
    reason="DrugMaster 데이터 없음",
)
def test_real_drug_master_component_keyword():
    import hana_app.core.ml_runner as M
    M._DRUG_MASTER_CACHE.update({"obj": None, "loaded": False})
    dm = M._load_drug_master()
    comp2wk = {}
    for wk, comps in dm._code_to_components.items():
        for c in comps:
            comp2wk.setdefault(c.lower(), wk)
    # HIGH_RISK 키워드 중 실 데이터에 존재하는 첫 약물로 검증
    wk = next((comp2wk[c] for c in comp2wk
               if any(k in c for k in _HIGH_RISK_KEYWORDS)), None)
    if wk is None:
        pytest.skip("실 데이터에 high-risk 성분 부재")
    assert _check_risk_drugs([_rec(wk)], _HIGH_RISK_KEYWORDS, (), dm) is True
