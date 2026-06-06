"""Phase 2-2 모델 피처 게이팅: rule_features_active 시 triple_whammy/위험플래그를
edi→wk→components 로 산출, 아니면 0(구 번들/atc 경로). 구 번들과 skew 방지 게이팅 검증.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from serving.predictor import RequestFeatureBuilder
from serving.schemas import DrugItem, PredictRequest


class _DM:
    def __init__(self, wk2comp):
        self._m = wk2comp

    def get_components(self, wk):
        return list(self._m.get(wk, []))

    def expand_drug_count(self, wks):
        out = set()
        for wk in wks:
            out.update(self.get_components(wk) or [wk])
        return out

    def get_ddi_ids(self, wk):
        return []


class _Std:
    def __init__(self, edi2wk, dm):
        self._edi2wk = edi2wk
        self._dm = dm

    def get_wk(self, edi):
        return self._edi2wk.get(str(edi))

    def get_efmdc(self, edi):
        return None

    def lookup_edi(self, edi):
        return (None, None)

    @property
    def drug_master(self):
        return self._dm


def _builder():
    dm = _DM({"WA": ["enalapril"], "WK": ["spironolactone"], "WN": ["ibuprofen"]})
    std = _Std({"EA": "WA", "EK": "WK", "EN": "WN"}, dm)
    return RequestFeatureBuilder(ddi_matrix=None, code_standardizer=std)


_REQ = PredictRequest(patient_id="P", patient_age=50, drugs=[
    DrugItem(edi_code="EA", total_days=30, start_date=date(2024, 7, 1)),
    DrugItem(edi_code="EK", total_days=30, start_date=date(2024, 7, 1)),
    DrugItem(edi_code="EN", total_days=30, start_date=date(2024, 7, 1)),
])


def test_active_computes_triple_whammy_and_risk_via_components():
    _, feat = _builder().build(_REQ, rule_features_active=True)
    assert feat["triple_whammy"] == 1.0            # ACEi+K이뇨제+NSAID 동시
    assert feat["has_renal_risk_drug"] == 1.0      # ibuprofen
    assert feat["has_high_risk_drug"] in (0.0, 1.0)  # 키워드 의존(존재만 확인)


def test_inactive_keeps_zero_matches_old_bundle():
    """비활성(구 번들) — atc/name 경로(실 edi 미해석) → 0 으로 구 번들 학습값과 정합."""
    _, feat = _builder().build(_REQ, rule_features_active=False)
    assert feat["triple_whammy"] == 0.0
    assert feat["has_high_risk_drug"] == 0.0
    assert feat["has_renal_risk_drug"] == 0.0
    assert feat["has_hepatic_risk_drug"] == 0.0


def test_default_is_inactive():
    """rule_features_active 미지정 → 기본 비활성(구 번들 안전)."""
    _, feat = _builder().build(_REQ)
    assert feat["triple_whammy"] == 0.0
