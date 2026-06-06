"""detect_triple_whammy — ACEi/ARB + K이뇨제 + NSAID 성분 키워드 판정 (Phase 2).

학습(aggregate_patient_features)·서빙(향후 edi→wk 후)이 **동일 공용함수**를 호출하므로
parity by construction. 본 테스트는 detector 로직 + 실데이터 성분명 정합 검증.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.etl.prescription_aggregator import detect_triple_whammy


class _DM:
    """get_components(wk) → 성분명 리스트만 흉내내는 경량 DrugMaster."""
    def __init__(self, mapping: dict):
        self._m = mapping

    def get_components(self, wk):
        return list(self._m.get(wk, []))


def test_three_classes_true():
    dm = _DM({"A": ["enalapril"], "K": ["spironolactone"], "N": ["ibuprofen"]})
    assert detect_triple_whammy(["A", "K", "N"], dm) is True


def test_arb_sartan_suffix_true():
    dm = _DM({"A": ["losartan"], "K": ["amiloride"], "N": ["naproxen"]})
    assert detect_triple_whammy(["A", "K", "N"], dm) is True


@pytest.mark.parametrize("missing", ["A", "K", "N"])
def test_missing_one_class_false(missing):
    full = {"A": ["ramipril"], "K": ["eplerenone"], "N": ["diclofenac"]}
    full.pop(missing)
    assert detect_triple_whammy(list(full), dm := _DM(full)) is False


def test_combination_drug_two_classes_in_one_wk():
    """복합제 한 wk 가 ACEi+NSAID 동시 성분이어도 클래스 합산."""
    dm = _DM({"X": ["enalapril", "ibuprofen"], "K": ["spironolactone"]})
    assert detect_triple_whammy(["X", "K"], dm) is True


def test_no_drug_master_or_empty_false():
    assert detect_triple_whammy(["A"], None) is False
    assert detect_triple_whammy([], _DM({})) is False


def test_non_triple_whammy_drugs_false():
    """무관 약물(메트포민/메토프롤롤)은 False."""
    dm = _DM({"A": ["metformin"], "B": ["metoprolol"]})
    assert detect_triple_whammy(["A", "B"], dm) is False


@pytest.mark.skipif(
    not (ROOT / "data" / "processed" / "hira_drug_master.parquet").exists(),
    reason="DrugMaster 데이터 없음",
)
def test_real_drug_master_keyword_match():
    """실 DrugMaster 성분명 포맷에서 3클래스 대표약물이 키워드로 잡히는지 검증."""
    import hana_app.core.ml_runner as M
    M._DRUG_MASTER_CACHE.update({"obj": None, "loaded": False})
    dm = M._load_drug_master()
    comp2wk = {}
    for wk, comps in dm._code_to_components.items():
        for c in comps:
            comp2wk.setdefault(c.lower(), wk)
    a = next((comp2wk[c] for c in comp2wk if c.endswith(("pril", "sartan"))), None)
    k = next((comp2wk[c] for c in comp2wk if "spironolactone" in c), None)
    n = next((comp2wk[c] for c in comp2wk if "ibuprofen" in c or "naproxen" in c), None)
    if not (a and k and n):
        pytest.skip("실 데이터에 3클래스 대표약물 부재")
    assert detect_triple_whammy([a, k, n], dm) is True
    assert detect_triple_whammy([a, k], dm) is False     # NSAID 빠지면 False
