"""결정적 DDI Red 백스톱 불변량 (Stage1 threshold 재검토 후속, cross-family MVP).

학습 collect_red_triggers 의 DDI 조건(ddi_contraindicated≥1 / ddi_major≥3)을 서빙이
edi→wk DDI 카운트(Task B)에 결정적으로 적용 → ML Stage1(룰 파생 라벨 degenerate τ_red
≈0.9998, recall 0.90 = ~10% 누락)·SafetyNet(약물명 기반, 실 edi 미해석) 의존 제거.
단방향 escalation(ML/Rule 이 Red 를 내리지 못함).
"""
from __future__ import annotations

import sys
import threading
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from serving.predictor import HybridPredictor, RequestFeatureBuilder
from serving.schemas import DrugItem, PredictRequest, RiskLevel


def _predictor(count_ddi_ret: dict) -> HybridPredictor:
    pred = HybridPredictor.__new__(HybridPredictor)
    pred._start_time = 0.0
    pred._ml = MagicMock(); pred._ml.loaded = False
    pred._ddi_matrix = None
    pred._cyp = None
    pred._std = None
    pred._builder = RequestFeatureBuilder(ddi_matrix=None, code_standardizer=None)
    pred._builder._count_ddi = lambda drugs, ref: count_ddi_ret  # 결정적 카운트 주입
    pred._safety_net = None
    pred._dup_detector = None
    pred._ml_lock = threading.Lock()
    pred._hier_lock = threading.RLock()
    pred._hierarchical = None
    return pred


_DRUGS = [DrugItem(edi_code="660700010", total_days=30, start_date=date(2024, 7, 1)),
          DrugItem(edi_code="642902720", total_days=30, start_date=date(2024, 7, 1))]
_ZERO = {"Contraindicated": 0, "Major": 0, "Moderate": 0, "Minor": 0}


def _predict(pred):
    req = PredictRequest(patient_id="P", drugs=_DRUGS)
    with patch("serving.predictor._run_safety_net") as sn, \
         patch("serving.predictor._run_duplicate_detector") as dup:
        sn.return_value = (RiskLevel.NORMAL, [], [])   # SafetyNet 백스톱 무력화(실 edi 시나리오)
        dup.return_value = (0, [])
        return pred.predict(req)


def test_contraindicated_forces_red():
    """ddi_contraindicated≥1 → ML/SafetyNet 무관하게 RED + RED_CONTRAINDICATED."""
    res = _predict(_predictor({**_ZERO, "Contraindicated": 1}))
    assert res.risk_level == RiskLevel.RED
    assert "RED_CONTRAINDICATED" in res.risk_reasons


def test_major3_forces_red():
    """ddi_major≥3 → RED + RED_MAJOR_3PLUS."""
    res = _predict(_predictor({**_ZERO, "Major": 3}))
    assert res.risk_level == RiskLevel.RED
    assert "RED_MAJOR_3PLUS" in res.risk_reasons


def test_major2_does_not_force_red():
    """ddi_major=2 (<3) → 강제 Red 아님 (학습 룰과 동일 임계)."""
    res = _predict(_predictor({**_ZERO, "Major": 2}))
    assert res.risk_level != RiskLevel.RED
    assert "RED_MAJOR_3PLUS" not in res.risk_reasons


def test_no_ddi_no_forced_red():
    """DDI 0 → 백스톱 미발동(SafetyNet/ML NORMAL → NORMAL)."""
    res = _predict(_predictor(dict(_ZERO)))
    assert res.risk_level != RiskLevel.RED


def test_backstop_is_one_way_escalation():
    """ML/Rule 이 NORMAL 이어도 결정적 DDI Red 가 RED 로 상향(단방향)."""
    pred = _predictor({**_ZERO, "Contraindicated": 1})
    pred._ml.loaded = True
    pred._ml.predict_proba = MagicMock(return_value=0.0)
    pred._ml.classify = MagicMock(return_value=RiskLevel.NORMAL)
    res = _predict(pred)
    assert res.risk_level == RiskLevel.RED


@pytest.mark.xfail(reason="triple_whammy 는 학습(prescription_aggregator)서 미산출(항상 0) — "
                          "Phase 2(ETL 계산 추가+재학습) 후 결정적 Red 활성화", strict=False)
def test_triple_whammy_deterministic_red_pending_phase2():
    """triple_whammy 결정적 Red 는 학습 parity 확보(Phase 2) 전까지 미적용."""
    # MVP 는 DDI 두 트리거만 — triple_whammy 백스톱은 의도적으로 제외.
    res = _predict(_predictor(dict(_ZERO)))  # triple_whammy 신호 없음
    assert res.risk_level == RiskLevel.RED  # Phase 2 전엔 실패(xfail)
