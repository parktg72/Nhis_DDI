"""결정적 Red 백스톱 불변량 (Stage1 threshold 재검토 후속, Phase 2-2 = 전체 룰).

predict() 가 RequestFeatureBuilder.red_triggers(collect_red_triggers 를 edi→wk 피처에 적용)
결과로 final_level 을 단방향 escalation. ML Stage1(룰파생 degenerate τ, ~10% 누락)·
SafetyNet(약물명 미해석) 의존 제거. red_triggers 산출은 edi→wk→DrugMaster(학습 동일).
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


def _predictor(red_triggers_ret: set) -> HybridPredictor:
    pred = HybridPredictor.__new__(HybridPredictor)
    pred._start_time = 0.0
    pred._ml = MagicMock(); pred._ml.loaded = False
    pred._ddi_matrix = None
    pred._cyp = None
    pred._std = None
    pred._builder = RequestFeatureBuilder(ddi_matrix=None, code_standardizer=None)
    # 결정적 Red 트리거 집합 주입 (edi→wk→collect_red_triggers 경로는 detect/etl 테스트가 검증)
    pred._builder.red_triggers = lambda drugs, ref, age=None: set(red_triggers_ret)
    pred._safety_net = None
    pred._dup_detector = None
    pred._ml_lock = threading.Lock()
    pred._hier_lock = threading.RLock()
    pred._hierarchical = None
    return pred


_DRUGS = [DrugItem(edi_code="660700010", total_days=30, start_date=date(2024, 7, 1)),
          DrugItem(edi_code="642902720", total_days=30, start_date=date(2024, 7, 1))]


def _predict(pred):
    req = PredictRequest(patient_id="P", patient_age=50, drugs=_DRUGS)
    with patch("serving.predictor._run_safety_net") as sn, \
         patch("serving.predictor._run_duplicate_detector") as dup:
        sn.return_value = (RiskLevel.NORMAL, [], [])   # SafetyNet 백스톱 무력화(실 edi 시나리오)
        dup.return_value = (0, [])
        return pred.predict(req)


# 활성 백스톱 = 금기(RED_CONTRAINDICATED)만 (2026-06-06 재설계: 나머지는 Y_TRIPLE 즉시개입).
def test_contraindicated_forces_red():
    """금기(RED_CONTRAINDICATED)는 ML/SafetyNet 무관하게 RED + 사유 노출."""
    res = _predict(_predictor({"RED_CONTRAINDICATED"}))
    assert res.risk_level == RiskLevel.RED
    assert "RED_CONTRAINDICATED" in res.risk_reasons


def test_no_trigger_no_forced_red():
    res = _predict(_predictor(set()))
    assert res.risk_level != RiskLevel.RED


def test_backstop_is_one_way_escalation():
    """ML 이 NORMAL 이어도 금기 트리거가 RED 로 상향(단방향)."""
    pred = _predictor({"RED_CONTRAINDICATED"})
    pred._ml.loaded = True
    pred._ml.predict_proba = MagicMock(return_value=0.0)
    pred._ml.classify = MagicMock(return_value=RiskLevel.NORMAL)
    res = _predict(pred)
    assert res.risk_level == RiskLevel.RED
