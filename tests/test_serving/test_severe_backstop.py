"""결정적 Yellow subtype floor 백스톱 (model-independent).

major DDI(≥1)→Y_DDI_MAJOR(약사전화) > 중증(triple_whammy/10drug+고위험/고령+장기)→Y_TRIPLE
(문자안내). 학습 _assign_yellow_subtype 위계를 edi→wk 로 미러 → 모델이 하향분류해도 최소 보장.
Red·상위 subtype 은 유지(단방향).
"""
from __future__ import annotations

import sys
import threading
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from serving.predictor import HybridPredictor, RequestFeatureBuilder
from serving.schemas import DrugItem, PredictRequest, RiskLevel


def _predictor(model_subtype, model_action, *, floor=None, floor_reasons=frozenset(),
               red=frozenset()):
    pred = HybridPredictor.__new__(HybridPredictor)
    pred._start_time = 0.0
    pred._ml = MagicMock(); pred._ml.loaded = False
    pred._ddi_matrix = None
    pred._cyp = None
    pred._std = None
    pred._builder = RequestFeatureBuilder(ddi_matrix=None, code_standardizer=None)
    pred._builder.build = lambda req, **kw: (np.zeros(3), {})
    pred._builder.red_triggers = lambda drugs, ref, age=None: set(red)
    pred._builder.rule_floor = lambda drugs, ref, age=None: (floor, set(floor_reasons))
    pred._safety_net = None
    pred._dup_detector = None
    pred._ml_lock = threading.Lock()
    pred._hier_lock = threading.RLock()
    hier = MagicMock()
    hier.loaded = True
    hier.feature_cols = []
    hier.feature_semantics_version = "rulefeat.v1"
    hier.predict_risk_single = lambda fv: {
        "risk_level": model_subtype, "p_red": 0.1,
        "stage2_probs": None, "red_suspect": False, "action": model_action,
    }
    pred._hierarchical = hier
    return pred


_DRUGS = [DrugItem(edi_code="660700010", total_days=30, start_date=date(2024, 7, 1))]


def _predict(pred):
    req = PredictRequest(patient_id="P", patient_age=80, drugs=_DRUGS)
    with patch("serving.predictor._run_safety_net") as sn, \
         patch("serving.predictor._run_duplicate_detector") as dup:
        sn.return_value = (RiskLevel.NORMAL, [], [])
        dup.return_value = (0, [])
        return pred.predict(req)


def test_major_ddi_floors_to_ddi_major():
    """major DDI floor + 모델 Y_DOUBLE(모니터링) → Y_DDI_MAJOR(약사 전화)."""
    res = _predict(_predictor("Y_DOUBLE", "모니터링",
                              floor="Y_DDI_MAJOR", floor_reasons={"DDI_MAJOR"}))
    assert res.risk_level == RiskLevel.YELLOW
    assert res.yellow_subtype == "Y_DDI_MAJOR"
    assert res.action == "약사 전화"
    assert "DDI_MAJOR" in res.risk_reasons


def test_severe_floors_to_ytriple():
    """중증 floor + 모델 Y_DOUBLE → Y_TRIPLE(문자 안내)."""
    res = _predict(_predictor("Y_DOUBLE", "모니터링",
                              floor="Y_TRIPLE", floor_reasons={"SEV_TRIPLE_WHAMMY"}))
    assert res.yellow_subtype == "Y_TRIPLE"
    assert res.action == "문자 안내"


def test_floor_does_not_downgrade_higher_model_subtype():
    """floor Y_TRIPLE 인데 모델이 이미 Y_DDI_MAJOR(상위) → 유지(하향 안 함)."""
    res = _predict(_predictor("Y_DDI_MAJOR", "약사 전화",
                              floor="Y_TRIPLE", floor_reasons={"SEV_ELDERLY_ORGAN"}))
    assert res.yellow_subtype == "Y_DDI_MAJOR"
    assert res.action == "약사 전화"


def test_major_floor_over_none_model():
    res = _predict(_predictor("No_Alert", "관여 안 함",
                              floor="Y_DDI_MAJOR", floor_reasons={"DDI_MAJOR_3PLUS"}))
    assert res.risk_level == RiskLevel.YELLOW
    assert res.yellow_subtype == "Y_DDI_MAJOR"
    assert res.action == "약사 전화"


def test_no_floor_keeps_model_subtype():
    res = _predict(_predictor("Y_DOUBLE", "모니터링", floor=None))
    assert res.yellow_subtype == "Y_DOUBLE"
    assert res.action == "모니터링"


def test_red_takes_precedence_over_floor():
    """금기(Red) + floor → Red 유지(floor 는 Red 미관여)."""
    res = _predict(_predictor("Y_DOUBLE", "모니터링",
                              floor="Y_DDI_MAJOR", floor_reasons={"DDI_MAJOR"},
                              red={"RED_CONTRAINDICATED"}))
    assert res.risk_level == RiskLevel.RED
