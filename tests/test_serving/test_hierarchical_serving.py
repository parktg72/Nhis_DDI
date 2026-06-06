"""serving 계층 모드 테스트 — HierarchicalPredictor 경로.

HybridPredictor 가 HIERARCHICAL_MODEL_DIR 환경에서 predict_risk() 를 호출해
PredictResponse 의 yellow_subtype / stage2_probs / red_suspect / action 확장 필드를
올바르게 채우는지 검증. 기존 75 테스트는 backward compat 으로 별도 검증.
"""
from __future__ import annotations

import threading
from datetime import date
from unittest.mock import MagicMock

import numpy as np
import pytest

from hana_app.core.hierarchical_runner import STAGE2_LABELS
from serving.predictor import (
    HierarchicalPredictor,
    HybridPredictor,
    RequestFeatureBuilder,
)
from serving.schemas import DrugItem, PredictRequest, RiskLevel


# ─────────────────────────────────────────────────────────────────────────────
# 픽스처
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def req_normal():
    return PredictRequest(
        patient_id="P001",
        drugs=[
            DrugItem(edi_code="B001", atc_code="A10BA02", drug_name="metformin",
                     total_days=30, start_date=date(2024, 1, 1)),
        ],
    )


def _make_hierarchical_predictor(
    p_red: float,
    stage2_probs_local: np.ndarray,
    tau_red: float = 0.70,
    tau_review: float = 0.30,
    classes_present: list[int] | None = None,
) -> HierarchicalPredictor:
    """stage1/stage2 모델을 MagicMock 으로 대체한 HierarchicalPredictor 인스턴스.

    predict_risk 가 받을 때 stage1.predict_proba → [[1-p_red, p_red]],
    stage2.predict_proba → stage2_probs_local (shape (1, k)) 로 강제.
    """
    if classes_present is None:
        # 기본: 모든 6-class 포함
        classes_present = list(range(len(STAGE2_LABELS)))

    stage1 = MagicMock()
    stage1.predict_proba = MagicMock(
        return_value=np.array([[1 - p_red, p_red]])
    )

    stage2 = MagicMock()
    local_probs = np.asarray(stage2_probs_local).reshape(1, -1)
    stage2.predict_proba = MagicMock(return_value=local_probs)

    encoder = MagicMock()
    encoder.classes_ = np.array([STAGE2_LABELS[i] for i in classes_present])

    hp = HierarchicalPredictor()
    hp._stage1 = stage1
    hp._stage2 = stage2
    hp._encoder = encoder
    hp._classes_present = classes_present
    hp._thresholds = {"tau_red": tau_red, "tau_review": tau_review}
    hp._feature_cols = ["drug_count", "age"]
    hp._meta = {"feature_cols": hp._feature_cols, "thresholds": hp._thresholds}
    return hp


def _make_hybrid_with_hierarchical(hp: HierarchicalPredictor) -> HybridPredictor:
    """HybridPredictor 싱글턴을 건너뛰고 hierarchical 만 연결된 인스턴스 생성."""
    pred = HybridPredictor.__new__(HybridPredictor)
    pred._start_time = 0.0
    pred._ml_lock = threading.RLock()
    pred._hier_lock = threading.RLock()
    pred._ml = MagicMock()
    pred._ml.loaded = False
    pred._hierarchical = hp
    pred._ddi_matrix = None
    pred._cyp = None
    pred._std = None
    pred._safety_net = None
    pred._dup_detector = None
    pred._builder = RequestFeatureBuilder(
        ddi_matrix=None, cyp_extractor=None, code_standardizer=None
    )
    return pred


# ─────────────────────────────────────────────────────────────────────────────
# 4 분기 테스트
# ─────────────────────────────────────────────────────────────────────────────

def test_hierarchical_red_confirmed(req_normal):
    """p_red ≥ τ_red → Red 확정, Stage 2 skip, stage2_probs=None."""
    hp = _make_hierarchical_predictor(
        p_red=0.95,
        stage2_probs_local=np.array([0.2, 0.2, 0.15, 0.15, 0.1, 0.1, 0.1]),
    )
    pred = _make_hybrid_with_hierarchical(hp)
    resp = pred.predict(req_normal)

    assert resp.risk_level == RiskLevel.RED
    assert resp.ml_level == RiskLevel.RED
    assert resp.ml_probability == pytest.approx(0.95)
    assert resp.yellow_subtype is None
    assert resp.stage2_probs is None
    assert resp.red_suspect is False
    assert resp.action == "즉각 개입"


def test_hierarchical_red_suspect_band(req_normal):
    """τ_review ≤ p_red < τ_red → Stage 2 라벨 + red_suspect=True."""
    hp = _make_hierarchical_predictor(
        p_red=0.50,  # 0.30 ≤ 0.50 < 0.70
        # Y_TRIPLE 최고 확률
        stage2_probs_local=np.array([0.6, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05]),
    )
    pred = _make_hybrid_with_hierarchical(hp)
    resp = pred.predict(req_normal)

    assert resp.yellow_subtype == "Y_TRIPLE"
    assert resp.red_suspect is True
    assert resp.stage2_probs is not None
    assert set(resp.stage2_probs.keys()) == set(STAGE2_LABELS)
    assert resp.action == "문자 안내"   # Y_TRIPLE → 문자 안내(2026-06-07 위계 재설계)
    # 운영팀 검수 큐 reason 포함
    assert any("Red 의심" in r for r in resp.risk_reasons)


def test_hierarchical_yellow_subtype_clean(req_normal):
    """p_red < τ_review + Yellow subtype 선택 → yellow_subtype 매핑."""
    hp = _make_hierarchical_predictor(
        p_red=0.10,  # 0.10 < 0.30 (τ_review)
        # Y_DDI_MOD 최고 확률 (STAGE2_LABELS index 3)
        stage2_probs_local=np.array([0.1, 0.1, 0.1, 0.5, 0.05, 0.05, 0.1]),
    )
    pred = _make_hybrid_with_hierarchical(hp)
    resp = pred.predict(req_normal)

    assert resp.yellow_subtype == "Y_DDI_MOD"  # 단일 중등도DDI
    assert resp.red_suspect is False
    assert resp.ml_level == RiskLevel.YELLOW
    assert resp.action == "모니터링"   # Y_DDI_MOD 단일차원 → 모니터링(2026-06-07)
    assert resp.ml_probability == pytest.approx(0.10)


def test_hierarchical_no_alert(req_normal):
    """p_red < τ_review + No_Alert 선택 → risk_level=Normal (rule 도 Normal 전제)."""
    hp = _make_hierarchical_predictor(
        p_red=0.05,
        # No_Alert (index 6) 최고 확률
        stage2_probs_local=np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.70]),
    )
    pred = _make_hybrid_with_hierarchical(hp)
    resp = pred.predict(req_normal)

    assert resp.yellow_subtype is None
    assert resp.red_suspect is False
    assert resp.ml_level == RiskLevel.NORMAL
    assert resp.action == "관여 안 함"   # No_Alert(2026-06-07)
    assert resp.stage2_probs is not None  # No_Alert 도 분포 유지


# ─────────────────────────────────────────────────────────────────────────────
# HierarchicalPredictor 로더 자체 테스트 (실제 파일 저장 후 재로드)
# ─────────────────────────────────────────────────────────────────────────────

def test_hierarchical_predictor_load_real_artifact(tmp_path):
    """train_hierarchical 이 저장한 실제 아티팩트로 HierarchicalPredictor.load() 검증."""
    import pandas as pd
    from hana_app.core.hierarchical_runner import train_hierarchical

    rng = np.random.default_rng(42)
    n = 500
    df = pd.DataFrame({
        "patient_id": [f"P{i}" for i in range(n)],
        "drug_count": rng.integers(1, 12, size=n),
        "age": rng.integers(45, 90, size=n),
        "sex_m": rng.integers(0, 2, size=n),
        "risk_level": (["Red"] * 25 + ["Yellow"] * 100
                       + ["Green"] * 150 + ["Normal"] * 225),
        "yellow_subtype": (
            [None] * 25
            + ["Y_TRIPLE"] * 10 + ["Y_DDI_MAJOR"] * 15 + ["Y_DDI_MOD"] * 30
            + ["Y_DUP"] * 25 + ["Y_FRAG"] * 20
            + [None] * 375
        ),
    })
    train_hierarchical(
        df=df,
        feature_cols=["drug_count", "age", "sex_m"],
        output_dir=tmp_path,
        seed=42,
    )

    hp = HierarchicalPredictor()
    ok = hp.load(tmp_path)
    assert ok, "HierarchicalPredictor.load() 실패"
    assert hp.loaded
    assert hp.feature_cols == ["drug_count", "age", "sex_m"]
    assert "tau_red" in hp._thresholds
    assert "tau_review" in hp._thresholds
    assert hp._thresholds["tau_review"] < hp._thresholds["tau_red"]


def test_hierarchical_predictor_sha_mismatch_rejects(tmp_path):
    """stage_meta.json 의 stage1_sha256 변조 시 로드 거부."""
    import json
    import pandas as pd
    from hana_app.core.hierarchical_runner import train_hierarchical

    rng = np.random.default_rng(42)
    n = 500
    df = pd.DataFrame({
        "patient_id": [f"P{i}" for i in range(n)],
        "drug_count": rng.integers(1, 12, size=n),
        "age": rng.integers(45, 90, size=n),
        "sex_m": rng.integers(0, 2, size=n),
        "risk_level": (["Red"] * 25 + ["Yellow"] * 100
                       + ["Green"] * 150 + ["Normal"] * 225),
        "yellow_subtype": (
            [None] * 25
            + ["Y_TRIPLE"] * 10 + ["Y_DDI_MAJOR"] * 15 + ["Y_DDI_MOD"] * 30
            + ["Y_DUP"] * 25 + ["Y_FRAG"] * 20
            + [None] * 375
        ),
    })
    train_hierarchical(
        df=df,
        feature_cols=["drug_count", "age", "sex_m"],
        output_dir=tmp_path,
        seed=42,
    )

    meta_path = tmp_path / "stage_meta.json"
    meta = json.loads(meta_path.read_text())
    meta["stage1_sha256"] = "0" * 64  # 변조
    meta_path.write_text(json.dumps(meta))

    hp = HierarchicalPredictor()
    ok = hp.load(tmp_path)
    assert not ok, "변조된 해시도 통과됨 — 무결성 검증 실패"


def _train_7class_bundle(tmp_path, seed=42):
    """7-class 계층 번들을 tmp_path 에 학습·저장 (가드 테스트 픽스처)."""
    import pandas as pd
    from hana_app.core.hierarchical_runner import train_hierarchical
    rng = np.random.default_rng(seed)
    n = 500
    df = pd.DataFrame({
        "patient_id": [f"P{i}" for i in range(n)],
        "drug_count": rng.integers(1, 12, size=n),
        "age": rng.integers(45, 90, size=n),
        "sex_m": rng.integers(0, 2, size=n),
        "risk_level": (["Red"] * 25 + ["Yellow"] * 100
                       + ["Green"] * 150 + ["Normal"] * 225),
        "yellow_subtype": (
            [None] * 25
            + ["Y_TRIPLE"] * 10 + ["Y_DOUBLE"] * 15 + ["Y_DDI_MAJOR"] * 20
            + ["Y_DDI_MOD"] * 25 + ["Y_DUP"] * 15 + ["Y_FRAG"] * 15
            + [None] * 375
        ),
    })
    train_hierarchical(df=df, feature_cols=["drug_count", "age", "sex_m"],
                       output_dir=tmp_path, seed=seed)


def test_hierarchical_predictor_rejects_old_label_space(tmp_path):
    """구 6-class(Y_MIX) 번들은 현재 7-class STAGE2_LABELS 와 불일치 → 로드 거부.

    d201743 류 silent train/serve skew 방지 — meta stage2_labels 가 현재와 다르면 거부.
    """
    import json
    _train_7class_bundle(tmp_path)
    meta_path = tmp_path / "stage_meta.json"
    meta = json.loads(meta_path.read_text())
    # 구 6-class 라벨 공간으로 변조 (Y_MIX 포함, Y_TRIPLE/Y_DOUBLE 없음)
    meta["stage2_labels"] = ["Y_MIX", "Y_DDI_MAJOR", "Y_DDI_MOD", "Y_DUP", "Y_FRAG", "No_Alert"]
    meta_path.write_text(json.dumps(meta, ensure_ascii=False))

    hp = HierarchicalPredictor()
    assert hp.load(tmp_path) is False, "구 라벨 공간 번들이 로드됨 — 가드 실패"
    assert hp.loaded is False


def test_hierarchical_predictor_rejects_missing_stage2_labels(tmp_path):
    """stage2_labels 메타 누락 시 추측 대신 거부(재학습 필요)."""
    import json
    _train_7class_bundle(tmp_path)
    meta_path = tmp_path / "stage_meta.json"
    meta = json.loads(meta_path.read_text())
    meta.pop("stage2_labels", None)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False))

    hp = HierarchicalPredictor()
    assert hp.load(tmp_path) is False, "stage2_labels 누락 번들이 로드됨 — 가드 실패"


def test_hierarchical_predictor_loads_current_7class(tmp_path):
    """현재 7-class 라벨 공간 번들은 정상 로드(가드가 정상 번들을 막지 않음)."""
    _train_7class_bundle(tmp_path)
    hp = HierarchicalPredictor()
    assert hp.load(tmp_path) is True
    assert hp.loaded is True


def test_hierarchical_predictor_rejects_old_ddi_version(tmp_path):
    """Q5: 구 DDI 시맨틱(ddi=0/ATC)으로 학습된 번들은 로드 거부 (train/serve 스큐 방지)."""
    import json
    _train_7class_bundle(tmp_path)
    meta_path = tmp_path / "stage_meta.json"
    meta = json.loads(meta_path.read_text())
    assert meta.get("ddi_feature_semantics_version")  # train_hierarchical 이 스탬프
    meta["ddi_feature_semantics_version"] = "ddi.v1"   # 구버전으로 변조
    meta_path.write_text(json.dumps(meta, ensure_ascii=False))

    hp = HierarchicalPredictor()
    assert hp.load(tmp_path) is False, "구 DDI 시맨틱 번들이 로드됨 — 가드 실패"


def test_hierarchical_predictor_rejects_missing_ddi_version(tmp_path):
    """ddi_feature_semantics_version 누락 시 추측 대신 거부(재학습 필요)."""
    import json
    _train_7class_bundle(tmp_path)
    meta_path = tmp_path / "stage_meta.json"
    meta = json.loads(meta_path.read_text())
    meta.pop("ddi_feature_semantics_version", None)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False))

    hp = HierarchicalPredictor()
    assert hp.load(tmp_path) is False, "DDI 버전 누락 번들이 로드됨 — 가드 실패"


def test_hierarchical_predictor_missing_files(tmp_path):
    """stage1_red.joblib 없을 때 load() 는 False 반환."""
    # stage_meta.json 만 있고 나머지 없음
    (tmp_path / "stage_meta.json").write_text('{"thresholds": {"tau_red": 0.7, "tau_review": 0.3}, "feature_cols": []}')
    hp = HierarchicalPredictor()
    assert hp.load(tmp_path) is False


# ─────────────────────────────────────────────────────────────────────────────
# HTTP 엔드투엔드 — JSON 직렬화 round-trip
# ─────────────────────────────────────────────────────────────────────────────

def test_hierarchical_http_roundtrip_serializes_new_fields():
    """POST /predict → 응답 JSON 에 신규 필드(yellow_subtype, stage2_probs, red_suspect, action)
    직렬화 형식이 올바른지 (dict → JSON object, bool → true/false) 검증."""
    from fastapi.testclient import TestClient
    import serving.predictor as pred_module
    from serving.main import app

    hp = _make_hierarchical_predictor(
        p_red=0.50,  # Red 의심 구간
        stage2_probs_local=np.array([0.6, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05]),
    )
    pred = _make_hybrid_with_hierarchical(hp)

    # lifespan 진입 후 _predictor 를 주입해야 덮어쓰지 않음 (init_predictor 가 교체)
    with TestClient(app, raise_server_exceptions=False) as client:
        original = pred_module._predictor
        pred_module._predictor = pred
        try:
            resp = client.post("/predict", json={
                "patient_id": "P001",
                "drugs": [
                    {
                        "edi_code": "B001",
                        "atc_code": "A10BA02",
                        "drug_name": "metformin",
                        "total_days": 30,
                        "start_date": "2024-01-01",
                    }
                ],
            })
            assert resp.status_code == 200, resp.text
            body = resp.json()
            # 신규 필드 직렬화 검증
            assert body["yellow_subtype"] == "Y_TRIPLE"
            assert body["red_suspect"] is True
            assert body["action"] == "문자 안내"   # Y_TRIPLE → 문자 안내(2026-06-07)
            assert isinstance(body["stage2_probs"], dict)
            assert set(body["stage2_probs"].keys()) == set(STAGE2_LABELS)
            assert all(isinstance(v, float) for v in body["stage2_probs"].values())
        finally:
            pred_module._predictor = original
