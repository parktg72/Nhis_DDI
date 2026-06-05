"""단일 ML / 계층 모델 feature schema strict validation — Codex 2026-05-07 P1.

배경: serving/predictor.py:664 의 `feat.get(name, 0.0)` fallback 이 학습 모델의
feature_names 중 RequestFeatureBuilder 미산출 컬럼을 silent 0.0 으로 채워 학습-
서빙 drift 가 *정상 예측처럼* 보이는 위험 (cross-family Codex/opencode/hermes
공동 P1 지적).

본 테스트는:
  - default strict: missing 컬럼 → 모델 로드 거부
  - FEATURE_SCHEMA_LENIENT=1: warning + 로드 성공 (legacy 호환 sunset 윈도우)
  - dup_efmdc (intentional allowlist) → strict 에서도 통과
  - 계층 모델 init / reload 도 동일 가드 적용
"""
from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from serving.predictor import (
    MLModel,
    HybridPredictor,
    HierarchicalPredictor,
    _FEATURE_ALLOWED,
    _BUILDER_KNOWN_COLS,
    _INTENTIONAL_FEATURE_ALLOWLIST,
    _validate_feature_schema,
)


class _FakeSklearnModel:
    def predict_proba(self, X):
        prob = np.full(len(X), 0.5)
        return np.column_stack([1 - prob, prob])


def _write_model(path: Path, feature_names: list[str]) -> Path:
    payload = {
        "model": _FakeSklearnModel(),
        "best_threshold": 0.5,
        "trainer_class": "XGBoostTrainer",
        "feature_names": feature_names,
        "artifact_version": 2,
    }
    content = pickle.dumps(payload)
    path.write_bytes(content)
    sha = hashlib.sha256(content).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(f"{sha}  {path.name}\n")
    return path


# ─── _validate_feature_schema 단위 ────────────────────────────────────────────

def test_allowlist_includes_dup_efmdc():
    """dup_efmdc 는 의도적 allowlist (serving 0.0 고정) — _FEATURE_ALLOWED 에 포함."""
    assert "dup_efmdc" in _INTENTIONAL_FEATURE_ALLOWLIST
    assert "dup_efmdc" in _FEATURE_ALLOWED


def test_validate_empty_passes():
    missing, ok = _validate_feature_schema([], "test")
    assert missing == [] and ok is True


def test_validate_all_known_passes():
    feats = sorted(_BUILDER_KNOWN_COLS)[:5]
    missing, ok = _validate_feature_schema(feats, "test")
    assert missing == [] and ok is True


def test_validate_dup_efmdc_passes():
    """dup_efmdc 는 builder 미산출이지만 allowlist 라 strict 통과."""
    missing, ok = _validate_feature_schema(["drug_count", "dup_efmdc"], "test")
    assert missing == [] and ok is True


def test_validate_unknown_fails_strict(monkeypatch):
    monkeypatch.delenv("FEATURE_SCHEMA_LENIENT", raising=False)
    missing, ok = _validate_feature_schema(
        ["drug_count", "fake_feature_xyz"], "test",
    )
    assert ok is False
    assert "fake_feature_xyz" in missing


def test_validate_unknown_passes_lenient(monkeypatch):
    monkeypatch.setenv("FEATURE_SCHEMA_LENIENT", "1")
    missing, ok = _validate_feature_schema(
        ["drug_count", "fake_feature_xyz"], "test",
    )
    assert ok is True
    assert "fake_feature_xyz" in missing


# ─── MLModel.load 통합 ────────────────────────────────────────────────────────

def test_mlmodel_load_rejects_unknown_feature_strict(tmp_path, monkeypatch):
    monkeypatch.delenv("FEATURE_SCHEMA_LENIENT", raising=False)
    path = _write_model(tmp_path / "m.pkl", ["drug_count", "fake_xyz"])
    ml = MLModel()
    assert ml.load(path) is False, (
        "strict 모드에서 unknown feature 모델은 로드 거부되어야 함 — silent 0.0 drift 방지"
    )
    # 부분 적용 상태 정리 검증
    assert ml._model is None
    assert ml._feature_names == []


def test_mlmodel_load_accepts_unknown_lenient(tmp_path, monkeypatch):
    monkeypatch.setenv("FEATURE_SCHEMA_LENIENT", "1")
    path = _write_model(tmp_path / "m.pkl", ["drug_count", "fake_xyz"])
    ml = MLModel()
    assert ml.load(path) is True
    assert ml._schema_drift == ["fake_xyz"]


def test_mlmodel_load_accepts_dup_efmdc(tmp_path, monkeypatch):
    """dup_efmdc 만 추가된 모델은 strict 에서도 로드 성공."""
    monkeypatch.delenv("FEATURE_SCHEMA_LENIENT", raising=False)
    path = _write_model(tmp_path / "m.pkl", ["drug_count", "age", "dup_efmdc"])
    ml = MLModel()
    assert ml.load(path) is True
    assert ml._schema_drift == []


# ─── reload_hierarchical 통합 ─────────────────────────────────────────────────

def _make_hier_predictor() -> HybridPredictor:
    """단순한 HybridPredictor 인스턴스 — reload_hierarchical 테스트용."""
    import threading
    pred = HybridPredictor.__new__(HybridPredictor)
    pred._start_time = 0.0
    pred._ml = MLModel()
    pred._ddi_matrix = None
    pred._cyp = None
    pred._std = None
    pred._safety_net = None
    pred._dup_detector = None
    pred._ml_lock = threading.Lock()
    pred._hier_lock = threading.RLock()
    pred._hierarchical = None
    return pred


def test_reload_hierarchical_rejects_unknown_strict(monkeypatch):
    """계층 모델 reload 도 schema strict — unknown 컬럼 → 로드 거부."""
    monkeypatch.delenv("FEATURE_SCHEMA_LENIENT", raising=False)
    pred = _make_hier_predictor()

    fake_hp = MagicMock()
    fake_hp.load = MagicMock(return_value=True)
    fake_hp.feature_cols = ["drug_count", "fake_unknown_col"]

    import serving.predictor as P
    monkeypatch.setattr(P, "HierarchicalPredictor", MagicMock(return_value=fake_hp))

    ok = pred.reload_hierarchical("/tmp/fake")
    assert ok is False, "schema 거부 시 reload 는 False"
    assert pred._hierarchical is None, "schema 거부 시 기존 _hierarchical 변경 X"


def test_reload_hierarchical_accepts_known(monkeypatch):
    """알려진 컬럼만 → reload 성공."""
    monkeypatch.delenv("FEATURE_SCHEMA_LENIENT", raising=False)
    pred = _make_hier_predictor()

    fake_hp = MagicMock()
    fake_hp.load = MagicMock(return_value=True)
    fake_hp.feature_cols = sorted(_BUILDER_KNOWN_COLS)[:5]

    import serving.predictor as P
    monkeypatch.setattr(P, "HierarchicalPredictor", MagicMock(return_value=fake_hp))

    ok = pred.reload_hierarchical("/tmp/fake")
    assert ok is True
    assert pred._hierarchical is fake_hp


def _write_hier_artifact(root: Path, feature_cols: list[str], *, corrupt_stage1_hash: bool = False) -> Path:
    import json
    import joblib
    import types
    import numpy as np
    from hana_app.core.hierarchical_runner import STAGE2_LABELS
    from scripts.etl.prescription_aggregator import DDI_FEATURE_SEMANTICS_VERSION

    root.mkdir(parents=True, exist_ok=True)
    stage1_path = root / "stage1_red.joblib"
    stage2_path = root / "stage2_yellow.joblib"
    joblib.dump(_FakeSklearnModel(), stage1_path)
    # 인코더는 학습 관례대로 full STAGE2_LABELS 를 classes_ 로 가진다(라벨 공간 가드 통과용).
    _encoder = types.SimpleNamespace(classes_=np.array(list(STAGE2_LABELS)))
    joblib.dump(
        {
            "model": _FakeSklearnModel(),
            "encoder": _encoder,
            "classes_present": list(range(len(STAGE2_LABELS))),
        },
        stage2_path,
    )
    stage1_sha = hashlib.sha256(stage1_path.read_bytes()).hexdigest()
    if corrupt_stage1_hash:
        stage1_sha = "0" * 64
    meta = {
        "thresholds": {"tau_red": 0.7, "tau_review": 0.3},
        "feature_cols": feature_cols,
        "stage2_labels": list(STAGE2_LABELS),
        "ddi_feature_semantics_version": DDI_FEATURE_SEMANTICS_VERSION,
        "stage1_sha256": stage1_sha,
        "stage2_sha256": hashlib.sha256(stage2_path.read_bytes()).hexdigest(),
    }
    (root / "stage_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return root


def test_hierarchical_predictor_load_rejects_unknown_feature_strict(tmp_path, monkeypatch):
    """HierarchicalPredictor.load 자체도 unknown feature_cols 를 거부해야 한다.

    HybridPredictor init/reload 경유가 아닌 직접 load 경로도 운영/테스트에서 쓰일 수
    있으므로, MLModel.load 와 같은 내부 schema guard 를 가져야 한다.
    """
    monkeypatch.delenv("FEATURE_SCHEMA_LENIENT", raising=False)
    artifact_dir = _write_hier_artifact(
        tmp_path / "unknown_schema",
        ["drug_count", "fake_unknown_col"],
    )

    hp = HierarchicalPredictor()
    assert hp.load(artifact_dir) is False, (
        "HierarchicalPredictor.load() 직접 호출도 strict schema drift 를 거부해야 함"
    )
    assert hp.loaded is False


def test_hierarchical_predictor_load_rejects_empty_feature_cols(tmp_path, monkeypatch):
    """계층 stage_meta.json 의 빈 feature_cols 는 명시적 로드 거부 대상이다.

    기존 good artifact 가 로드된 인스턴스에서도 empty artifact 실패 후 stale state 를
    남기면 안 된다.
    """
    monkeypatch.delenv("FEATURE_SCHEMA_LENIENT", raising=False)
    good_dir = _write_hier_artifact(tmp_path / "good_schema", ["drug_count", "age"])
    empty_dir = _write_hier_artifact(tmp_path / "empty_schema", [])

    hp = HierarchicalPredictor()
    assert hp.load(good_dir) is True
    assert hp.loaded is True

    assert hp.load(empty_dir) is False, (
        "빈 feature_cols 는 0-width 입력/학습-서빙 contract 붕괴라 로드 거부해야 함"
    )
    assert hp.loaded is False
    assert hp.feature_cols == []


def test_reload_hierarchical_rejects_empty_feature_cols(monkeypatch):
    """계층 모델 reload 도 빈 feature_cols 를 핫스왑하지 않아야 한다.

    이미 로드된 기존 계층 모델은 실패한 reload 후에도 그대로 보존되어야 한다.
    """
    monkeypatch.delenv("FEATURE_SCHEMA_LENIENT", raising=False)
    pred = _make_hier_predictor()
    existing_hp = MagicMock(spec=HierarchicalPredictor)
    pred._hierarchical = existing_hp

    fake_hp = MagicMock()
    fake_hp.load = MagicMock(return_value=True)
    fake_hp.feature_cols = []

    import serving.predictor as P
    monkeypatch.setattr(P, "HierarchicalPredictor", MagicMock(return_value=fake_hp))

    ok = pred.reload_hierarchical("/tmp/fake-empty-schema")
    assert ok is False, "빈 feature_cols 거부 시 reload 는 False"
    assert pred._hierarchical is existing_hp, "빈 feature_cols 거부 시 기존 _hierarchical 보존"


def test_hierarchical_predictor_load_hash_failure_clears_previous_state(tmp_path, monkeypatch):
    """이미 로드된 인스턴스도 다음 load 실패 시 stale loaded 상태를 남기면 안 된다."""
    monkeypatch.delenv("FEATURE_SCHEMA_LENIENT", raising=False)
    good_dir = _write_hier_artifact(tmp_path / "good", ["drug_count", "age"])
    bad_dir = _write_hier_artifact(
        tmp_path / "bad_hash",
        ["drug_count", "age"],
        corrupt_stage1_hash=True,
    )

    hp = HierarchicalPredictor()
    assert hp.load(good_dir) is True
    assert hp.loaded is True

    assert hp.load(bad_dir) is False
    assert hp.loaded is False
    assert hp.feature_cols == []
