"""scaler/selector sidecar hash 검증 회귀 가드 — Codex 2026-05-07 #2.

배경: 직전까지 MLModel.load 가 주 모델만 sha256 무결성 검증, scaler/selector
sidecar pickle 은 hash 없이 직접 unpickle. 전처리 artifact 가 바뀌면 feature
vector 의미가 조용히 바뀌어 prediction 이 잘못된 확률 반환 가능.

Codex 합의 정책 (이번 PR 적용):
  - sidecar `scaler_path`/`selector_path` 가 state 에 명시되면 artifact 구성요소
  - .sha256 부재 / hash 불일치 / 파일 부재 / path traversal 모두 모델 로드 실패
  - read_bytes → _verify_hash → pickle.loads (TOCTOU 회피, 메인 모델 동일 패턴)
  - 모든 sidecar 검증 통과 후에만 instance state 반영 (partial state 오염 방지)
"""
from __future__ import annotations

import hashlib
import pickle
from pathlib import Path


class _FakeModel:
    def predict_proba(self, X):
        import numpy as np
        prob = np.full(len(X), 0.5)
        return np.column_stack([1 - prob, prob])


def _write_main_model(
    model_dir: Path,
    *,
    scaler_path: str | None = None,
    selector_path: str | None = None,
    feature_names: list[str] | None = None,
) -> Path:
    """주 모델 pkl + sha256. feature_names 디폴트는 known cols (schema strict 통과)."""
    path = model_dir / "model.pkl"
    payload: dict = {
        "model": _FakeModel(),
        "best_threshold": 0.5,
        "trainer_class": "XGBoostTrainer",
        "feature_names": feature_names or ["drug_count", "age"],
        "artifact_version": 2,
    }
    if scaler_path is not None:
        payload["scaler_path"] = scaler_path
    if selector_path is not None:
        payload["selector_path"] = selector_path
    content = pickle.dumps(payload)
    path.write_bytes(content)
    sha = hashlib.sha256(content).hexdigest()
    path.with_suffix(".pkl.sha256").write_text(f"{sha}  {path.name}\n")
    return path


def _write_sidecar(path: Path, obj=None, *, with_hash: bool = True) -> Path:
    """sidecar pickle artifact + sha256 sidecar."""
    obj = obj or {"dummy": "scaler"}
    content = pickle.dumps(obj)
    path.write_bytes(content)
    if with_hash:
        sha = hashlib.sha256(content).hexdigest()
        path.with_suffix(path.suffix + ".sha256").write_text(f"{sha}  {path.name}\n")
    return path


# ─── 5 케이스 + traversal ────────────────────────────────────────────────────


def test_sidecar_valid_hash_loads_ok(tmp_path):
    """정상 sidecar + .sha256 일치 → 모델 + scaler/selector 모두 instance 반영."""
    from serving.predictor import MLModel

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    _write_sidecar(model_dir / "scaler.pkl", obj={"k": "scaler_v"})
    _write_sidecar(model_dir / "selector.pkl", obj={"k": "selector_v"})
    main_path = _write_main_model(
        model_dir,
        scaler_path="scaler.pkl",
        selector_path="selector.pkl",
    )

    ml = MLModel()
    assert ml.load(main_path) is True
    assert ml._scaler == {"k": "scaler_v"}
    assert ml._selector == {"k": "selector_v"}


def test_sidecar_missing_sha256_rejects_load(tmp_path):
    """sidecar 자체는 있지만 .sha256 부재 → 모델 로드 실패."""
    from serving.predictor import MLModel

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    _write_sidecar(model_dir / "scaler.pkl", with_hash=False)  # hash 없음
    main_path = _write_main_model(model_dir, scaler_path="scaler.pkl")

    ml = MLModel()
    assert ml.load(main_path) is False, "sidecar .sha256 부재 → 로드 거부"
    # state 오염 없음
    assert ml._model is None
    assert ml._scaler is None
    assert ml._feature_names == []


def test_sidecar_hash_mismatch_rejects_load(tmp_path):
    """sidecar 과 .sha256 불일치 → 모델 로드 실패."""
    from serving.predictor import MLModel

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    sc_path = _write_sidecar(model_dir / "scaler.pkl")
    # hash 변조
    sc_path.with_suffix(".pkl.sha256").write_text(
        "0" * 64 + f"  {sc_path.name}\n"
    )
    main_path = _write_main_model(model_dir, scaler_path="scaler.pkl")

    ml = MLModel()
    assert ml.load(main_path) is False
    assert ml._scaler is None


def test_sidecar_file_missing_rejects_load(tmp_path):
    """state 에 scaler_path 명시됐지만 파일 자체 부재 → 모델 로드 실패."""
    from serving.predictor import MLModel

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    # scaler.pkl 자체를 만들지 않음
    main_path = _write_main_model(model_dir, scaler_path="scaler.pkl")

    ml = MLModel()
    assert ml.load(main_path) is False, "sidecar 파일 부재 → 로드 거부"


def test_sidecar_traversal_rejects_load(tmp_path):
    """sidecar path 가 model_dir 외부 (traversal) → 모델 로드 실패 + state 오염 없음."""
    from serving.predictor import MLModel

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    _write_sidecar(outside / "scaler.pkl")
    # `../outside/scaler.pkl` — model_dir 외부
    main_path = _write_main_model(
        model_dir, scaler_path="../outside/scaler.pkl"
    )

    ml = MLModel()
    assert ml.load(main_path) is False, "traversal path → 로드 거부 (직전엔 continue 였음)"
    # state 오염 없음
    assert ml._model is None
    assert ml._scaler is None
    assert ml._selector is None


def test_no_sidecar_path_specified_loads_ok(tmp_path):
    """state 에 scaler_path/selector_path 미명시 → sidecar 검증 skip, 모델 정상 로드."""
    from serving.predictor import MLModel

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    main_path = _write_main_model(model_dir)  # sidecar path 미지정

    ml = MLModel()
    assert ml.load(main_path) is True
    assert ml._scaler is None
    assert ml._selector is None
    assert ml._model is not None


def test_partial_state_clean_when_one_sidecar_invalid(tmp_path):
    """scaler 정상 + selector hash 불일치 → 둘 다 instance 반영 X (partial state 방지)."""
    from serving.predictor import MLModel

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    _write_sidecar(model_dir / "scaler.pkl")  # 정상
    sel_path = _write_sidecar(model_dir / "selector.pkl")
    # selector hash 변조
    sel_path.with_suffix(".pkl.sha256").write_text(
        "0" * 64 + f"  {sel_path.name}\n"
    )
    main_path = _write_main_model(
        model_dir, scaler_path="scaler.pkl", selector_path="selector.pkl"
    )

    ml = MLModel()
    assert ml.load(main_path) is False
    # 정상 scaler 도 instance 반영 안 됨 (모든 검증 통과 후에만 setattr)
    assert ml._scaler is None
    assert ml._selector is None
