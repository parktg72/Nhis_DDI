import hashlib
import pickle

import numpy as np


class _FakeModel:
    """Picklable fake model for testing."""
    def predict_proba(self, X):
        return np.array([[0.3, 0.7]])


def _write_model(tmp_path, feature_names):
    """Write a minimal model pkl with sha256 sidecar."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    feat_dir = tmp_path / "features"
    feat_dir.mkdir()

    path = model_dir / "model.pkl"
    # Codex 2026-05-07 #2 — sidecar hash 검증 정책 적용 후 scaler_path/selector_path
    # 명시 시 sidecar 무결성 검증 필수. 본 테스트는 feature_names 정렬만 보는 거라
    # sidecar 의존성 제거 (정상 sidecar 케이스는 test_sidecar_hash.py 별도 검증).
    from scripts.etl.prescription_aggregator import DDI_FEATURE_SEMANTICS_VERSION
    payload = {
        "model": _FakeModel(),
        "params": {},
        "feature_importances": None,
        "best_threshold": 0.5,
        "trainer_class": "XGBoostTrainer",
        "artifact_version": 2,
        "feature_names": feature_names,
        # Q5: ddi_* 피처 모델은 DDI 시맨틱 버전 스탬프 필요(없으면 로드 거부).
        "ddi_feature_semantics_version": DDI_FEATURE_SEMANTICS_VERSION,
    }
    content = pickle.dumps(payload)
    path.write_bytes(content)
    sha = hashlib.sha256(content).hexdigest()
    path.with_suffix(".pkl.sha256").write_text(f"{sha}  {path.name}\n")
    return path, feat_dir


def test_mlmodel_loads_feature_names(tmp_path):
    from serving.predictor import MLModel
    path, _ = _write_model(tmp_path, ["drug_count", "age", "ddi_major"])
    ml = MLModel()
    ok = ml.load(path)
    assert ok
    assert ml._feature_names == ["drug_count", "age", "ddi_major"]
    assert ml._artifact_version == 2


def test_builder_aligns_to_feature_names(tmp_path):
    """Builder must reorder features to match training order."""
    from serving.predictor import RequestFeatureBuilder
    from serving.schemas import DrugItem, PredictRequest

    req = PredictRequest(
        patient_id="p1",
        drugs=[DrugItem(edi_code="A001", drug_name="aspirin", total_days=30)],
        patient_age=65,
    )
    builder = RequestFeatureBuilder()
    feature_names = ["age", "drug_count", "ddi_major"]
    vec, feat = builder.build(req, feature_names=feature_names)

    assert len(vec) == len(feature_names)
    assert vec[0] == feat.get("age", 0.0)
    assert vec[1] == feat.get("drug_count", 0.0)


def test_training_default_feature_cols_allowed_by_serving_schema():
    """Page 3 training defaults are covered before a model artifact exists.

    `test_feature_schema_strict.py` validates saved model artifact feature_names.
    This test validates the default training list itself, catching drift earlier.
    """
    from hana_app.core.ml_runner import FEATURE_COLS
    from serving.predictor import _FEATURE_ALLOWED

    extra = set(FEATURE_COLS) - _FEATURE_ALLOWED
    assert not extra, f"FEATURE_COLS not covered by serving allowed-set: {extra}"


def test_builder_aligns_to_training_default_feature_cols():
    """RequestFeatureBuilder must compute and order every default training feature."""
    from hana_app.core.ml_runner import FEATURE_COLS
    from serving.predictor import RequestFeatureBuilder
    from serving.schemas import DrugItem, PredictRequest

    req = PredictRequest(
        patient_id="p1",
        drugs=[
            DrugItem(
                edi_code="A001",
                drug_name="warfarin",
                atc_code="B01AA03",
                total_days=30,
                institution_id="I1",
            ),
            DrugItem(
                edi_code="A002",
                drug_name="ibuprofen",
                atc_code="M01AE01",
                total_days=10,
                institution_id="I2",
            ),
        ],
        patient_age=75,
        patient_sex="M",
    )
    builder = RequestFeatureBuilder()
    vec, feat = builder.build(req, feature_names=FEATURE_COLS)

    assert len(vec) == len(FEATURE_COLS)
    missing = set(FEATURE_COLS) - set(feat.keys())
    assert not missing, f"FEATURE_COLS produced via silent 0.0 fallback: {missing}"
    for i, name in enumerate(FEATURE_COLS):
        assert vec[i] == feat[name]
    assert np.isfinite(vec).all()
