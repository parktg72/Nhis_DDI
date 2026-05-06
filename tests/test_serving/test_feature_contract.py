import hashlib, pickle
from pathlib import Path
import numpy as np
import pytest


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
    payload = {
        "model": _FakeModel(),
        "params": {},
        "feature_importances": None,
        "best_threshold": 0.5,
        "trainer_class": "XGBoostTrainer",
        "artifact_version": 2,
        "feature_names": feature_names,
        "scaler_path": "../features/scaler.pkl",   # relative to model_dir
        "selector_path": "../features/selector.pkl",
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
    from serving.schemas import PredictRequest, DrugItem

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
