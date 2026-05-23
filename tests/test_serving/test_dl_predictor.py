from __future__ import annotations

import importlib
import json
import pickle
import sys

import pandas as pd
import pytest

from scripts.datasets.contracts import (
    BundleHashMismatchError,
    DL_BUNDLE_REQUIRED_FILES,
    LookbackMismatchError,
    write_dl_bundle_manifest,
)


def _write_bundle(root, lookback_days: int = 365) -> None:
    for name in DL_BUNDLE_REQUIRED_FILES:
        (root / name).write_bytes(f"artifact:{name}".encode("utf-8"))
    write_dl_bundle_manifest(
        root,
        run_id="dl-run-001",
        schema_version="dl.v1",
        lookback_days=lookback_days,
    )


def _write_inference_bundle(
    root,
    lookback_days: int = 365,
    architecture: str | None = None,
) -> None:
    model_config = {
        "encoding_strategy": "multi_hot",
        "input_dim": 3,
        "output_labels": ["low", "high"],
    }
    if architecture is not None:
        model_config["architecture"] = architecture
    (root / "model.pt").write_bytes(b"fake scripted module")
    (root / "model_config.json").write_text(
        json.dumps(model_config),
        encoding="utf-8",
    )
    (root / "drug_vocab.json").write_text(
        json.dumps({"D1": 0, "D2": 1, "D3": 2}),
        encoding="utf-8",
    )
    (root / "edge_index.pt").write_bytes(b"fake edge index")
    (root / "feature_normalizer.pkl").write_bytes(
        pickle.dumps({"type": "identity"})
    )
    (root / "schema_version.json").write_text(
        json.dumps({"schema_version": "dl.v1"}),
        encoding="utf-8",
    )
    write_dl_bundle_manifest(
        root,
        run_id="dl-run-001",
        schema_version="dl.v1",
        lookback_days=lookback_days,
    )


class _FakeTensor:
    def __init__(self, data):
        self._data = data

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self._data


class _FakeModel:
    def __init__(self) -> None:
        self.inputs = []

    def eval(self) -> None:
        return None

    def __call__(self, features):
        self.inputs.append(features)
        return _FakeTensor([[0.25, 0.75]])


class _FakeGraphModel:
    def __init__(self) -> None:
        self.inputs = []

    def eval(self) -> None:
        return None

    def __call__(self, features, edge_index):
        self.inputs.append((features, edge_index))
        return _FakeTensor([[0.2, 0.8]])


class _FakeNoGrad:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeTorch:
    float32 = "float32"

    def __init__(self, loaded_model=None) -> None:
        self.loaded_model = loaded_model or _FakeModel()
        self.loaded_edge_index = _FakeTensor([[0], [1]])

    class _Jit:
        def __init__(self, outer) -> None:
            self._outer = outer

        def load(self, path, map_location=None):
            return self._outer.loaded_model

    @property
    def jit(self):
        return self._Jit(self)

    def load(self, path, map_location=None, weights_only=None):
        return self.loaded_edge_index

    def tensor(self, data, dtype=None, device=None):
        return data

    def softmax(self, logits, dim=-1):
        return logits

    def no_grad(self):
        return _FakeNoGrad()


def test_dl_predictor_module_does_not_import_torch() -> None:
    torch_before = sys.modules.get("torch")
    torch_geometric_before = sys.modules.get("torch_geometric")
    sys.modules.pop("serving.dl_predictor", None)

    importlib.import_module("serving.dl_predictor")

    assert sys.modules.get("torch") is torch_before
    assert sys.modules.get("torch_geometric") is torch_geometric_before


def test_dl_model_loads_valid_bundle_with_matching_lookback(tmp_path) -> None:
    from serving.dl_predictor import DLModel

    _write_bundle(tmp_path, lookback_days=365)

    model = DLModel(runtime_lookback_days=365)
    assert model.load(tmp_path) is True
    assert model.is_loaded
    assert model.loaded
    assert model.bundle_dir == tmp_path
    assert model.run_id == "dl-run-001"
    assert model.schema_version == "dl.v1"
    assert model.lookback_days == 365
    assert model.manifest is not None


def test_dl_model_lookback_mismatch_fails_fast_and_can_recover(tmp_path) -> None:
    from serving.dl_predictor import DLModel

    bad_bundle = tmp_path / "bad"
    bad_bundle.mkdir()
    _write_bundle(bad_bundle, lookback_days=180)
    good_bundle = tmp_path / "good"
    good_bundle.mkdir()
    _write_bundle(good_bundle, lookback_days=365)

    model = DLModel(runtime_lookback_days=365)
    with pytest.raises(LookbackMismatchError, match="artifact=180, runtime=365"):
        model.load(bad_bundle)

    assert not model.is_loaded
    assert model.bundle_dir is None
    assert model.manifest is None

    assert model.load(good_bundle) is True
    assert model.is_loaded
    assert model.bundle_dir == good_bundle


def test_dl_model_rejects_bundle_hash_mismatch(tmp_path) -> None:
    from serving.dl_predictor import DLModel

    _write_bundle(tmp_path, lookback_days=365)
    (tmp_path / "drug_vocab.json").write_text("changed", encoding="utf-8")

    model = DLModel(runtime_lookback_days=365)
    with pytest.raises(BundleHashMismatchError, match="hash mismatch"):
        model.load(tmp_path)
    assert not model.is_loaded


def test_dl_model_keeps_previous_bundle_when_reload_fails(tmp_path) -> None:
    from serving.dl_predictor import DLModel

    good_bundle = tmp_path / "good"
    good_bundle.mkdir()
    _write_bundle(good_bundle, lookback_days=365)
    bad_bundle = tmp_path / "bad"
    bad_bundle.mkdir()
    _write_bundle(bad_bundle, lookback_days=180)

    model = DLModel(runtime_lookback_days=365)
    model.load(good_bundle)
    assert model.is_loaded
    assert model.bundle_dir == good_bundle
    assert model.run_id == "dl-run-001"

    with pytest.raises(LookbackMismatchError):
        model.load(bad_bundle)

    assert model.is_loaded
    assert model.bundle_dir == good_bundle
    assert model.run_id == "dl-run-001"


def test_dl_model_predict_loads_runtime_artifacts_lazily(monkeypatch, tmp_path) -> None:
    fake_torch = _FakeTorch()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    from serving.dl_predictor import DLModel

    _write_inference_bundle(tmp_path, lookback_days=365)
    model = DLModel(runtime_lookback_days=365)
    model.load(tmp_path)

    history = pd.DataFrame({
        "patient_id": ["P1", "P1", "P1"],
        "drug_code": ["D2", "UNKNOWN", "D1"],
        "prescription_date": ["20260510", "20260510", "20260511"],
    })

    result = model.predict(history)

    assert fake_torch.loaded_model.inputs == [[[1.0, 1.0, 0.0]]]
    assert result == {
        "run_id": "dl-run-001",
        "encoding_strategy": "multi_hot",
        "predicted_label": "high",
        "score": 0.75,
        "probabilities": {"low": 0.25, "high": 0.75},
        "known_drug_count": 2,
        "unknown_drug_count": 1,
    }


def test_dl_model_predict_passes_edge_index_for_graph_architecture(monkeypatch, tmp_path) -> None:
    graph_model = _FakeGraphModel()
    fake_torch = _FakeTorch(loaded_model=graph_model)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    from serving.dl_predictor import DLModel

    _write_inference_bundle(tmp_path, architecture="gat", lookback_days=365)
    model = DLModel(runtime_lookback_days=365)
    model.load(tmp_path)

    history = pd.DataFrame({
        "patient_id": ["P1", "P1"],
        "drug_code": ["D2", "D1"],
        "prescription_date": ["20260510", "20260511"],
    })

    result = model.predict(history)

    assert graph_model.inputs == [(
        [[1.0, 1.0, 0.0]],
        fake_torch.loaded_edge_index,
    )]
    assert result["predicted_label"] == "high"
    assert result["score"] == 0.8
