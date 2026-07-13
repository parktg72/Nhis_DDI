from __future__ import annotations

import json
import pickle
import sys
from datetime import date

from scripts.datasets.contracts import (
    DL_DATASET_REQUIRED_COLUMNS,
    write_dl_bundle_manifest,
)
from serving.hana_history import validate_history_frame


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
    def __call__(self, features):
        return _FakeTensor([[0.2, 0.8]])

    def eval(self) -> None:
        return None


class _FakeNoGrad:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeTorch:
    float32 = "float32"

    class _Jit:
        def load(self, path, map_location=None):
            return _FakeModel()

    @property
    def jit(self):
        return self._Jit()

    def load(self, path, map_location=None, weights_only=None):
        return _FakeTensor([[0], [1]])

    def tensor(self, data, dtype=None, device=None):
        return data

    def softmax(self, logits, dim=-1):
        return logits

    def no_grad(self):
        return _FakeNoGrad()


def _write_fake_inference_bundle(root) -> None:
    root.mkdir(parents=True)
    (root / "model.pt").write_bytes(b"fake scripted module")
    (root / "model_config.json").write_text(
        json.dumps({
            "encoding_strategy": "multi_hot",
            "input_dim": 3,
            "output_labels": ["low", "high"],
        }),
        encoding="utf-8",
    )
    (root / "drug_vocab.json").write_text(
        json.dumps({"D1": 0, "D2": 1, "D3": 2}),
        encoding="utf-8",
    )
    (root / "edge_index.pt").write_bytes(b"fake edge index")
    (root / "feature_normalizer.pkl").write_bytes(pickle.dumps({"type": "identity"}))
    (root / "schema_version.json").write_text(
        json.dumps({"schema_version": "dl.v1.smoke"}),
        encoding="utf-8",
    )
    write_dl_bundle_manifest(
        root,
        run_id="smoke-provider",
        schema_version="dl.v1.smoke",
        lookback_days=365,
    )


def test_smoke_history_provider_returns_valid_dl_history_frame() -> None:
    from scripts.ops.smoke_history_provider import SmokeHistoryProvider

    provider = SmokeHistoryProvider()

    result = provider.fetch_patient_history(
        "P001",
        reference_date=date(2026, 5, 18),
        lookback_days=365,
    )

    validate_history_frame(result, context="smoke history")
    assert tuple(result.columns[:3]) == DL_DATASET_REQUIRED_COLUMNS
    assert result["patient_id"].tolist() == ["P001", "P001"]
    assert result["drug_code"].tolist() == ["D1", "D2"]
    assert result["prescription_date"].tolist() == ["20260517", "20260518"]


def test_smoke_history_provider_can_include_unknown_drug_for_warning_path() -> None:
    from scripts.ops.smoke_history_provider import SmokeHistoryProvider

    provider = SmokeHistoryProvider(include_unknown=True)

    result = provider.fetch_patient_history(
        "P002",
        reference_date=date(2026, 5, 18),
        lookback_days=7,
    )

    assert result["patient_id"].tolist() == ["P002", "P002", "P002"]
    assert result["drug_code"].tolist() == ["D1", "D2", "UNKNOWN_SMOKE"]
    assert result["prescription_date"].tolist()[-1] == "20260518"


def test_smoke_history_provider_enables_hybrid_predictor_dl_prediction(
    monkeypatch,
    tmp_path,
) -> None:
    from scripts.ops.smoke_history_provider import SmokeHistoryProvider
    from serving.predictor import HybridPredictor
    from serving.schemas import DrugItem, PredictRequest, RiskLevel

    monkeypatch.setitem(sys.modules, "torch", _FakeTorch())
    bundle = tmp_path / "models" / "dl" / "smoke"
    _write_fake_inference_bundle(bundle)
    pred = HybridPredictor(
        ddi_matrix_path=tmp_path / "missing-ddi.parquet",
        drug_index_path=tmp_path / "missing-drugs.parquet",
        cyp_matrix_path=tmp_path / "missing-cyp.parquet",
        dl_history_provider=SmokeHistoryProvider(),
    )
    pred.reload_dl(bundle)

    monkeypatch.setattr(
        "serving.predictor._run_safety_net",
        lambda drugs, **kwargs: (RiskLevel.NORMAL, [], []),
    )
    monkeypatch.setattr(
        "serving.predictor._run_duplicate_detector",
        lambda drugs, **kwargs: (0, []),
    )

    result = pred.predict(PredictRequest(
        patient_id="P001",
        reference_date=date(2026, 5, 18),
        drugs=[DrugItem(edi_code="D1", total_days=7)],
    ))

    assert result.risk_level == RiskLevel.NORMAL
    assert result.dl_error is None
    assert result.dl_prediction is not None
    assert result.dl_prediction.run_id == "smoke-provider"
    assert result.dl_prediction.predicted_label == "high"
    assert result.dl_prediction.known_drug_count == 2
    assert result.dl_prediction.unknown_drug_count == 0
