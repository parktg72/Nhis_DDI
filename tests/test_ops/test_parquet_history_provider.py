from __future__ import annotations

from datetime import date
import json
import pickle
import sys

import pandas as pd

from scripts.datasets.contracts import DL_DATASET_REQUIRED_COLUMNS, write_dl_bundle_manifest
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
        json.dumps({"schema_version": "dl.v1.parquet"}),
        encoding="utf-8",
    )
    write_dl_bundle_manifest(
        root,
        run_id="parquet-provider",
        schema_version="dl.v1.parquet",
        lookback_days=365,
    )


def _write_records(path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_parquet(path, index=False)


def _base_rows() -> list[dict]:
    return [
        {
            "patient_id": "P001",
            "institution_id": "I001",
            "bill_no": "B001",
            "wk_compn_cd": "W1",
            "edi_code": "D1",
            "gnl_nm_cd": None,
            "efmdc_clsf_no": "100",
            "start_date": "2024-10-01",
            "end_date": "2024-10-03",
            "total_days": 3,
            "dose_once": 1.0,
            "dose_freq": 1,
            "sick_code": "K291",
            "sex": "1",
            "age_id": "62",
            "institution_type": "20",
            "source": "T30",
        },
        {
            "patient_id": "P001",
            "institution_id": "I002",
            "bill_no": "B002",
            "wk_compn_cd": "W2",
            "edi_code": "D2",
            "gnl_nm_cd": None,
            "efmdc_clsf_no": "200",
            "start_date": "2024-10-01",
            "end_date": "2024-10-01",
            "total_days": 1,
            "dose_once": 1.0,
            "dose_freq": 1,
            "sick_code": "I612",
            "sex": "1",
            "age_id": "62",
            "institution_type": "20",
            "source": "T60",
        },
        {
            "patient_id": "P002",
            "institution_id": "I003",
            "bill_no": "B003",
            "wk_compn_cd": "W3",
            "edi_code": "D3",
            "gnl_nm_cd": None,
            "efmdc_clsf_no": "300",
            "start_date": "2024-10-01",
            "end_date": "2024-10-01",
            "total_days": 1,
            "dose_once": 1.0,
            "dose_freq": 1,
            "sick_code": "H109",
            "sex": "2",
            "age_id": "44",
            "institution_type": "0",
            "source": "T30",
        },
    ]


def test_parquet_history_provider_returns_minimal_history_schema(tmp_path) -> None:
    from scripts.ops.parquet_history_provider import ParquetHistoryProvider

    path = tmp_path / "records.parquet"
    _write_records(path, _base_rows())

    result = ParquetHistoryProvider(path).fetch_patient_history(
        "P001",
        reference_date=date(2026, 5, 18),
        lookback_days=365,
    )

    validate_history_frame(result, context="parquet history")
    assert tuple(result.columns) == DL_DATASET_REQUIRED_COLUMNS


def test_parquet_history_provider_filters_by_patient_id(tmp_path) -> None:
    from scripts.ops.parquet_history_provider import ParquetHistoryProvider

    path = tmp_path / "records.parquet"
    _write_records(path, _base_rows())

    result = ParquetHistoryProvider(path).fetch_patient_history(
        "P002",
        reference_date=date(2026, 5, 18),
        lookback_days=365,
    )

    assert result["patient_id"].tolist() == ["P002"]
    assert result["drug_code"].tolist() == ["D3"]


def test_parquet_history_provider_returns_empty_schema_for_unknown_patient(tmp_path) -> None:
    from scripts.ops.parquet_history_provider import ParquetHistoryProvider

    path = tmp_path / "records.parquet"
    _write_records(path, _base_rows())

    result = ParquetHistoryProvider(path).fetch_patient_history(
        "UNKNOWN",
        reference_date=date(2026, 5, 18),
        lookback_days=365,
    )

    assert result.empty
    assert tuple(result.columns) == DL_DATASET_REQUIRED_COLUMNS


def test_parquet_history_provider_deduplicates_full_rows(tmp_path) -> None:
    from scripts.ops.parquet_history_provider import ParquetHistoryProvider

    path = tmp_path / "records.parquet"
    rows = _base_rows()
    rows.append(rows[0].copy())
    _write_records(path, rows)

    result = ParquetHistoryProvider(path).fetch_patient_history(
        "P001",
        reference_date=date(2026, 5, 18),
        lookback_days=365,
    )

    assert result["drug_code"].tolist() == ["D1", "D2"]


def test_parquet_history_provider_deduplicates_output_keys(tmp_path) -> None:
    from scripts.ops.parquet_history_provider import ParquetHistoryProvider

    path = tmp_path / "records.parquet"
    rows = _base_rows()
    key_duplicate = rows[0].copy()
    key_duplicate["bill_no"] = "B999"
    key_duplicate["source"] = "T60"
    key_duplicate["total_days"] = 90
    rows.append(key_duplicate)
    _write_records(path, rows)

    result = ParquetHistoryProvider(path).fetch_patient_history(
        "P001",
        reference_date=date(2026, 5, 18),
        lookback_days=365,
    )

    assert result["drug_code"].tolist() == ["D1", "D2"]


def test_parquet_history_provider_formats_prescription_date_as_yyyymmdd(tmp_path) -> None:
    from scripts.ops.parquet_history_provider import ParquetHistoryProvider

    path = tmp_path / "records.parquet"
    _write_records(path, _base_rows())

    result = ParquetHistoryProvider(path).fetch_patient_history(
        "P001",
        reference_date=date(2026, 5, 18),
        lookback_days=365,
    )

    assert result["prescription_date"].tolist() == ["20241001", "20241001"]


def test_parquet_history_provider_enables_hybrid_predictor_dl_prediction(
    monkeypatch,
    tmp_path,
) -> None:
    from scripts.ops.parquet_history_provider import ParquetHistoryProvider
    from serving.predictor import HybridPredictor
    from serving.schemas import DrugItem, PredictRequest, RiskLevel

    path = tmp_path / "records.parquet"
    _write_records(path, _base_rows())
    monkeypatch.setitem(sys.modules, "torch", _FakeTorch())
    bundle = tmp_path / "models" / "dl" / "parquet"
    _write_fake_inference_bundle(bundle)
    pred = HybridPredictor(
        ddi_matrix_path=tmp_path / "missing-ddi.parquet",
        drug_index_path=tmp_path / "missing-drugs.parquet",
        cyp_matrix_path=tmp_path / "missing-cyp.parquet",
        dl_history_provider=ParquetHistoryProvider(path),
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
    assert result.dl_prediction.run_id == "parquet-provider"
    assert result.dl_prediction.predicted_label == "high"
    assert result.dl_prediction.known_drug_count == 2
    assert result.dl_prediction.unknown_drug_count == 0
