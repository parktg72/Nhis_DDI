from __future__ import annotations

from datetime import date
import json

import numpy as np
import pandas as pd
from scipy import sparse


def _hist(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_build_future_outcome_sparse_dataset_filters_to_evaluable_patients() -> None:
    from scripts.ops.build_future_outcome_dataset import build_future_outcome_sparse_dataset

    vocab = {"_unk": 0, "D_A": 1, "D_B": 2}
    patient_ids = ["P_pos", "P_neg", "P_cens", "P_persist"]
    oct_histories = _hist([
        {"patient_id": "P_pos", "drug_code": "D_A", "institution_id": "A"},
        {"patient_id": "P_neg", "drug_code": "D_B", "institution_id": "A"},
        {"patient_id": "P_cens", "drug_code": "D_A", "institution_id": "A"},
        {"patient_id": "P_persist", "drug_code": "D_B", "institution_id": "A"},
        {"patient_id": "P_persist", "drug_code": "D_A", "institution_id": "B"},
    ])
    nov_histories = _hist([
        {"patient_id": "P_pos", "drug_code": "NOV_ONLY", "institution_id": "A"},
        {"patient_id": "P_pos", "drug_code": "NOV_ONLY", "institution_id": "B"},
        {"patient_id": "P_neg", "drug_code": "NOV_ONLY", "institution_id": "A"},
        {"patient_id": "P_persist", "drug_code": "NOV_ONLY", "institution_id": "A"},
    ])

    X, y, kept_patient_ids, metadata = build_future_outcome_sparse_dataset(
        oct_histories,
        nov_histories,
        patient_ids,
        vocab,
        threshold=2,
    )

    assert kept_patient_ids == ["P_pos", "P_neg"]
    assert y.tolist() == [1, 0]
    assert X.shape == (2, 3)
    assert X[0, 1] == 1
    assert X[1, 2] == 1
    assert X[:, 0].sum() == 0
    assert metadata["label_source"] == "future_multi_institution_onset"
    assert metadata["n_censored"] == 1
    assert metadata["persistence_excluded_count"] == 1


def test_unknown_oct_drug_maps_to_unk_and_nov_drugs_do_not_enter_features() -> None:
    from scripts.ops.build_future_outcome_dataset import build_future_outcome_sparse_dataset

    vocab = {"_unk": 0, "D_A": 1}
    X, y, kept_patient_ids, metadata = build_future_outcome_sparse_dataset(
        _hist([{"patient_id": "P1", "drug_code": "UNKNOWN", "institution_id": "A"}]),
        _hist([
            {"patient_id": "P1", "drug_code": "D_A", "institution_id": "A"},
            {"patient_id": "P1", "drug_code": "D_A", "institution_id": "B"},
        ]),
        ["P1"],
        vocab,
        threshold=2,
    )

    assert kept_patient_ids == ["P1"]
    assert y.tolist() == [1]
    assert X[0, 0] == 1
    assert X[0, 1] == 0
    assert metadata["unknown_drug_count"] == 1
    assert metadata["total_drug_rows"] == 1


def test_write_future_outcome_dataset_outputs_artifacts(tmp_path) -> None:
    from scripts.ops.build_future_outcome_dataset import write_future_outcome_dataset

    X = sparse.eye(2, 3, dtype=np.float32, format="csr")
    y = np.array([1, 0], dtype=np.int8)
    patient_ids = ["P1", "P2"]
    metadata = {
        "label_source": "future_multi_institution_onset",
        "feature_reference_date": date(2024, 10, 31).isoformat(),
    }

    result = write_future_outcome_dataset(
        X,
        y,
        patient_ids,
        metadata,
        tmp_path,
    )

    assert (tmp_path / "X_csr.npz").exists()
    assert (tmp_path / "y.npy").exists()
    assert (tmp_path / "patient_ids.npy").exists()
    assert (tmp_path / "metadata.json").exists()
    assert result["n_patients"] == 2
    assert json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))["label_source"] == "future_multi_institution_onset"


def test_add_institution_count_feature_appends_normalized_scalar() -> None:
    from scripts.ops.build_future_outcome_dataset import build_future_outcome_sparse_dataset

    X, y, kept_patient_ids, metadata = build_future_outcome_sparse_dataset(
        _hist([
            {"patient_id": "P1", "drug_code": "D_A", "institution_id": "A"},
            {"patient_id": "P1", "drug_code": "D_A", "institution_id": "B"},
            {"patient_id": "P2", "drug_code": "D_A", "institution_id": "A"},
        ]),
        _hist([
            {"patient_id": "P1", "drug_code": "D_A", "institution_id": "A"},
            {"patient_id": "P1", "drug_code": "D_A", "institution_id": "B"},
            {"patient_id": "P1", "drug_code": "D_A", "institution_id": "C"},
            {"patient_id": "P2", "drug_code": "D_A", "institution_id": "A"},
        ]),
        ["P1", "P2"],
        {"_unk": 0, "D_A": 1},
        threshold=3,
        add_institution_count_feature=True,
    )

    assert kept_patient_ids == ["P1", "P2"]
    assert y.tolist() == [1, 0]
    assert X.shape == (2, 3)
    assert X[0, 2] == 1.0
    assert X[1, 2] == 0.5
    assert metadata["add_institution_count_feature"] is True
    assert metadata["institution_count_feature_index"] == 2
    assert metadata["input_dim"] == 3
