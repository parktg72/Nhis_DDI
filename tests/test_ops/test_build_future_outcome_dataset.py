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
    assert "no_third_month_caveat" not in metadata
    assert metadata["temporal_holdout_status"] == (
        "temporal holdout is available when a later feature/outcome pair is built"
    )


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


def test_add_demographics_feature_appends_age_and_sex_scalars() -> None:
    from scripts.ops.build_future_outcome_dataset import build_future_outcome_sparse_dataset

    demographics = {
        "P1": (0.44, 1.0),
        "P2": (0.74, 0.0),
    }

    X, y, kept_patient_ids, metadata = build_future_outcome_sparse_dataset(
        _hist([
            {"patient_id": "P1", "drug_code": "D_A", "institution_id": "A"},
            {"patient_id": "P2", "drug_code": "D_A", "institution_id": "A"},
        ]),
        _hist([
            {"patient_id": "P1", "drug_code": "D_A", "institution_id": "A"},
            {"patient_id": "P1", "drug_code": "D_A", "institution_id": "B"},
            {"patient_id": "P2", "drug_code": "D_A", "institution_id": "A"},
        ]),
        ["P1", "P2"],
        {"_unk": 0, "D_A": 1},
        threshold=2,
        demographics_features=demographics,
    )

    assert kept_patient_ids == ["P1", "P2"]
    assert y.tolist() == [1, 0]
    assert X.shape == (2, 4)
    assert X[0, 2] == 0.44
    assert X[0, 3] == 1.0
    assert X[1, 2] == 0.74
    assert X[1, 3] == 0.0
    assert metadata["add_demographics_feature"] is True
    assert metadata["demographics_feature_indices"] == {
        "age_years_div_100": 2,
        "sex_type_1_flag": 3,
    }
    assert metadata["demographics_sex_semantics"] == (
        "sex_type=1 -> 1.0, sex_type=2 -> 0.0, missing/other -> 0.5"
    )
    assert metadata["demographics_missing_patient_count"] == 0


def test_add_demographics_feature_records_missing_defaults() -> None:
    from scripts.ops.build_future_outcome_dataset import build_future_outcome_sparse_dataset

    X, _, kept_patient_ids, metadata = build_future_outcome_sparse_dataset(
        _hist([
            {"patient_id": "P1", "drug_code": "D_A", "institution_id": "A"},
            {"patient_id": "P2", "drug_code": "D_A", "institution_id": "A"},
        ]),
        _hist([
            {"patient_id": "P1", "drug_code": "D_A", "institution_id": "A"},
            {"patient_id": "P2", "drug_code": "D_A", "institution_id": "A"},
        ]),
        ["P1", "P2"],
        {"_unk": 0, "D_A": 1},
        threshold=2,
        demographics_features={"P1": (0.44, 1.0)},
    )

    assert kept_patient_ids == ["P1", "P2"]
    assert X.shape == (2, 4)
    assert X[1, 2] == 0.0
    assert X[1, 3] == 0.5
    assert metadata["demographics_missing_patient_count"] == 1
    assert metadata["demographics_missing_patient_rate_pct"] == 50.0


def test_add_medication_class_feature_appends_class_multihot_after_institution() -> None:
    from scripts.ops.medication_class_features import EFMDC_NULL_TOKEN, EFMDC_UNK_TOKEN
    from scripts.ops.build_future_outcome_dataset import build_future_outcome_sparse_dataset

    class_vocab = {
        EFMDC_NULL_TOKEN: 0,
        EFMDC_UNK_TOKEN: 1,
        "222": 2,
    }

    X, y, kept_patient_ids, metadata = build_future_outcome_sparse_dataset(
        _hist([
            {"patient_id": "P1", "drug_code": "D_A", "institution_id": "A", "efmdc_clsf_no": "222"},
            {"patient_id": "P1", "drug_code": "D_A", "institution_id": "B", "efmdc_clsf_no": None},
            {"patient_id": "P2", "drug_code": "D_A", "institution_id": "A", "efmdc_clsf_no": "999"},
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
        medication_class_vocab=class_vocab,
    )

    assert kept_patient_ids == ["P1", "P2"]
    assert y.tolist() == [1, 0]
    assert X.shape == (2, 6)
    assert X[0, 2] == 1.0
    assert X[0, 3] == 1.0
    assert X[0, 5] == 1.0
    assert X[1, 4] == 1.0
    assert metadata["add_medication_class_feature"] is True
    assert metadata["medication_class_feature_start_index"] == 3
    assert metadata["medication_class_feature_count"] == 3
    assert metadata["medication_class_null_token_index"] == 3
    assert metadata["medication_class_unknown_token_index"] == 4
    assert metadata["medication_class_null_row_rate_pct"] == 33.3333
    assert metadata["medication_class_oov_row_rate_pct"] == 33.3333
