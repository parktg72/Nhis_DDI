from __future__ import annotations

from datetime import date

import pandas as pd

VOCAB = {"_unk": 0, "D1": 1, "D2": 2, "D3": 3}


def _write_records(path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_parquet(path, index=False)


def _row(patient_id: str, edi_code: str, start_date: str, efmdc_clsf_no: str | None) -> dict:
    return {
        "patient_id": patient_id,
        "edi_code": edi_code,
        "start_date": start_date,
        "end_date": start_date,
        "total_days": 1,
        "source": "T30",
        "efmdc_clsf_no": efmdc_clsf_no,
    }


def test_full_cohort_loader_reads_window_once_and_normalizes(tmp_path) -> None:
    from scripts.ops.full_cohort_history_loader import FullCohortHistoryLoader

    _write_records(tmp_path / "records_20241001.parquet", [_row("P1", "D1", "2024-10-01", "114")])
    _write_records(tmp_path / "records_20241002.parquet", [_row("P1", "D2", "2024-10-02", "114")])
    _write_records(tmp_path / "records_20241003.parquet", [_row("P1", "D3", "2024-10-03", "396")])

    loader = FullCohortHistoryLoader(tmp_path, extra_columns=["efmdc_clsf_no"])
    histories = loader.load_window(
        reference_date=date(2024, 10, 2),
        lookback_days=1,
    )

    assert histories["drug_code"].tolist() == ["D1", "D2"]
    assert histories["prescription_date"].tolist() == [date(2024, 10, 1), date(2024, 10, 2)]
    assert histories["efmdc_clsf_no"].tolist() == ["114", "114"]
    assert loader.last_loaded_file_count == 2


def test_full_cohort_loader_can_filter_patient_ids(tmp_path) -> None:
    from scripts.ops.full_cohort_history_loader import FullCohortHistoryLoader

    _write_records(
        tmp_path / "records_20241001.parquet",
        [
            _row("P1", "D1", "2024-10-01", "114"),
            _row("P2", "D2", "2024-10-01", "114"),
        ],
    )

    histories = FullCohortHistoryLoader(tmp_path, extra_columns=["efmdc_clsf_no"]).load_window(
        reference_date=date(2024, 10, 1),
        lookback_days=0,
        patient_ids=["P2"],
    )

    assert histories["patient_id"].tolist() == ["P2"]


def test_iter_patient_batches_preserves_requested_order(tmp_path) -> None:
    from scripts.ops.full_cohort_history_loader import iter_patient_batches

    histories = pd.DataFrame([
        {"patient_id": "P2", "drug_code": "D2"},
        {"patient_id": "P1", "drug_code": "D1"},
        {"patient_id": "P3", "drug_code": "D3"},
    ])

    batches = list(iter_patient_batches(histories, ["P1", "P2", "P3"], batch_size=2))

    assert batches[0][0] == ["P1", "P2"]
    assert batches[0][1]["patient_id"].tolist() == ["P2", "P1"]
    assert batches[1][0] == ["P3"]


def test_build_sparse_dataset_outputs_csr_labels_and_stats() -> None:
    from scripts.ops.build_sparse_training_dataset import build_sparse_dataset

    histories = pd.DataFrame([
        {"patient_id": "P1", "drug_code": "D1", "efmdc_clsf_no": "114"},
        {"patient_id": "P1", "drug_code": "D2", "efmdc_clsf_no": "114"},
        {"patient_id": "P2", "drug_code": "D3", "efmdc_clsf_no": "114"},
        {"patient_id": "P2", "drug_code": "UNKNOWN", "efmdc_clsf_no": None},
        {"patient_id": "P3", "drug_code": None, "efmdc_clsf_no": None},
    ])

    X, y, stats = build_sparse_dataset(
        histories,
        ["P1", "P2", "P3"],
        VOCAB,
        label_source="therapeutic_duplication",
        therapeutic_dup_threshold=1,
    )

    assert X.shape == (3, len(VOCAB))
    assert X[0, VOCAB["D1"]] == 1
    assert X[0, VOCAB["D2"]] == 1
    assert X[1, VOCAB["D3"]] == 1
    assert X[1, VOCAB["_unk"]] == 1
    assert y.tolist() == [1, 0, 0]
    assert stats["n_patients"] == 3
    assert stats["input_dim"] == len(VOCAB)
    assert stats["nnz"] == 4
    assert stats["label_positive"] == 1
    assert stats["unknown_drug_count"] == 1
    assert stats["unk_flag_patients"] == 1
    assert stats["zero_vector_patients"] == 1


def test_build_sparse_dataset_supports_multi_institution_labels() -> None:
    from scripts.ops.build_sparse_training_dataset import build_sparse_dataset

    histories = pd.DataFrame([
        {"patient_id": "P1", "drug_code": "D1", "institution_id": "H1"},
        {"patient_id": "P1", "drug_code": "D2", "institution_id": "H2"},
        {"patient_id": "P1", "drug_code": "D3", "institution_id": "H3"},
        {"patient_id": "P2", "drug_code": "D1", "institution_id": "H1"},
        {"patient_id": "P2", "drug_code": "D2", "institution_id": "H1"},
    ])

    X, y, stats = build_sparse_dataset(
        histories,
        ["P1", "P2"],
        VOCAB,
        label_source="multi_institution",
        multi_institution_threshold=3,
    )

    assert X.shape == (2, len(VOCAB))
    assert y.tolist() == [1, 0]
    assert stats["label_source"] == "multi_institution"
    assert stats["multi_institution_threshold"] == 3


def test_build_sparse_dataset_rejects_bad_vocab() -> None:
    from scripts.ops.build_sparse_training_dataset import build_sparse_dataset

    try:
        build_sparse_dataset(
            pd.DataFrame(columns=["patient_id", "drug_code", "efmdc_clsf_no"]),
            ["P1"],
            {"D1": 0},
            label_source="therapeutic_duplication",
        )
    except ValueError as exc:
        assert "_unk" in str(exc)
    else:
        raise AssertionError("expected missing _unk ValueError")


def test_build_sparse_dataset_from_raw_keeps_reference_day_patient_cohort(tmp_path) -> None:
    import json

    import numpy as np
    from scipy import sparse

    from scripts.ops.build_sparse_training_dataset import build_sparse_dataset_from_raw

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_records(raw_dir / "records_20241001.parquet", [_row("OLD_ONLY", "D1", "2024-10-01", "114")])
    _write_records(
        raw_dir / "records_20241002.parquet",
        [
            _row("P1", "D1", "2024-10-02", "114"),
            _row("P1", "D2", "2024-10-02", "114"),
        ],
    )
    vocab_path = tmp_path / "vocab.json"
    vocab_path.write_text(json.dumps(VOCAB), encoding="utf-8")
    output_dir = tmp_path / "out"

    metadata = build_sparse_dataset_from_raw(
        raw_dir,
        vocab_path,
        output_dir,
        reference_date=date(2024, 10, 2),
        lookback_days=1,
        label_source="therapeutic_duplication",
        therapeutic_dup_threshold=1,
    )

    patient_ids = np.load(output_dir / "patient_ids.npy", allow_pickle=True).tolist()
    X = sparse.load_npz(output_dir / "X_csr.npz")

    assert patient_ids == ["P1"]
    assert X.shape == (1, len(VOCAB))
    assert metadata["n_patients"] == 1
    assert metadata["peak_rss_mb"] >= 0
