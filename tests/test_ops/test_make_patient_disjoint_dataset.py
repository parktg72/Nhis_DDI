from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import sparse


def _write_dataset(path: Path, patient_ids: list[object], y: list[int]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    rows = len(patient_ids)
    X = sparse.csr_matrix(
        (
            np.arange(1, rows + 1, dtype=np.float32),
            (np.arange(rows), np.arange(rows)),
        ),
        shape=(rows, rows),
        dtype=np.float32,
    )
    sparse.save_npz(path / "X_csr.npz", X)
    np.save(path / "y.npy", np.array(y, dtype=np.int8))
    np.save(path / "patient_ids.npy", np.array(patient_ids, dtype=object), allow_pickle=True)
    (path / "metadata.json").write_text(
        json.dumps(
            {
                "label_source": "unit_label",
                "n_patients": rows,
                "input_dim": rows,
                "label_positive_rate_pct": round(sum(y) / rows * 100, 4),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_make_patient_disjoint_dataset_filters_overlap_and_records_distribution(tmp_path: Path) -> None:
    from scripts.ops.make_patient_disjoint_dataset import make_patient_disjoint_dataset

    source = tmp_path / "source"
    exclude = tmp_path / "exclude"
    output = tmp_path / "output"
    _write_dataset(source, [" 1 ", "P2", 3, "P4"], [1, 0, 1, 0])
    _write_dataset(exclude, ["1", " 3 "], [0, 1])

    metadata = make_patient_disjoint_dataset(source, exclude, output, min_positive=0)

    kept_ids = np.load(output / "patient_ids.npy", allow_pickle=True).tolist()
    X = sparse.load_npz(output / "X_csr.npz")
    y = np.load(output / "y.npy")
    written_metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))

    assert kept_ids == ["P2", "P4"]
    assert y.tolist() == [0, 0]
    assert X.shape == (2, 4)
    assert X[0, 1] == 2
    assert X[1, 3] == 4
    assert metadata == written_metadata
    assert metadata["source_n_patients"] == 4
    assert metadata["n_patients"] == 2
    assert metadata["excluded_patient_overlap_count"] == 2
    assert metadata["overlap_rate_in_source_pct"] == 50.0
    assert metadata["source_label_positive_rate_pct"] == 50.0
    assert metadata["excluded_positive_count"] == 2
    assert metadata["excluded_positive_rate_pct"] == 100.0
    assert metadata["label_positive"] == 0
    assert metadata["label_positive_rate_pct"] == 0.0
    assert metadata["density"] == 0.25
    assert metadata["sparsity_pct"] == 75.0
    assert metadata["zero_vector_patients"] == 0
    assert "unknown_drug_count" not in metadata
    assert metadata["patient_overlap_count_after_filter"] == 0
    assert set(metadata["artifact_sha256"]) == {"X_csr.npz", "y.npy", "patient_ids.npy"}


def test_make_patient_disjoint_dataset_rejects_row_mismatch(tmp_path: Path) -> None:
    import pytest

    from scripts.ops.make_patient_disjoint_dataset import make_patient_disjoint_dataset

    source = tmp_path / "source"
    exclude = tmp_path / "exclude"
    output = tmp_path / "output"
    _write_dataset(source, ["P1", "P2"], [1, 0])
    _write_dataset(exclude, ["P2"], [0])
    np.save(source / "y.npy", np.array([1], dtype=np.int8))

    with pytest.raises(ValueError, match="row alignment"):
        make_patient_disjoint_dataset(source, exclude, output)


def test_make_patient_disjoint_dataset_rejects_too_few_positive(tmp_path: Path) -> None:
    import pytest

    from scripts.ops.make_patient_disjoint_dataset import make_patient_disjoint_dataset

    source = tmp_path / "source"
    exclude = tmp_path / "exclude"
    output = tmp_path / "output"
    _write_dataset(source, ["P1", "P2", "P3"], [1, 0, 0])
    _write_dataset(exclude, ["P2"], [0])

    with pytest.raises(ValueError, match="positive"):
        make_patient_disjoint_dataset(source, exclude, output, min_positive=2)


def test_make_patient_disjoint_dataset_preserves_leading_zero_identity(tmp_path: Path) -> None:
    from scripts.ops.make_patient_disjoint_dataset import make_patient_disjoint_dataset

    source = tmp_path / "source"
    exclude = tmp_path / "exclude"
    output = tmp_path / "output"
    _write_dataset(source, ["00123", "123", "P9"], [1, 1, 0])
    _write_dataset(exclude, [123], [1])

    metadata = make_patient_disjoint_dataset(source, exclude, output, min_positive=1)

    kept_ids = np.load(output / "patient_ids.npy", allow_pickle=True).tolist()
    assert kept_ids == ["00123", "P9"]
    assert metadata["excluded_patient_overlap_count"] == 1
    assert metadata["patient_overlap_count_after_filter"] == 0
