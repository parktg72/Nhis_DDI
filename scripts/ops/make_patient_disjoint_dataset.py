"""Create a patient-disjoint sparse validation dataset from existing artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
from scipy import sparse

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def make_patient_disjoint_dataset(
    source_dataset_dir: str | Path,
    exclude_patient_ids_from: str | Path,
    output_dir: str | Path,
    *,
    min_positive: int = 1,
) -> dict:
    source_path = Path(source_dataset_dir)
    exclude_path = Path(exclude_patient_ids_from)
    output_path = Path(output_dir)

    X, y, patient_ids, source_metadata = _load_dataset(source_path)
    _, _, exclude_ids, _ = _load_dataset(exclude_path, load_matrix=False)
    _validate_alignment(X, y, patient_ids, context="source row alignment")

    normalized_ids = np.array([_normalize_patient_id(value) for value in patient_ids], dtype=object)
    exclude_set = {_normalize_patient_id(value) for value in exclude_ids}
    keep_mask = np.array([patient_id not in exclude_set for patient_id in normalized_ids], dtype=bool)
    excluded_mask = ~keep_mask

    X_filtered = X[keep_mask].tocsr()
    y_filtered = y[keep_mask]
    patient_ids_filtered = normalized_ids[keep_mask]
    _validate_alignment(X_filtered, y_filtered, patient_ids_filtered, context="filtered row alignment")

    overlap_after = int(np.intersect1d(patient_ids_filtered.astype(str), np.array(sorted(exclude_set), dtype=str)).size)
    if overlap_after != 0:
        raise ValueError("patient overlap remains after filtering")

    positive_count = int(y_filtered.sum())
    if positive_count < min_positive:
        raise ValueError(f"filtered dataset positive count {positive_count} is below min_positive={min_positive}")

    output_path.mkdir(parents=True, exist_ok=True)
    x_path = output_path / "X_csr.npz"
    y_path = output_path / "y.npy"
    patient_path = output_path / "patient_ids.npy"
    metadata_path = output_path / "metadata.json"
    sparse.save_npz(x_path, X_filtered)
    np.save(y_path, np.asarray(y_filtered, dtype=np.int8))
    np.save(patient_path, patient_ids_filtered, allow_pickle=True)

    excluded_y = y[excluded_mask]
    density = _density(X_filtered)
    row_nnz = np.asarray(X_filtered.getnnz(axis=1))
    metadata = {
        **_stable_source_metadata(source_metadata),
        "source_dataset_dir": str(source_path),
        "exclude_dataset_dir": str(exclude_path),
        "patient_disjoint_from": str(exclude_path),
        "source_n_patients": int(len(patient_ids)),
        "source_label_positive": int(y.sum()),
        "source_label_positive_rate_pct": _pct(int(y.sum()), len(y)),
        "excluded_patient_overlap_count": int(excluded_mask.sum()),
        "overlap_rate_in_source_pct": _pct(int(excluded_mask.sum()), len(patient_ids)),
        "excluded_positive_count": int(excluded_y.sum()),
        "excluded_positive_rate_pct": _pct(int(excluded_y.sum()), len(excluded_y)),
        "n_patients": int(len(patient_ids_filtered)),
        "input_dim": int(X_filtered.shape[1]),
        "nnz": int(X_filtered.nnz),
        "density": round(density, 10),
        "sparsity_pct": round((1.0 - density) * 100.0, 6) if X_filtered.shape[0] and X_filtered.shape[1] else 0.0,
        "zero_vector_patients": int((row_nnz == 0).sum()) if len(row_nnz) else 0,
        "zero_vector_rate_pct": _pct(int((row_nnz == 0).sum()), len(row_nnz)) if len(row_nnz) else 0.0,
        "label_positive": positive_count,
        "label_positive_rate_pct": _pct(positive_count, len(y_filtered)),
        "patient_overlap_count_after_filter": overlap_after,
        "artifact_sha256": {
            "X_csr.npz": _sha256(x_path),
            "y.npy": _sha256(y_path),
            "patient_ids.npy": _sha256(patient_path),
        },
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def _stable_source_metadata(metadata: dict) -> dict:
    row_dependent_keys = {
        "n_patients",
        "nnz",
        "density",
        "sparsity_pct",
        "label_positive",
        "label_positive_rate_pct",
        "unknown_drug_count",
        "unknown_drug_rate_pct",
        "unk_flag_patients",
        "unk_flag_rate_pct",
        "zero_vector_patients",
        "zero_vector_rate_pct",
        "artifact_sha256",
    }
    return {key: value for key, value in metadata.items() if key not in row_dependent_keys}


def _load_dataset(
    dataset_path: Path,
    *,
    load_matrix: bool = True,
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray, dict]:
    X = sparse.load_npz(dataset_path / "X_csr.npz").tocsr() if load_matrix else sparse.csr_matrix((0, 0))
    y = np.load(dataset_path / "y.npy") if load_matrix else np.array([], dtype=np.int8)
    patient_ids = np.load(dataset_path / "patient_ids.npy", allow_pickle=True)
    metadata_path = dataset_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    return X, y, patient_ids, metadata


def _validate_alignment(X: sparse.csr_matrix, y: np.ndarray, patient_ids: np.ndarray, *, context: str) -> None:
    if X.shape[0] != len(y) or len(y) != len(patient_ids):
        raise ValueError(
            f"{context}: X rows ({X.shape[0]}), y rows ({len(y)}), "
            f"patient_ids rows ({len(patient_ids)}) must match"
        )


def _normalize_patient_id(value: object) -> str:
    return str(value).strip()


def _pct(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(float(numerator) / float(denominator) * 100.0, 4)


def _density(X: sparse.csr_matrix) -> float:
    rows, cols = X.shape
    if rows == 0 or cols == 0:
        return 0.0
    return float(X.nnz) / float(rows * cols)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a patient-disjoint sparse dataset.")
    parser.add_argument("--source-dataset-dir", required=True)
    parser.add_argument("--exclude-patient-ids-from", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-positive", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metadata = make_patient_disjoint_dataset(
        args.source_dataset_dir,
        args.exclude_patient_ids_from,
        args.output_dir,
        min_positive=args.min_positive,
    )
    print(f"[OK] wrote {args.output_dir}")
    print(f"n_patients={metadata['n_patients']}")
    print(f"excluded_patient_overlap_count={metadata['excluded_patient_overlap_count']}")
    print(f"label_positive_rate_pct={metadata['label_positive_rate_pct']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
