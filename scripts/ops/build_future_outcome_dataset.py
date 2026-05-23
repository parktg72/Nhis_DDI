"""Build sparse datasets for Oct feature -> Nov future outcome labels."""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import sparse

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.ops.full_cohort_history_loader import FullCohortHistoryLoader
from scripts.ops.future_outcome_label import (
    FUTURE_MULTI_INSTITUTION_THRESHOLD,
    label_future_multi_institution_onset,
)


def build_future_outcome_sparse_dataset(
    oct_histories: pd.DataFrame,
    nov_histories: pd.DataFrame,
    patient_ids: Sequence[str],
    vocab: dict[str, int],
    *,
    threshold: int = FUTURE_MULTI_INSTITUTION_THRESHOLD,
    add_institution_count_feature: bool = False,
) -> tuple[sparse.csr_matrix, np.ndarray, list[str], dict]:
    _validate_vocab(vocab)
    patient_id_list = [str(patient_id) for patient_id in patient_ids]
    label_result = label_future_multi_institution_onset(
        patient_id_list,
        oct_histories,
        nov_histories,
        threshold=threshold,
    )
    kept_patient_ids = [
        patient_id
        for patient_id in patient_id_list
        if patient_id in label_result.labels
    ]
    patient_row = {patient_id: row for row, patient_id in enumerate(kept_patient_ids)}
    histories = _normalize_oct_histories(oct_histories)

    row_indices: list[int] = []
    col_indices: list[int] = []
    seen_pairs: set[tuple[int, int]] = set()
    unknown_drug_count = 0
    total_drug_rows = 0
    if not histories.empty and kept_patient_ids:
        for patient_id, drug_code in histories[["patient_id", "drug_code"]].itertuples(index=False):
            if patient_id not in patient_row or not drug_code:
                continue
            total_drug_rows += 1
            column = vocab.get(drug_code)
            if column is None:
                column = vocab["_unk"]
                unknown_drug_count += 1
            pair = (patient_row[patient_id], int(column))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            row_indices.append(pair[0])
            col_indices.append(pair[1])

    input_dim = len(vocab) + (1 if add_institution_count_feature else 0)
    if add_institution_count_feature and kept_patient_ids:
        max_oct_count = max(label_result.oct_institution_counts[patient_id] for patient_id in kept_patient_ids)
        denominator = max(max_oct_count, 1)
        institution_column = len(vocab)
        for patient_id in kept_patient_ids:
            value = label_result.oct_institution_counts[patient_id] / denominator
            if value == 0:
                continue
            row_indices.append(patient_row[patient_id])
            col_indices.append(institution_column)

    X = sparse.csr_matrix(
        (
            np.ones(len(row_indices), dtype=np.float32),
            (row_indices, col_indices),
        ),
        shape=(len(kept_patient_ids), input_dim),
        dtype=np.float32,
    )
    if add_institution_count_feature and kept_patient_ids:
        institution_column = len(vocab)
        denominator = max(max(label_result.oct_institution_counts[patient_id] for patient_id in kept_patient_ids), 1)
        values = [
            label_result.oct_institution_counts[patient_id] / denominator
            for patient_id in kept_patient_ids
        ]
        X = X.tolil()
        X[:, institution_column] = np.array(values, dtype=np.float32).reshape(-1, 1)
        X = X.tocsr()
    y = np.array(
        [label_result.labels[patient_id] for patient_id in kept_patient_ids],
        dtype=np.int8,
    )
    metadata = {
        "label_source": "future_multi_institution_onset",
        "threshold": threshold,
        "n_feature_cohort": label_result.n_patients,
        "n_patients": len(kept_patient_ids),
        "label_positive": int(y.sum()),
        "label_positive_rate_pct": _pct(int(y.sum()), len(y)),
        "n_censored": label_result.n_censored,
        "censoring_rate_pct": _pct(label_result.n_censored, label_result.n_patients),
        "onset_eligible_n": label_result.onset_eligible_n,
        "persistence_excluded_count": label_result.persistence_excluded_count,
        "persistence_cohort_size": label_result.persistence_cohort_size,
        "persistence_rate_pct": label_result.persistence_rate_pct,
        "oct_history_zero_excluded": label_result.oct_history_zero_excluded,
        "clean_onset_positive": label_result.clean_onset_positive,
        "escalation_positive": label_result.escalation_positive,
        "input_dim": input_dim,
        "nnz": int(X.nnz),
        "add_institution_count_feature": add_institution_count_feature,
        "institution_count_feature_index": len(vocab) if add_institution_count_feature else None,
        "institution_count_feature_normalization": "oct_institution_count / max(oct_institution_count in kept dataset)" if add_institution_count_feature else None,
        "unknown_drug_count": unknown_drug_count,
        "total_drug_rows": total_drug_rows,
        "unknown_drug_rate_pct": _pct(unknown_drug_count, total_drug_rows),
        "feature_semantics": "Oct observation-window sparse drug_code multi-hot only; no Nov rows enter X",
        "label_semantics": "positive when oct_institution_count < T and nov_institution_count >= T",
        "onset_type_note": (
            "Positive cases are escalation when oct_institution_count is 1..T-1 and nov_institution_count >= T. "
            "Clean de-novo onset may be absent under strict oct_history_rows >= 1 observability."
        ),
        "no_third_month_caveat": "2024-12 Raw is unavailable; random split training is internal feasibility only",
    }
    return X, y, kept_patient_ids, metadata


def build_future_outcome_dataset_from_raw(
    raw_dir: str | Path,
    vocab_path: str | Path,
    output_dir: str | Path,
    *,
    feature_reference_date: date = date(2024, 10, 31),
    outcome_reference_date: date = date(2024, 11, 30),
    lookback_days: int = 29,
    threshold: int = FUTURE_MULTI_INSTITUTION_THRESHOLD,
    max_patients: int | None = None,
    add_institution_count_feature: bool = False,
) -> dict:
    start = perf_counter()
    raw_path = Path(raw_dir)
    vocab_file = Path(vocab_path)
    vocab = json.loads(vocab_file.read_text(encoding="utf-8"))
    patient_ids = _sample_patient_ids(raw_path, feature_reference_date, max_patients)
    loader = FullCohortHistoryLoader(raw_path, extra_columns=["institution_id"])
    oct_histories = loader.load_window(
        reference_date=feature_reference_date,
        lookback_days=lookback_days,
        patient_ids=patient_ids,
    )
    oct_loaded_file_count = loader.last_loaded_file_count
    nov_histories = loader.load_window(
        reference_date=outcome_reference_date,
        lookback_days=lookback_days,
        patient_ids=patient_ids,
    )
    nov_loaded_file_count = loader.last_loaded_file_count
    X, y, kept_patient_ids, metadata = build_future_outcome_sparse_dataset(
        oct_histories,
        nov_histories,
        patient_ids,
        vocab,
        threshold=threshold,
        add_institution_count_feature=add_institution_count_feature,
    )
    metadata.update({
        "feature_reference_date": feature_reference_date.isoformat(),
        "outcome_reference_date": outcome_reference_date.isoformat(),
        "lookback_days": lookback_days,
        "feature_window": _window_dict(feature_reference_date, lookback_days),
        "outcome_window": _window_dict(outcome_reference_date, lookback_days),
        "raw_dir": str(raw_path),
        "feature_loaded_file_count": oct_loaded_file_count,
        "outcome_loaded_file_count": nov_loaded_file_count,
        "max_patients": max_patients,
        "vocab_path": str(vocab_file),
        "vocab_sha256": _sha256(vocab_file),
        "build_time_sec": round(perf_counter() - start, 3),
    })
    result = write_future_outcome_dataset(X, y, kept_patient_ids, metadata, output_dir)
    return result


def write_future_outcome_dataset(
    X: sparse.csr_matrix,
    y: np.ndarray,
    patient_ids: Sequence[str],
    metadata: dict,
    output_dir: str | Path,
) -> dict:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    x_path = output_path / "X_csr.npz"
    y_path = output_path / "y.npy"
    patient_path = output_path / "patient_ids.npy"
    metadata_path = output_path / "metadata.json"
    sparse.save_npz(x_path, X)
    np.save(y_path, np.asarray(y, dtype=np.int8))
    np.save(patient_path, np.array([str(patient_id) for patient_id in patient_ids], dtype=object), allow_pickle=True)
    final_metadata = {
        **metadata,
        "n_patients": int(X.shape[0]),
        "input_dim": int(X.shape[1]),
        "nnz": int(X.nnz),
        "label_positive": int(np.asarray(y).sum()),
        "label_positive_rate_pct": _pct(int(np.asarray(y).sum()), len(y)),
    }
    final_metadata["artifact_sha256"] = {
        "X_csr.npz": _sha256(x_path),
        "y.npy": _sha256(y_path),
        "patient_ids.npy": _sha256(patient_path),
    }
    metadata_path.write_text(json.dumps(final_metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return final_metadata


def _normalize_oct_histories(histories: pd.DataFrame) -> pd.DataFrame:
    if histories.empty:
        return pd.DataFrame(columns=["patient_id", "drug_code"])
    normalized = histories.copy()
    normalized["patient_id"] = normalized["patient_id"].astype(str)
    normalized["drug_code"] = normalized["drug_code"].where(normalized["drug_code"].notna(), "")
    normalized["drug_code"] = normalized["drug_code"].astype(str).str.strip()
    return normalized


def _validate_vocab(vocab: dict[str, int]) -> None:
    if "_unk" not in vocab:
        raise ValueError("vocab must include _unk token")
    values = sorted(vocab.values())
    expected = list(range(len(vocab)))
    if values != expected:
        raise ValueError("vocab indices must be contiguous from 0")


def _sample_patient_ids(raw_dir: Path, reference_date: date, max_patients: int | None) -> list[str]:
    path = raw_dir / f"records_{reference_date.strftime('%Y%m%d')}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"reference records file not found: {path}")
    df = pd.read_parquet(path, columns=["patient_id"])
    patient_ids = df["patient_id"].dropna().astype(str).drop_duplicates().tolist()
    if max_patients is not None:
        return patient_ids[:max_patients]
    return patient_ids


def _window_dict(reference_date: date, lookback_days: int) -> dict[str, str]:
    start = reference_date - timedelta(days=lookback_days)
    return {"start": start.isoformat(), "end": reference_date.isoformat()}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y%m%d").date()


def _pct(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round((numerator / denominator) * 100, 4)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build future outcome sparse dataset.")
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--vocab-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--feature-reference-date", default="20241031")
    parser.add_argument("--outcome-reference-date", default="20241130")
    parser.add_argument("--lookback-days", type=int, default=29)
    parser.add_argument("--threshold", type=int, default=FUTURE_MULTI_INSTITUTION_THRESHOLD)
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--add-institution-count-feature", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metadata = build_future_outcome_dataset_from_raw(
        args.raw_dir,
        args.vocab_path,
        args.output_dir,
        feature_reference_date=_parse_date(args.feature_reference_date),
        outcome_reference_date=_parse_date(args.outcome_reference_date),
        lookback_days=args.lookback_days,
        threshold=args.threshold,
        max_patients=args.max_patients,
        add_institution_count_feature=args.add_institution_count_feature,
    )
    print(f"[OK] wrote {args.output_dir}")
    print(f"n_patients={metadata['n_patients']}")
    print(f"label_positive_rate_pct={metadata['label_positive_rate_pct']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
