"""Build sparse datasets for Oct feature -> Nov future outcome labels."""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import sparse

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.ops.full_cohort_history_loader import FullCohortHistoryLoader
from scripts.ops.eligibility_loader import DEFAULT_DEMOGRAPHICS_FEATURE, load_demographics
from scripts.ops.future_outcome_label import (
    FUTURE_MULTI_INSTITUTION_THRESHOLD,
    label_future_multi_institution_onset,
)
from scripts.ops.medication_class_features import (
    EFMDC_NULL_TOKEN,
    EFMDC_UNK_TOKEN,
    build_medication_class_vocab,
    patient_medication_class_pairs,
    read_medication_class_vocab,
    write_medication_class_vocab,
)


def build_future_outcome_sparse_dataset(
    oct_histories: pd.DataFrame,
    nov_histories: pd.DataFrame,
    patient_ids: Sequence[str],
    vocab: dict[str, int],
    *,
    threshold: int = FUTURE_MULTI_INSTITUTION_THRESHOLD,
    add_institution_count_feature: bool = False,
    demographics_features: Mapping[str, tuple[float, float]] | None = None,
    medication_class_vocab: Mapping[str, int] | None = None,
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

    demographics_start_index = len(vocab) + (1 if add_institution_count_feature else 0)
    add_demographics_feature = demographics_features is not None
    medication_class_start_index = demographics_start_index + (2 if add_demographics_feature else 0)
    add_medication_class_feature = medication_class_vocab is not None
    medication_class_feature_count = len(medication_class_vocab or {})
    input_dim = medication_class_start_index + medication_class_feature_count
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
    demographics_missing_patient_count = 0
    if add_demographics_feature and kept_patient_ids:
        assert demographics_features is not None
        age_column = demographics_start_index
        sex_column = demographics_start_index + 1
        age_values: list[float] = []
        sex_values: list[float] = []
        for patient_id in kept_patient_ids:
            if patient_id not in demographics_features:
                demographics_missing_patient_count += 1
            age_value, sex_value = demographics_features.get(patient_id, DEFAULT_DEMOGRAPHICS_FEATURE)
            age_values.append(float(age_value))
            sex_values.append(float(sex_value))
        X = X.tolil()
        X[:, age_column] = np.array(age_values, dtype=np.float32).reshape(-1, 1)
        X[:, sex_column] = np.array(sex_values, dtype=np.float32).reshape(-1, 1)
        X = X.tocsr()
    medication_class_stats = {
        "medication_class_total_rows": 0,
        "medication_class_null_row_count": 0,
        "medication_class_oov_row_count": 0,
    }
    if add_medication_class_feature and kept_patient_ids:
        assert medication_class_vocab is not None
        class_pairs, medication_class_stats = patient_medication_class_pairs(
            histories,
            kept_patient_ids,
            medication_class_vocab,
        )
        if class_pairs:
            X = X.tolil()
            for patient_id, class_index in class_pairs:
                X[patient_row[patient_id], medication_class_start_index + int(class_index)] = 1.0
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
        "add_demographics_feature": add_demographics_feature,
        "demographics_feature_indices": {
            "age_years_div_100": demographics_start_index,
            "sex_type_1_flag": demographics_start_index + 1,
        } if add_demographics_feature else None,
        "demographics_feature_source": "eligibility_demographics.parquet" if add_demographics_feature else None,
        "demographics_feature_normalization": "age uses reference_year - byear divided by 100; invalid byear falls back to eligibility age divided by 100; sex_type '1' -> 1.0, '2' -> 0.0, missing/other -> 0.5" if add_demographics_feature else None,
        "demographics_sex_semantics": "sex_type=1 -> 1.0, sex_type=2 -> 0.0, missing/other -> 0.5" if add_demographics_feature else None,
        "demographics_missing_patient_count": demographics_missing_patient_count if add_demographics_feature else None,
        "demographics_missing_patient_rate_pct": _pct(demographics_missing_patient_count, len(kept_patient_ids)) if add_demographics_feature else None,
        "add_medication_class_feature": add_medication_class_feature,
        "medication_class_feature_start_index": medication_class_start_index if add_medication_class_feature else None,
        "medication_class_feature_count": medication_class_feature_count if add_medication_class_feature else None,
        "medication_class_null_token": EFMDC_NULL_TOKEN if add_medication_class_feature else None,
        "medication_class_unknown_token": EFMDC_UNK_TOKEN if add_medication_class_feature else None,
        "medication_class_null_token_index": (
            medication_class_start_index + int(medication_class_vocab[EFMDC_NULL_TOKEN])
            if add_medication_class_feature and medication_class_vocab is not None else None
        ),
        "medication_class_unknown_token_index": (
            medication_class_start_index + int(medication_class_vocab[EFMDC_UNK_TOKEN])
            if add_medication_class_feature and medication_class_vocab is not None else None
        ),
        "medication_class_total_rows": int(medication_class_stats["medication_class_total_rows"]) if add_medication_class_feature else None,
        "medication_class_null_row_count": int(medication_class_stats["medication_class_null_row_count"]) if add_medication_class_feature else None,
        "medication_class_null_row_rate_pct": _pct(
            int(medication_class_stats["medication_class_null_row_count"]),
            int(medication_class_stats["medication_class_total_rows"]),
        ) if add_medication_class_feature else None,
        "medication_class_oov_row_count": int(medication_class_stats["medication_class_oov_row_count"]) if add_medication_class_feature else None,
        "medication_class_oov_row_rate_pct": _pct(
            int(medication_class_stats["medication_class_oov_row_count"]),
            int(medication_class_stats["medication_class_total_rows"]),
        ) if add_medication_class_feature else None,
        "medication_class_feature_semantics": "multi-hot efmdc_clsf_no class features from feature window only; blank/null rows map to __NULL_EFMDC__, unseen classes map to __UNK_EFMDC__" if add_medication_class_feature else None,
        "unknown_drug_count": unknown_drug_count,
        "total_drug_rows": total_drug_rows,
        "unknown_drug_rate_pct": _pct(unknown_drug_count, total_drug_rows),
        "feature_semantics": "Oct observation-window sparse drug_code multi-hot only; no Nov rows enter X",
        "label_semantics": "positive when oct_institution_count < T and nov_institution_count >= T",
        "onset_type_note": (
            "Positive cases are escalation when oct_institution_count is 1..T-1 and nov_institution_count >= T. "
            "Clean de-novo onset may be absent under strict oct_history_rows >= 1 observability."
        ),
        "temporal_holdout_status": "temporal holdout is available when a later feature/outcome pair is built",
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
    add_demographics_feature: bool = False,
    add_medication_class_feature: bool = False,
    medication_class_vocab_path: str | Path | None = None,
    medication_class_min_count: int = 1,
) -> dict:
    start = perf_counter()
    raw_path = Path(raw_dir)
    vocab_file = Path(vocab_path)
    vocab = json.loads(vocab_file.read_text(encoding="utf-8"))
    patient_ids = _sample_patient_ids(raw_path, feature_reference_date, max_patients)
    extra_columns = ["institution_id"]
    if add_medication_class_feature:
        extra_columns.append("efmdc_clsf_no")
    loader = FullCohortHistoryLoader(raw_path, extra_columns=extra_columns)
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
    demographics_features = (
        load_demographics(raw_path, patient_ids, reference_date=feature_reference_date)
        if add_demographics_feature
        else None
    )
    medication_class_vocab = None
    medication_class_vocab_metadata: dict = {}
    if add_medication_class_feature:
        if medication_class_vocab_path:
            medication_class_vocab = read_medication_class_vocab(medication_class_vocab_path)
            medication_class_vocab_metadata = {
                "medication_class_vocab_path": str(medication_class_vocab_path),
                "medication_class_vocab_source": "loaded_from_path",
                "medication_class_min_count": None,
                "medication_class_vocab_size": len(medication_class_vocab),
            }
        else:
            medication_class_vocab, medication_class_vocab_metadata = build_medication_class_vocab(
                oct_histories,
                min_count=medication_class_min_count,
            )
    X, y, kept_patient_ids, metadata = build_future_outcome_sparse_dataset(
        oct_histories,
        nov_histories,
        patient_ids,
        vocab,
        threshold=threshold,
        add_institution_count_feature=add_institution_count_feature,
        demographics_features=demographics_features,
        medication_class_vocab=medication_class_vocab,
    )
    medication_class_vocab_output_path = None
    if add_medication_class_feature and medication_class_vocab is not None:
        medication_class_vocab_output_path = Path(output_dir) / "medication_class_vocab.json"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        write_medication_class_vocab(medication_class_vocab_output_path, medication_class_vocab)
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
    if add_medication_class_feature:
        metadata.update(medication_class_vocab_metadata)
        metadata["medication_class_vocab_output_path"] = str(medication_class_vocab_output_path)
        if medication_class_vocab_output_path is not None:
            metadata["medication_class_vocab_sha256"] = _sha256(medication_class_vocab_output_path)
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
    parser.add_argument("--add-demographics-feature", action="store_true")
    parser.add_argument("--add-medication-class-feature", action="store_true")
    parser.add_argument("--medication-class-vocab-path", default=None)
    parser.add_argument("--medication-class-min-count", type=int, default=1)
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
        add_demographics_feature=args.add_demographics_feature,
        add_medication_class_feature=args.add_medication_class_feature,
        medication_class_vocab_path=args.medication_class_vocab_path,
        medication_class_min_count=args.medication_class_min_count,
    )
    print(f"[OK] wrote {args.output_dir}")
    print(f"n_patients={metadata['n_patients']}")
    print(f"label_positive_rate_pct={metadata['label_positive_rate_pct']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
