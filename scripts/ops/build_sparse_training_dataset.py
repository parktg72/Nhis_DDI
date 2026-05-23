"""Build sparse multi-hot training datasets for ops-scale smoke runs."""
from __future__ import annotations

import argparse
from datetime import date, datetime
import hashlib
import json
import os
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
from scripts.ops.multi_institution_label import (
    MULTI_INSTITUTION_THRESHOLD,
    label_patient_histories as label_multi_institution_histories,
)
from scripts.ops.therapeutic_duplication_label import (
    THERAPEUTIC_DUP_THRESHOLD,
    label_therapeutic_duplication,
)


def build_sparse_dataset(
    histories: pd.DataFrame,
    patient_ids: Sequence[str],
    vocab: dict[str, int],
    *,
    label_source: str = "therapeutic_duplication",
    therapeutic_dup_threshold: int = THERAPEUTIC_DUP_THRESHOLD,
    multi_institution_threshold: int = MULTI_INSTITUTION_THRESHOLD,
) -> tuple[sparse.csr_matrix, np.ndarray, dict]:
    _validate_vocab(vocab)
    patient_id_list = [str(patient_id) for patient_id in patient_ids]
    patient_row = {patient_id: row for row, patient_id in enumerate(patient_id_list)}
    input_dim = len(vocab)

    histories = _normalize_histories(histories)
    row_indices: list[int] = []
    col_indices: list[int] = []
    unknown_drug_count = 0
    total_drug_rows = 0

    if not histories.empty:
        seen_pairs: set[tuple[int, int]] = set()
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

    data = np.ones(len(row_indices), dtype=np.float32)
    X = sparse.csr_matrix(
        (data, (row_indices, col_indices)),
        shape=(len(patient_id_list), input_dim),
        dtype=np.float32,
    )
    labels, label_metadata = _build_labels(
        patient_id_list,
        histories,
        label_source=label_source,
        therapeutic_dup_threshold=therapeutic_dup_threshold,
        multi_institution_threshold=multi_institution_threshold,
    )
    y = np.array([labels[patient_id] for patient_id in patient_id_list], dtype=np.int8)
    stats = _sparse_stats(
        X,
        y,
        vocab,
        unknown_drug_count=unknown_drug_count,
        total_drug_rows=total_drug_rows,
    )
    stats.update(label_metadata)
    return X, y, stats


def build_sparse_dataset_from_raw(
    raw_dir: str | Path,
    vocab_path: str | Path,
    output_dir: str | Path,
    *,
    reference_date: date | None = None,
    lookback_days: int = 60,
    max_patients: int | None = None,
    batch_size: int = 5000,
    label_source: str = "therapeutic_duplication",
    therapeutic_dup_threshold: int = THERAPEUTIC_DUP_THRESHOLD,
    multi_institution_threshold: int = MULTI_INSTITUTION_THRESHOLD,
) -> dict:
    start = perf_counter()
    raw_path = Path(raw_dir)
    vocab_file = Path(vocab_path)
    vocab = json.loads(vocab_file.read_text(encoding="utf-8"))
    resolved_date = reference_date or _latest_records_date(raw_path)
    patient_ids = _sample_patient_ids(raw_path, resolved_date, max_patients)
    loader = FullCohortHistoryLoader(raw_path, extra_columns=[_extra_column_for_label_source(label_source)])
    histories = loader.load_window(
        reference_date=resolved_date,
        lookback_days=lookback_days,
        patient_ids=patient_ids,
    )

    X, y, stats = build_sparse_dataset(
        histories,
        patient_ids,
        vocab,
        label_source=label_source,
        therapeutic_dup_threshold=therapeutic_dup_threshold,
        multi_institution_threshold=multi_institution_threshold,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    x_path = output_path / "X_csr.npz"
    y_path = output_path / "y.npy"
    patient_path = output_path / "patient_ids.npy"
    metadata_path = output_path / "metadata.json"
    sparse.save_npz(x_path, X)
    np.save(y_path, y)
    np.save(patient_path, np.array(patient_ids, dtype=object), allow_pickle=True)

    metadata = {
        "reference_date": resolved_date.isoformat(),
        "lookback_days": lookback_days,
        "raw_date_range": _raw_date_range(loader, resolved_date, lookback_days),
        "raw_loaded_file_count": loader.last_loaded_file_count,
        "max_patients": max_patients,
        "batch_size": batch_size,
        "vocab_path": str(vocab_file),
        "vocab_sha256": _sha256(vocab_file),
        "build_time_sec": round(perf_counter() - start, 3),
        "peak_rss_mb": _peak_rss_mb(),
        **stats,
    }
    metadata["artifact_sha256"] = {
        "X_csr.npz": _sha256(x_path),
        "y.npy": _sha256(y_path),
        "patient_ids.npy": _sha256(patient_path),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def _build_labels(
    patient_ids: Sequence[str],
    histories: pd.DataFrame,
    *,
    label_source: str,
    therapeutic_dup_threshold: int,
    multi_institution_threshold: int,
) -> tuple[dict[str, int], dict]:
    if label_source == "therapeutic_duplication":
        result = label_therapeutic_duplication(
            patient_ids,
            histories,
            min_duplicate_classes=therapeutic_dup_threshold,
        )
        return result.labels, {
            "label_source": "therapeutic_duplication",
            "therapeutic_dup_threshold": therapeutic_dup_threshold,
            "label_semantics": "therapeutic duplication proxy: same efmdc_clsf_no with distinct drug_code count >= 2; label positive when duplicate class count >= threshold",
            "null_efmdc_row_count": result.null_efmdc_row_count,
            "evaluable_patient_count": result.evaluable_patient_count,
        }
    if label_source == "multi_institution":
        result = label_multi_institution_histories(
            patient_ids,
            histories,
            threshold=multi_institution_threshold,
        )
        return result.labels, {
            "label_source": "multi_institution",
            "multi_institution_threshold": multi_institution_threshold,
            "label_semantics": "multi-institution proxy: distinct institution_id count >= threshold within lookback window",
            "null_institution_count": result.null_institution_count,
        }
    raise ValueError(f"unsupported label_source: {label_source}")


def _extra_column_for_label_source(label_source: str) -> str:
    if label_source == "therapeutic_duplication":
        return "efmdc_clsf_no"
    if label_source == "multi_institution":
        return "institution_id"
    raise ValueError(f"unsupported label_source: {label_source}")


def _normalize_histories(histories: pd.DataFrame) -> pd.DataFrame:
    if histories.empty:
        return pd.DataFrame(columns=["patient_id", "drug_code", "efmdc_clsf_no"])
    normalized = histories.copy()
    normalized["patient_id"] = normalized["patient_id"].astype(str)
    normalized["drug_code"] = normalized["drug_code"].where(normalized["drug_code"].notna(), "")
    normalized["drug_code"] = normalized["drug_code"].astype(str).str.strip()
    return normalized


def _sparse_stats(
    X: sparse.csr_matrix,
    y: np.ndarray,
    vocab: dict[str, int],
    *,
    unknown_drug_count: int,
    total_drug_rows: int,
) -> dict:
    n_patients, input_dim = X.shape
    row_nnz = np.asarray(X.getnnz(axis=1))
    unk_flags = np.asarray(X[:, vocab["_unk"]].toarray()).reshape(-1) if n_patients else np.array([])
    density = float(X.nnz / (n_patients * input_dim)) if n_patients and input_dim else 0.0
    label_positive = int(y.sum())
    return {
        "n_patients": int(n_patients),
        "input_dim": int(input_dim),
        "nnz": int(X.nnz),
        "density": round(density, 10),
        "sparsity_pct": round((1.0 - density) * 100, 6) if n_patients and input_dim else 0.0,
        "label_positive": label_positive,
        "label_positive_rate_pct": _pct(label_positive, n_patients),
        "unknown_drug_count": int(unknown_drug_count),
        "unknown_drug_rate_pct": _pct(unknown_drug_count, total_drug_rows),
        "unk_flag_patients": int(unk_flags.sum()) if len(unk_flags) else 0,
        "unk_flag_rate_pct": _pct(int(unk_flags.sum()), n_patients),
        "zero_vector_patients": int((row_nnz == 0).sum()) if n_patients else 0,
        "zero_vector_rate_pct": _pct(int((row_nnz == 0).sum()), n_patients),
    }


def _validate_vocab(vocab: dict[str, int]) -> None:
    if "_unk" not in vocab:
        raise ValueError("vocab must include '_unk' index")
    values = sorted(int(value) for value in vocab.values())
    if values != list(range(len(values))):
        raise ValueError("vocab indices must be contiguous from 0")


def _sample_patient_ids(raw_dir: Path, reference_date: date, max_patients: int | None) -> list[str]:
    path = raw_dir / f"records_{reference_date.strftime('%Y%m%d')}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"reference records file not found: {path}")
    df = pd.read_parquet(path, columns=["patient_id"])
    ids = df["patient_id"].dropna().astype(str).drop_duplicates().tolist()
    return ids if max_patients is None else ids[:max_patients]


def _ordered_patient_ids(histories: pd.DataFrame) -> list[str]:
    if histories.empty:
        return []
    return histories["patient_id"].astype(str).drop_duplicates().tolist()


def _latest_records_date(raw_dir: Path) -> date:
    dates = []
    for path in raw_dir.glob("records_*.parquet"):
        try:
            dates.append(datetime.strptime(path.stem.removeprefix("records_"), "%Y%m%d").date())
        except ValueError:
            continue
    if not dates:
        raise FileNotFoundError(f"no records_YYYYMMDD.parquet files found in {raw_dir}")
    return max(dates)


def _raw_date_range(loader: FullCohortHistoryLoader, reference_date: date, lookback_days: int) -> list[str | None]:
    paths = loader._paths_for_window(reference_date, lookback_days)
    dates = [loader._date_from_path(path) for path in paths]
    dates = [value for value in dates if value is not None]
    if not dates:
        return [None, None]
    return [min(dates).isoformat(), max(dates).isoformat()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _peak_rss_mb() -> float:
    if os.name == "nt":
        return _windows_peak_working_set_mb()
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        raw = float(usage.ru_maxrss)
        if sys.platform == "darwin":
            return round(raw / (1024 * 1024), 3)
        return round(raw / 1024, 3)
    except Exception:
        return 0.0


def _windows_peak_working_set_mb() -> float:
    try:
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        psapi = ctypes.WinDLL("psapi.dll")
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        ok = psapi.GetProcessMemoryInfo(
            handle,
            ctypes.byref(counters),
            counters.cb,
        )
        if not ok:
            return 0.0
        return round(float(counters.PeakWorkingSetSize) / (1024 * 1024), 3)
    except Exception:
        return 0.0


def _parse_date(value: str | None) -> date | None:
    if value is None:
        return None
    return datetime.strptime(value, "%Y%m%d").date()


def _pct(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round((numerator / denominator) * 100, 4)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build sparse training dataset.")
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--vocab-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-date", default=None)
    parser.add_argument("--lookback-days", type=int, default=60)
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--label-source", choices=["therapeutic_duplication", "multi_institution"], default="therapeutic_duplication")
    parser.add_argument("--therapeutic-dup-threshold", type=int, default=THERAPEUTIC_DUP_THRESHOLD)
    parser.add_argument("--multi-institution-threshold", type=int, default=MULTI_INSTITUTION_THRESHOLD)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metadata = build_sparse_dataset_from_raw(
        args.raw_dir,
        args.vocab_path,
        args.output_dir,
        reference_date=_parse_date(args.reference_date),
        lookback_days=args.lookback_days,
        max_patients=args.max_patients,
        batch_size=args.batch_size,
        label_source=args.label_source,
        therapeutic_dup_threshold=args.therapeutic_dup_threshold,
        multi_institution_threshold=args.multi_institution_threshold,
    )
    print(f"[OK] wrote {Path(args.output_dir) / 'X_csr.npz'}")
    print(f"[OK] wrote {Path(args.output_dir) / 'y.npy'}")
    print(f"[OK] wrote {Path(args.output_dir) / 'patient_ids.npy'}")
    print(f"[OK] wrote {Path(args.output_dir) / 'metadata.json'}")
    print(f"n_patients={metadata['n_patients']} label_positive_rate_pct={metadata['label_positive_rate_pct']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
