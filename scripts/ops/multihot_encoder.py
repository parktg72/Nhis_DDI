"""Multi-hot encoding helpers for drug history smoke validation."""
from __future__ import annotations

from datetime import date
from typing import Protocol, Sequence

import numpy as np
import pandas as pd


class HistoryBatchProvider(Protocol):
    def get_history_batch(
        self,
        patient_ids: Sequence[str],
        reference_date: date,
        lookback_days: int = 60,
    ) -> pd.DataFrame:
        ...


def encode_patient_history(
    history_df: pd.DataFrame,
    vocab: dict[str, int],
) -> np.ndarray:
    _validate_vocab(vocab)
    vector = np.zeros(len(vocab), dtype=np.float32)
    if history_df.empty or "drug_code" not in history_df.columns:
        return vector

    unk_index = vocab["_unk"]
    for code in _normalized_drug_codes(history_df["drug_code"]):
        vector[vocab.get(code, unk_index)] = 1.0
    return vector


def encode_batch(
    provider: HistoryBatchProvider,
    patient_ids: Sequence[str],
    vocab: dict[str, int],
    reference_date: date,
    lookback_days: int = 60,
) -> tuple[np.ndarray, dict]:
    _validate_vocab(vocab)
    patient_id_list = [str(patient_id) for patient_id in patient_ids]
    matrix = np.zeros((len(patient_id_list), len(vocab)), dtype=np.float32)
    histories = provider.get_history_batch(
        patient_id_list,
        reference_date=reference_date,
        lookback_days=lookback_days,
    )
    if histories.empty:
        return matrix, _batch_stats(matrix, vocab, total_unk_prescriptions=0)

    total_unk_prescriptions = 0
    unk_index = vocab["_unk"]
    histories = histories.copy()
    histories["patient_id"] = histories["patient_id"].astype(str)
    grouped = {patient_id: group for patient_id, group in histories.groupby("patient_id")}
    for row_index, patient_id in enumerate(patient_id_list):
        patient_history = grouped.get(patient_id, pd.DataFrame(columns=histories.columns))
        matrix[row_index] = encode_patient_history(patient_history, vocab)
        total_unk_prescriptions += _unknown_prescription_count(patient_history, vocab)

    stats = _batch_stats(
        matrix,
        vocab,
        total_unk_prescriptions=total_unk_prescriptions,
    )
    stats["unk_flag_patients"] = int(matrix[:, unk_index].sum()) if len(matrix) else 0
    stats["unk_flag_rate_pct"] = _pct(stats["unk_flag_patients"], len(patient_id_list))
    return matrix, stats


def _normalized_drug_codes(values: pd.Series) -> list[str]:
    cleaned = values.dropna().astype(str).str.strip()
    return [value for value in cleaned.tolist() if value]


def _validate_vocab(vocab: dict[str, int]) -> None:
    if "_unk" not in vocab:
        raise ValueError("vocab must include '_unk' index")


def _unknown_prescription_count(history_df: pd.DataFrame, vocab: dict[str, int]) -> int:
    if history_df.empty or "drug_code" not in history_df.columns:
        return 0
    return sum(1 for code in _normalized_drug_codes(history_df["drug_code"]) if code not in vocab)


def _batch_stats(
    matrix: np.ndarray,
    vocab: dict[str, int],
    *,
    total_unk_prescriptions: int,
) -> dict:
    n_patients = int(matrix.shape[0])
    input_dim = int(matrix.shape[1])
    nonzero_counts = matrix.sum(axis=1) if n_patients else np.array([], dtype=np.float32)
    unk_index = vocab["_unk"]
    unk_flags = matrix[:, unk_index] if n_patients else np.array([], dtype=np.float32)
    known_counts = nonzero_counts - unk_flags
    return {
        "n_patients": n_patients,
        "input_dim": input_dim,
        "density_mean": round(float(nonzero_counts.mean() / input_dim), 8)
        if n_patients and input_dim
        else 0.0,
        "density_p95": round(float(np.percentile(nonzero_counts / input_dim, 95)), 8)
        if n_patients and input_dim
        else 0.0,
        "known_bits_mean": round(float(known_counts.mean()), 4) if n_patients else 0.0,
        "unk_flag_patients": int(unk_flags.sum()) if n_patients else 0,
        "unk_flag_rate_pct": _pct(int(unk_flags.sum()), n_patients),
        "zero_vector_patients": int((nonzero_counts == 0).sum()) if n_patients else 0,
        "zero_vector_rate_pct": _pct(int((nonzero_counts == 0).sum()), n_patients)
        if n_patients
        else 0.0,
        "total_unk_prescriptions": int(total_unk_prescriptions),
    }


def _pct(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round((numerator / denominator) * 100, 4)
