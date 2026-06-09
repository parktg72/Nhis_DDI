"""Medication-class feature helpers for sparse future-outcome datasets."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd


EFMDC_NULL_TOKEN = "__NULL_EFMDC__"
EFMDC_UNK_TOKEN = "__UNK_EFMDC__"


def build_medication_class_vocab(
    histories: pd.DataFrame,
    *,
    min_count: int = 1,
) -> tuple[dict[str, int], dict]:
    if min_count < 1:
        raise ValueError("min_count must be >= 1")
    classes = _normalized_class_series(histories)
    nonblank = classes[classes != ""]
    counts = nonblank.value_counts()
    kept_classes = sorted(str(class_code) for class_code, count in counts.items() if int(count) >= min_count)
    vocab = {
        EFMDC_NULL_TOKEN: 0,
        EFMDC_UNK_TOKEN: 1,
        **{class_code: index + 2 for index, class_code in enumerate(kept_classes)},
    }
    metadata = {
        "medication_class_vocab_size": len(vocab),
        "medication_class_nonblank_unique_count": int(counts.shape[0]),
        "medication_class_min_count": int(min_count),
        "medication_class_dropped_rare_count": int(sum(count for count in counts.values if count < min_count)),
        "medication_class_vocab_source": "feature_window_unique_efmdc_clsf_no",
        "medication_class_null_token": EFMDC_NULL_TOKEN,
        "medication_class_unknown_token": EFMDC_UNK_TOKEN,
    }
    return vocab, metadata


def patient_medication_class_pairs(
    histories: pd.DataFrame,
    patient_ids: Sequence[str],
    vocab: Mapping[str, int],
) -> tuple[set[tuple[str, int]], dict]:
    _validate_class_vocab(vocab)
    patient_id_set = {str(patient_id) for patient_id in patient_ids}
    pairs: set[tuple[str, int]] = set()
    stats = {
        "medication_class_total_rows": 0,
        "medication_class_null_row_count": 0,
        "medication_class_oov_row_count": 0,
    }
    if histories.empty or "efmdc_clsf_no" not in histories.columns:
        return pairs, stats

    normalized = histories.copy()
    normalized["patient_id"] = normalized["patient_id"].astype(str)
    normalized["efmdc_class"] = _normalized_class_series(normalized)
    for patient_id, class_code in normalized[["patient_id", "efmdc_class"]].itertuples(index=False):
        if patient_id not in patient_id_set:
            continue
        stats["medication_class_total_rows"] += 1
        if class_code == "":
            stats["medication_class_null_row_count"] += 1
            class_index = int(vocab[EFMDC_NULL_TOKEN])
        elif class_code in vocab:
            class_index = int(vocab[class_code])
        else:
            stats["medication_class_oov_row_count"] += 1
            class_index = int(vocab[EFMDC_UNK_TOKEN])
        pairs.add((patient_id, class_index))
    return pairs, stats


def read_medication_class_vocab(path: str | Path) -> dict[str, int]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    vocab = {str(key): int(value) for key, value in raw.items()}
    _validate_class_vocab(vocab)
    return vocab


def write_medication_class_vocab(path: str | Path, vocab: Mapping[str, int]) -> None:
    _validate_class_vocab(vocab)
    Path(path).write_text(
        json.dumps(dict(vocab), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _normalized_class_series(histories: pd.DataFrame) -> pd.Series:
    if histories.empty or "efmdc_clsf_no" not in histories.columns:
        return pd.Series([], dtype=str)
    return histories["efmdc_clsf_no"].where(histories["efmdc_clsf_no"].notna(), "").astype(str).str.strip()


def _validate_class_vocab(vocab: Mapping[str, int]) -> None:
    if EFMDC_NULL_TOKEN not in vocab:
        raise ValueError(f"medication class vocab must include {EFMDC_NULL_TOKEN}")
    if EFMDC_UNK_TOKEN not in vocab:
        raise ValueError(f"medication class vocab must include {EFMDC_UNK_TOKEN}")
    values = sorted(int(value) for value in vocab.values())
    if values != list(range(len(values))):
        raise ValueError("medication class vocab indices must be contiguous from 0")
