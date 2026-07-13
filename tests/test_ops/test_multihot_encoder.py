from __future__ import annotations

from datetime import date

import pandas as pd

VOCAB = {"_unk": 0, "D1": 1, "D2": 2, "12345": 3}


def _history(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_known_drug_sets_correct_index() -> None:
    from scripts.ops.multihot_encoder import encode_patient_history

    result = encode_patient_history(_history([{"drug_code": "D1"}]), VOCAB)

    assert result[VOCAB["D1"]] == 1
    assert result.sum() == 1


def test_unknown_drug_sets_unk_bit() -> None:
    from scripts.ops.multihot_encoder import encode_patient_history

    result = encode_patient_history(_history([{"drug_code": "UNKNOWN"}]), VOCAB)

    assert result[VOCAB["_unk"]] == 1
    assert result.sum() == 1


def test_all_known_unk_bit_is_zero() -> None:
    from scripts.ops.multihot_encoder import encode_patient_history

    result = encode_patient_history(_history([{"drug_code": "D1"}, {"drug_code": "D2"}]), VOCAB)

    assert result[VOCAB["_unk"]] == 0
    assert result[VOCAB["D1"]] == 1
    assert result[VOCAB["D2"]] == 1


def test_empty_history_zero_vector() -> None:
    from scripts.ops.multihot_encoder import encode_patient_history

    result = encode_patient_history(pd.DataFrame(columns=["drug_code"]), VOCAB)

    assert result.shape == (len(VOCAB),)
    assert result.sum() == 0


def test_duplicate_prescriptions_binary() -> None:
    from scripts.ops.multihot_encoder import encode_patient_history

    result = encode_patient_history(
        _history([{"drug_code": "D1"}, {"drug_code": "D1"}, {"drug_code": "D1"}]),
        VOCAB,
    )

    assert result[VOCAB["D1"]] == 1
    assert result.sum() == 1


def test_vector_shape() -> None:
    from scripts.ops.multihot_encoder import encode_patient_history

    result = encode_patient_history(_history([{"drug_code": "D1"}]), VOCAB)

    assert result.shape == (len(VOCAB),)
    assert str(result.dtype) == "float32"


def test_known_and_unknown_mixed() -> None:
    from scripts.ops.multihot_encoder import encode_patient_history

    result = encode_patient_history(
        _history([{"drug_code": "D1"}, {"drug_code": "UNKNOWN"}]),
        VOCAB,
    )

    assert result[VOCAB["_unk"]] == 1
    assert result[VOCAB["D1"]] == 1
    assert result.sum() == 2


def test_null_nan_and_blank_drug_codes_are_skipped() -> None:
    from scripts.ops.multihot_encoder import encode_patient_history

    result = encode_patient_history(
        _history([{"drug_code": None}, {"drug_code": float("nan")}, {"drug_code": ""}]),
        VOCAB,
    )

    assert result.sum() == 0


def test_numeric_drug_code_matches_string_vocab_key() -> None:
    from scripts.ops.multihot_encoder import encode_patient_history

    result = encode_patient_history(_history([{"drug_code": 12345}]), VOCAB)

    assert result[VOCAB["12345"]] == 1
    assert result[VOCAB["_unk"]] == 0


class _FakeProvider:
    def __init__(self, histories: dict[str, pd.DataFrame]) -> None:
        self.histories = histories

    def get_history_batch(
        self,
        patient_ids,
        reference_date: date,
        lookback_days: int = 60,
    ) -> pd.DataFrame:
        del reference_date, lookback_days
        frames = []
        for patient_id in patient_ids:
            history = self.histories.get(patient_id, pd.DataFrame(columns=["drug_code"]))
            if history.empty:
                continue
            chunk = history.copy()
            chunk["patient_id"] = patient_id
            frames.append(chunk)
        if not frames:
            return pd.DataFrame(columns=["patient_id", "drug_code"])
        return pd.concat(frames, ignore_index=True)


def test_encode_batch_returns_matrix_and_stats() -> None:
    from scripts.ops.multihot_encoder import encode_batch

    provider = _FakeProvider({
        "P1": _history([{"drug_code": "D1"}, {"drug_code": "UNKNOWN"}]),
        "P2": _history([{"drug_code": "D2"}]),
    })

    matrix, stats = encode_batch(
        provider,
        ["P1", "P2", "P3"],
        VOCAB,
        reference_date=date(2024, 11, 30),
        lookback_days=60,
    )

    assert matrix.shape == (3, len(VOCAB))
    assert matrix[0, VOCAB["D1"]] == 1
    assert matrix[0, VOCAB["_unk"]] == 1
    assert matrix[1, VOCAB["D2"]] == 1
    assert matrix[2].sum() == 0
    assert stats["n_patients"] == 3
    assert stats["input_dim"] == len(VOCAB)
    assert stats["unk_flag_patients"] == 1
    assert stats["unk_flag_rate_pct"] == 33.3333
    assert stats["total_unk_prescriptions"] == 1
    assert stats["zero_vector_patients"] == 1
    assert stats["zero_vector_rate_pct"] == 33.3333
    assert stats["known_bits_mean"] == 0.6667
    assert stats["density_p95"] > 0


def test_encode_batch_preserves_duplicate_patient_order() -> None:
    from scripts.ops.multihot_encoder import encode_batch

    provider = _FakeProvider({
        "P1": _history([{"drug_code": "D1"}]),
        "P2": _history([{"drug_code": "D2"}]),
    })

    matrix, stats = encode_batch(
        provider,
        ["P1", "P2", "P1"],
        VOCAB,
        reference_date=date(2024, 11, 30),
    )

    assert matrix.shape == (3, len(VOCAB))
    assert matrix[0, VOCAB["D1"]] == 1
    assert matrix[1, VOCAB["D2"]] == 1
    assert matrix[2, VOCAB["D1"]] == 1
    assert stats["n_patients"] == 3


def test_missing_unk_key_fails_clearly() -> None:
    from scripts.ops.multihot_encoder import encode_patient_history

    try:
        encode_patient_history(_history([{"drug_code": "UNKNOWN"}]), {"D1": 0})
    except ValueError as exc:
        assert "_unk" in str(exc)
    else:
        raise AssertionError("expected missing _unk ValueError")
