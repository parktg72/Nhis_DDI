from __future__ import annotations

import pandas as pd


def test_build_medication_class_vocab_keeps_null_and_unknown_tokens() -> None:
    from scripts.ops.medication_class_features import (
        EFMDC_NULL_TOKEN,
        EFMDC_UNK_TOKEN,
        build_medication_class_vocab,
    )

    histories = pd.DataFrame({
        "patient_id": ["P1", "P1", "P2", "P3", "P4"],
        "efmdc_clsf_no": ["222", None, "114", "222", "RARE"],
    })

    vocab, metadata = build_medication_class_vocab(histories, min_count=2)

    assert vocab == {
        EFMDC_NULL_TOKEN: 0,
        EFMDC_UNK_TOKEN: 1,
        "222": 2,
    }
    assert metadata["medication_class_vocab_size"] == 3
    assert metadata["medication_class_nonblank_unique_count"] == 3
    assert metadata["medication_class_min_count"] == 2
    assert metadata["medication_class_dropped_rare_count"] == 2


def test_patient_medication_class_pairs_separate_null_and_oov() -> None:
    from scripts.ops.medication_class_features import (
        EFMDC_NULL_TOKEN,
        EFMDC_UNK_TOKEN,
        patient_medication_class_pairs,
    )

    vocab = {
        EFMDC_NULL_TOKEN: 0,
        EFMDC_UNK_TOKEN: 1,
        "222": 2,
    }
    histories = pd.DataFrame({
        "patient_id": ["P1", "P1", "P2", "P2", "P3"],
        "efmdc_clsf_no": ["222", None, "999", "", "222"],
    })

    pairs, stats = patient_medication_class_pairs(histories, ["P1", "P2"], vocab)

    assert pairs == {
        ("P1", 0),
        ("P1", 2),
        ("P2", 0),
        ("P2", 1),
    }
    assert stats["medication_class_total_rows"] == 4
    assert stats["medication_class_null_row_count"] == 2
    assert stats["medication_class_oov_row_count"] == 1
