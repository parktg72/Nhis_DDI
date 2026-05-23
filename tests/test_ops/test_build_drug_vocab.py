from __future__ import annotations

import json


def test_unk_always_index_zero() -> None:
    from scripts.ops.build_drug_vocab import build_drug_vocab

    result = build_drug_vocab({"D1": 10, "D2": 20}, cutoff=1)

    assert result["_unk"] == 0


def test_freq_desc_ordering() -> None:
    from scripts.ops.build_drug_vocab import build_drug_vocab

    result = build_drug_vocab({"D_LOW": 10, "D_HIGH": 30, "D_MID": 20}, cutoff=1)

    assert result["D_HIGH"] == 1
    assert result["D_MID"] == 2
    assert result["D_LOW"] == 3


def test_tie_break_lex_asc() -> None:
    from scripts.ops.build_drug_vocab import build_drug_vocab

    result = build_drug_vocab({"D_B": 10, "D_A": 10, "D_C": 10}, cutoff=1)

    assert list(result) == ["_unk", "D_A", "D_B", "D_C"]


def test_cutoff_filters_below_threshold_and_includes_equal() -> None:
    from scripts.ops.build_drug_vocab import build_drug_vocab

    result = build_drug_vocab({"D99": 99, "D100": 100, "D101": 101}, cutoff=100)

    assert "D99" not in result
    assert result["D101"] == 1
    assert result["D100"] == 2


def test_indices_contiguous() -> None:
    from scripts.ops.build_drug_vocab import build_drug_vocab

    result = build_drug_vocab({"D1": 3, "D2": 2, "D3": 1}, cutoff=1)

    assert sorted(result.values()) == [0, 1, 2, 3]


def test_vocab_size_matches_included_codes() -> None:
    from scripts.ops.build_drug_vocab import build_drug_vocab

    result = build_drug_vocab({"D1": 3, "D2": 1, "D3": 0}, cutoff=1)

    assert len(result) - 1 == 2


def test_empty_input_returns_unk_only() -> None:
    from scripts.ops.build_drug_vocab import build_drug_vocab

    assert build_drug_vocab({}, cutoff=100) == {"_unk": 0}


def test_input_unk_key_does_not_override_reserved_unk() -> None:
    from scripts.ops.build_drug_vocab import build_drug_vocab

    result = build_drug_vocab({"_unk": 999, "D1": 100}, cutoff=100)

    assert result == {"_unk": 0, "D1": 1}


def test_write_json_format(tmp_path) -> None:
    from scripts.ops.build_drug_vocab import VocabBuildMeta, write_vocab_outputs

    vocab = {"_unk": 0, "D1": 1}
    meta = VocabBuildMeta(
        cutoff=100,
        vocab_size=1,
        input_dim=2,
        unk_index=0,
        sort_policy="freq_desc_lex_asc",
        date_range=("2024-10-01", "2024-11-30"),
        total_files=61,
    )

    vocab_path, meta_path = write_vocab_outputs(vocab, meta, tmp_path)

    assert json.loads(vocab_path.read_text(encoding="utf-8")) == vocab
    meta_payload = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta_payload["cutoff"] == 100
    assert meta_payload["vocab_size"] == 1
    assert meta_payload["input_dim"] == 2
    assert meta_payload["unk_index"] == 0
    assert meta_payload["sort_policy"] == "freq_desc_lex_asc"
