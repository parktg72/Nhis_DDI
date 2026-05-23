from __future__ import annotations

import json

import pandas as pd


def _write_records(path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_parquet(path, index=False)


def _row(patient_id: str | None, edi_code: str | None, source: str) -> dict:
    return {
        "patient_id": patient_id,
        "edi_code": edi_code,
        "source": source,
    }


def test_freq_counts_correct(tmp_path) -> None:
    from scripts.ops.build_drug_vocab_audit import build_vocab_audit

    _write_records(
        tmp_path / "records_20241001.parquet",
        [
            _row("P1", "D1", "T30"),
            _row("P2", "D1", "T30"),
            _row("P1", "D2", "T60"),
            _row("P3", "D3", "T60"),
        ],
    )

    result = build_vocab_audit(tmp_path)

    assert result.meta.total_rows == 4
    assert result.meta.unique_patients == 3
    assert result.meta.unique_edi_codes == 3
    assert result.code_stats["D1"].row_count == 2
    assert result.code_stats["D1"].patient_count == 2


def test_cutoff_table_thresholds(tmp_path) -> None:
    from scripts.ops.build_drug_vocab_audit import build_vocab_audit

    rows = [_row(f"P{i}", "D1", "T30") for i in range(10)]
    rows.extend(_row(f"Q{i}", "D2", "T30") for i in range(9))
    _write_records(tmp_path / "records_20241001.parquet", rows)

    result = build_vocab_audit(tmp_path, cutoffs=(10,))

    assert result.cutoff_table[0].cutoff == 10
    assert result.cutoff_table[0].vocab_size == 1


def test_row_coverage_fraction(tmp_path) -> None:
    from scripts.ops.build_drug_vocab_audit import build_vocab_audit

    rows = [_row(f"P{i}", "D1", "T30") for i in range(10)]
    rows.extend(_row(f"Q{i}", "D2", "T60") for i in range(5))
    _write_records(tmp_path / "records_20241001.parquet", rows)

    result = build_vocab_audit(tmp_path, cutoffs=(10,))

    assert result.cutoff_table[0].row_coverage_pct == 66.6667


def test_patient_coverage_fraction(tmp_path) -> None:
    from scripts.ops.build_drug_vocab_audit import build_vocab_audit

    _write_records(
        tmp_path / "records_20241001.parquet",
        [
            _row("P1", "D1", "T30"),
            _row("P2", "D1", "T30"),
            _row("P3", "D2", "T60"),
            _row("P4", "D3", "T60"),
        ],
    )

    result = build_vocab_audit(tmp_path, cutoffs=(2,))

    assert result.cutoff_table[0].patient_coverage_pct == 50.0


def test_source_split_counts(tmp_path) -> None:
    from scripts.ops.build_drug_vocab_audit import build_vocab_audit

    _write_records(
        tmp_path / "records_20241001.parquet",
        [
            _row("P1", "D1", "T30"),
            _row("P2", "D1", "T30"),
            _row("P3", "D2", "T60"),
        ],
    )

    result = build_vocab_audit(tmp_path)

    assert result.source_split["T30"].total_rows == 2
    assert result.source_split["T30"].unique_edi_codes == 1
    assert result.source_split["T60"].total_rows == 1
    assert result.source_split["T60"].unique_edi_codes == 1


def test_output_json_required_keys(tmp_path) -> None:
    from scripts.ops.build_drug_vocab_audit import build_vocab_audit, write_audit_outputs

    _write_records(tmp_path / "records_20241001.parquet", [_row("P1", "D1", "T30")])

    result = build_vocab_audit(tmp_path)
    json_path, md_path = write_audit_outputs(result, tmp_path / "out")

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert set(payload) == {"meta", "cutoff_table", "source_split", "top20_by_frequency"}
    assert md_path.exists()
    assert "cutoff" in md_path.read_text(encoding="utf-8")


def test_empty_raw_dir_no_crash(tmp_path) -> None:
    from scripts.ops.build_drug_vocab_audit import build_vocab_audit

    result = build_vocab_audit(tmp_path)

    assert result.meta.total_files == 0
    assert result.meta.total_rows == 0
    assert result.meta.unique_patients == 0
    assert result.meta.unique_edi_codes == 0
    assert result.cutoff_table[0].vocab_size == 0


def test_date_range_filters_files_by_name_before_read(tmp_path) -> None:
    from scripts.ops.build_drug_vocab_audit import build_vocab_audit

    _write_records(tmp_path / "records_20241001.parquet", [_row("P1", "D1", "T30")])
    _write_records(tmp_path / "records_20241002.parquet", [_row("P2", "D2", "T30")])

    result = build_vocab_audit(tmp_path, date_from="20241002", date_to="20241002")

    assert result.meta.total_files == 1
    assert result.meta.total_rows == 1
    assert result.meta.date_range == ("2024-10-02", "2024-10-02")
    assert "D1" not in result.code_stats


def test_null_patient_or_edi_rows_are_excluded(tmp_path) -> None:
    from scripts.ops.build_drug_vocab_audit import build_vocab_audit

    _write_records(
        tmp_path / "records_20241001.parquet",
        [
            _row("P1", "D1", "T30"),
            _row(None, "D2", "T30"),
            _row("P2", None, "T60"),
        ],
    )

    result = build_vocab_audit(tmp_path)

    assert result.meta.total_rows == 1
    assert result.meta.unique_patients == 1
    assert result.meta.unique_edi_codes == 1
    assert list(result.code_stats) == ["D1"]
