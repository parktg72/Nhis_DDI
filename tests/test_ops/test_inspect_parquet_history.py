from __future__ import annotations

import pandas as pd


def _write_records(path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_parquet(path, index=False)


def _base_rows() -> list[dict]:
    return [
        {
            "patient_id": "P001",
            "edi_code": "D1",
            "start_date": "2024-10-01",
            "end_date": "2024-10-03",
            "source": "T30",
            "total_days": 3,
        },
        {
            "patient_id": "P001",
            "edi_code": "D2",
            "start_date": "2024-10-01",
            "end_date": "2024-10-01",
            "source": "T60",
            "total_days": 1,
        },
        {
            "patient_id": "P002",
            "edi_code": "D3",
            "start_date": "2024-10-02",
            "end_date": "2024-10-02",
            "source": "T30",
            "total_days": 1,
        },
    ]


def test_inspect_parquet_history_reports_file_stats(tmp_path) -> None:
    from scripts.ops.inspect_parquet_history import run_inspection

    path = tmp_path / "records.parquet"
    _write_records(path, _base_rows())

    result = run_inspection(path)

    assert result.ok is True
    assert result.file_stats.rows == 3
    assert result.file_stats.cols == 6
    assert result.file_stats.required_columns_missing == []
    assert result.file_stats.source_counts == {"T30": 2, "T60": 1}
    assert result.file_stats.start_date_range == ("2024-10-01", "2024-10-02")
    assert result.file_stats.end_date_range == ("2024-10-01", "2024-10-03")
    assert result.file_stats.unique_patient_count == 2


def test_inspect_parquet_history_detects_missing_required_column(tmp_path) -> None:
    from scripts.ops.inspect_parquet_history import run_inspection

    path = tmp_path / "bad.parquet"
    _write_records(path, [{"patient_id": "P001", "start_date": "2024-10-01"}])

    result = run_inspection(path)

    assert result.ok is False
    assert result.file_stats.required_columns_missing == ["edi_code"]
    assert result.provider_sample is None


def test_inspect_parquet_history_counts_full_and_output_key_duplicates(tmp_path) -> None:
    from scripts.ops.inspect_parquet_history import run_inspection

    path = tmp_path / "records.parquet"
    rows = _base_rows()
    rows.append(rows[0].copy())
    key_duplicate = rows[0].copy()
    key_duplicate["source"] = "T60"
    key_duplicate["end_date"] = "2024-12-31"
    rows.append(key_duplicate)
    _write_records(path, rows)

    result = run_inspection(path)

    assert result.file_stats.full_duplicate_count == 1
    assert result.file_stats.output_key_duplicate_count == 2


def test_inspect_parquet_history_samples_first_patient_without_printing_id(tmp_path) -> None:
    from scripts.ops.inspect_parquet_history import run_inspection

    path = tmp_path / "records.parquet"
    _write_records(path, _base_rows())

    result = run_inspection(path)

    assert result.provider_sample is not None
    assert result.provider_sample.patient_label == "<first patient>"
    assert result.provider_sample.found is True
    assert result.provider_sample.rows == 2
    assert result.provider_sample.unique_drug_count == 2
    assert result.provider_sample.date_range == ("20241001", "20241001")
    assert result.provider_sample.schema_ok is True


def test_inspect_parquet_history_samples_provided_patient_and_unknown_patient(tmp_path) -> None:
    from scripts.ops.inspect_parquet_history import run_inspection

    path = tmp_path / "records.parquet"
    _write_records(path, _base_rows())

    known = run_inspection(path, patient_id="P002")
    unknown = run_inspection(path, patient_id="NOPE")

    assert known.provider_sample is not None
    assert known.provider_sample.patient_label == "<provided>"
    assert known.provider_sample.found is True
    assert known.provider_sample.rows == 1
    assert known.provider_sample.unique_drug_count == 1
    assert unknown.provider_sample is not None
    assert unknown.provider_sample.patient_label == "<provided>"
    assert unknown.provider_sample.found is False
    assert unknown.provider_sample.rows == 0
    assert unknown.provider_sample.unique_drug_count == 0


def test_inspect_parquet_history_cli_returns_nonzero_on_missing_column(tmp_path) -> None:
    from scripts.ops.inspect_parquet_history import main

    path = tmp_path / "bad.parquet"
    _write_records(path, [{"patient_id": "P001", "start_date": "2024-10-01"}])

    assert main([str(path)]) == 1
