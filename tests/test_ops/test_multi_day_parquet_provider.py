from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

EXPECTED_COLUMNS = (
    "patient_id",
    "drug_code",
    "prescription_date",
    "end_date",
    "total_days",
    "source",
)

EXPECTED_COLUMNS_WITH_SICK_CODE = (
    "patient_id",
    "drug_code",
    "prescription_date",
    "end_date",
    "total_days",
    "source",
    "sick_code",
)


def _write_records(path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_parquet(path, index=False)


def _row(
    patient_id: str,
    edi_code: str,
    start_date: str,
    *,
    end_date: str | None = None,
    total_days: int = 1,
    source: str = "T30",
    sick_code: str = "K92.1",
) -> dict:
    return {
        "patient_id": patient_id,
        "edi_code": edi_code,
        "start_date": start_date,
        "end_date": end_date or start_date,
        "total_days": total_days,
        "source": source,
        "sick_code": sick_code,
    }


def test_file_index_parsing_skips_bad_file_names(tmp_path) -> None:
    from scripts.ops.multi_day_parquet_provider import MultiDayParquetHistoryProvider

    _write_records(tmp_path / "records_20241001.parquet", [_row("P001", "D1", "2024-10-01")])
    _write_records(tmp_path / "records_20241003.parquet", [_row("P001", "D2", "2024-10-03")])
    _write_records(tmp_path / "records_bad.parquet", [_row("P001", "D3", "2024-10-02")])

    provider = MultiDayParquetHistoryProvider(tmp_path)

    assert tuple(provider.available_dates) == (date(2024, 10, 1), date(2024, 10, 3))


def test_window_inclusive_boundaries(tmp_path) -> None:
    from scripts.ops.multi_day_parquet_provider import MultiDayParquetHistoryProvider

    _write_records(tmp_path / "records_20241001.parquet", [_row("P001", "D1", "2024-10-01")])
    _write_records(tmp_path / "records_20241002.parquet", [_row("P001", "D2", "2024-10-02")])
    _write_records(tmp_path / "records_20241003.parquet", [_row("P001", "D3", "2024-10-03")])

    result = MultiDayParquetHistoryProvider(tmp_path).get_history(
        "P001",
        reference_date=date(2024, 10, 3),
        lookback_days=2,
    )

    assert result["drug_code"].tolist() == ["D1", "D2", "D3"]


def test_window_excludes_files_after_reference_date(tmp_path) -> None:
    from scripts.ops.multi_day_parquet_provider import MultiDayParquetHistoryProvider

    _write_records(tmp_path / "records_20241002.parquet", [_row("P001", "D2", "2024-10-02")])
    _write_records(tmp_path / "records_20241003.parquet", [_row("P001", "D3", "2024-10-03")])

    result = MultiDayParquetHistoryProvider(tmp_path).get_history(
        "P001",
        reference_date=date(2024, 10, 2),
        lookback_days=1,
    )

    assert result["drug_code"].tolist() == ["D2"]


def test_single_patient_found_returns_schema_and_deduplicates(tmp_path) -> None:
    from scripts.ops.multi_day_parquet_provider import MultiDayParquetHistoryProvider

    first = _row("P001", "D1", "2024-10-01", total_days=3)
    _write_records(tmp_path / "records_20241001.parquet", [first, first.copy()])
    _write_records(tmp_path / "records_20241002.parquet", [_row("P002", "D9", "2024-10-02")])
    _write_records(tmp_path / "records_20241003.parquet", [_row("P001", "D2", "2024-10-03")])

    result = MultiDayParquetHistoryProvider(tmp_path).get_history(
        "P001",
        reference_date=date(2024, 10, 3),
        lookback_days=2,
    )

    assert tuple(result.columns) == EXPECTED_COLUMNS
    assert result["patient_id"].tolist() == ["P001", "P001"]
    assert result["drug_code"].tolist() == ["D1", "D2"]
    assert result["prescription_date"].tolist() == [date(2024, 10, 1), date(2024, 10, 3)]


def test_patient_not_found_returns_empty_schema(tmp_path) -> None:
    from scripts.ops.multi_day_parquet_provider import MultiDayParquetHistoryProvider

    _write_records(tmp_path / "records_20241001.parquet", [_row("P001", "D1", "2024-10-01")])

    result = MultiDayParquetHistoryProvider(tmp_path).get_history(
        "UNKNOWN",
        reference_date=date(2024, 10, 1),
        lookback_days=0,
    )

    assert result.empty
    assert tuple(result.columns) == EXPECTED_COLUMNS


def test_dedup_t30_priority_for_same_patient_drug_and_date(tmp_path) -> None:
    from scripts.ops.multi_day_parquet_provider import MultiDayParquetHistoryProvider

    _write_records(
        tmp_path / "records_20241001.parquet",
        [
            _row("P001", "D1", "2024-10-01", total_days=90, source="T60"),
            _row("P001", "D1", "2024-10-01", total_days=3, source="T30"),
        ],
    )

    result = MultiDayParquetHistoryProvider(tmp_path).get_history(
        "P001",
        reference_date=date(2024, 10, 1),
        lookback_days=0,
    )

    assert result["drug_code"].tolist() == ["D1"]
    assert result["source"].tolist() == ["T30"]
    assert result["total_days"].tolist() == [3]


def test_batch_returns_all_found_patients(tmp_path) -> None:
    from scripts.ops.multi_day_parquet_provider import MultiDayParquetHistoryProvider

    _write_records(tmp_path / "records_20241001.parquet", [_row("P001", "D1", "2024-10-01")])
    _write_records(tmp_path / "records_20241002.parquet", [_row("P002", "D2", "2024-10-02")])

    result = MultiDayParquetHistoryProvider(tmp_path).get_history_batch(
        ["P001", "P002", "UNKNOWN"],
        reference_date=date(2024, 10, 2),
        lookback_days=1,
    )

    assert result["patient_id"].tolist() == ["P001", "P002"]
    assert result["drug_code"].tolist() == ["D1", "D2"]


def test_missing_required_columns_fails_with_file_name(tmp_path) -> None:
    from scripts.ops.multi_day_parquet_provider import MultiDayParquetHistoryProvider

    _write_records(tmp_path / "records_20241001.parquet", [{"patient_id": "P001"}])

    with pytest.raises(ValueError, match=r"records_20241001\.parquet.*missing columns"):
        MultiDayParquetHistoryProvider(tmp_path).get_history(
            "P001",
            reference_date=date(2024, 10, 1),
            lookback_days=0,
        )


def test_extra_columns_are_passed_through_when_requested(tmp_path) -> None:
    from scripts.ops.multi_day_parquet_provider import MultiDayParquetHistoryProvider

    _write_records(
        tmp_path / "records_20241001.parquet",
        [_row("P001", "D1", "2024-10-01", sick_code="K92.1")],
    )

    result = MultiDayParquetHistoryProvider(
        tmp_path,
        extra_columns=["sick_code"],
    ).get_history(
        "P001",
        reference_date=date(2024, 10, 1),
        lookback_days=0,
    )

    assert tuple(result.columns) == EXPECTED_COLUMNS_WITH_SICK_CODE
    assert result["sick_code"].tolist() == ["K92.1"]


def test_extra_columns_can_preserve_same_key_rows_when_key_dedup_disabled(tmp_path) -> None:
    from scripts.ops.multi_day_parquet_provider import MultiDayParquetHistoryProvider

    _write_records(
        tmp_path / "records_20241001.parquet",
        [
            _row("P001", "D1", "2024-10-01", sick_code="K92.1"),
            _row("P001", "D1", "2024-10-01", sick_code="N17"),
        ],
    )

    result = MultiDayParquetHistoryProvider(
        tmp_path,
        extra_columns=["sick_code"],
        deduplicate_keys=False,
    ).get_history(
        "P001",
        reference_date=date(2024, 10, 1),
        lookback_days=0,
    )

    assert result["sick_code"].tolist() == ["K92.1", "N17"]
