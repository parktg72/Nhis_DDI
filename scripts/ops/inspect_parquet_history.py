"""Inspect daily parquet prescription samples without printing patient IDs."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

from scripts.datasets.contracts import DL_DATASET_REQUIRED_COLUMNS
from scripts.ops.parquet_history_provider import ParquetHistoryProvider
from serving.hana_history import validate_history_frame


REQUIRED_SOURCE_COLUMNS = ["patient_id", "edi_code", "start_date"]


@dataclass(frozen=True)
class FileStats:
    rows: int
    cols: int
    required_columns_present: list[str]
    required_columns_missing: list[str]
    source_counts: dict[str, int]
    start_date_range: tuple[str, str] | None
    end_date_range: tuple[str, str] | None
    unique_patient_count: int
    full_duplicate_count: int
    output_key_duplicate_count: int


@dataclass(frozen=True)
class ProviderSample:
    patient_label: str
    found: bool
    rows: int
    unique_drug_count: int
    date_range: tuple[str, str] | None
    schema_ok: bool


@dataclass(frozen=True)
class InspectionResult:
    path: Path
    file_stats: FileStats
    provider_sample: ProviderSample | None

    @property
    def ok(self) -> bool:
        return not self.file_stats.required_columns_missing


def _range_for(df: pd.DataFrame, column: str) -> tuple[str, str] | None:
    if column not in df.columns or df.empty:
        return None
    values = df[column].dropna().astype(str)
    if values.empty:
        return None
    return values.min(), values.max()


def _source_counts(df: pd.DataFrame) -> dict[str, int]:
    if "source" not in df.columns:
        return {}
    counts = df["source"].astype(str).value_counts(dropna=False)
    return {str(key): int(value) for key, value in counts.sort_index().items()}


def _output_key_duplicate_count(df: pd.DataFrame) -> int:
    if any(col not in df.columns for col in REQUIRED_SOURCE_COLUMNS):
        return 0
    prescription_dates = pd.to_datetime(df["start_date"], errors="coerce")
    keys = pd.DataFrame({
        "patient_id": df["patient_id"].astype(str),
        "drug_code": df["edi_code"].astype(str).str.strip(),
        "prescription_date": prescription_dates.dt.strftime("%Y%m%d"),
    })
    return int(keys.duplicated(list(DL_DATASET_REQUIRED_COLUMNS)).sum())


def _file_stats(df: pd.DataFrame) -> FileStats:
    present = [col for col in REQUIRED_SOURCE_COLUMNS if col in df.columns]
    missing = [col for col in REQUIRED_SOURCE_COLUMNS if col not in df.columns]
    unique_patient_count = (
        int(df["patient_id"].nunique(dropna=True)) if "patient_id" in df.columns else 0
    )
    return FileStats(
        rows=len(df),
        cols=len(df.columns),
        required_columns_present=present,
        required_columns_missing=missing,
        source_counts=_source_counts(df),
        start_date_range=_range_for(df, "start_date"),
        end_date_range=_range_for(df, "end_date"),
        unique_patient_count=unique_patient_count,
        full_duplicate_count=int(df.duplicated().sum()),
        output_key_duplicate_count=_output_key_duplicate_count(df),
    )


def _select_patient_id(df: pd.DataFrame, patient_id: str | None) -> tuple[str | None, str]:
    if patient_id:
        return str(patient_id), "<provided>"
    if "patient_id" not in df.columns or df.empty:
        return None, "<first patient>"
    values = df["patient_id"].dropna().astype(str)
    if values.empty:
        return None, "<first patient>"
    return str(values.iloc[0]), "<first patient>"


def _provider_sample(
    path: Path,
    df: pd.DataFrame,
    patient_id: str | None,
) -> ProviderSample | None:
    selected, label = _select_patient_id(df, patient_id)
    if selected is None:
        return ProviderSample(
            patient_label=label,
            found=False,
            rows=0,
            unique_drug_count=0,
            date_range=None,
            schema_ok=False,
        )
    history = ParquetHistoryProvider(path).fetch_patient_history(
        selected,
        reference_date=pd.Timestamp.today().date(),
        lookback_days=365,
    )
    schema_ok = True
    try:
        validate_history_frame(history, context="parquet inspection sample")
    except Exception:
        schema_ok = False
    dates = history["prescription_date"] if "prescription_date" in history.columns else pd.Series()
    date_range = None if dates.empty else (str(dates.min()), str(dates.max()))
    return ProviderSample(
        patient_label=label,
        found=not history.empty,
        rows=len(history),
        unique_drug_count=(
            int(history["drug_code"].nunique(dropna=True))
            if "drug_code" in history.columns
            else 0
        ),
        date_range=date_range,
        schema_ok=schema_ok,
    )


def run_inspection(
    path: str | Path,
    *,
    patient_id: str | None = None,
) -> InspectionResult:
    sample_path = Path(path)
    df = pd.read_parquet(sample_path)
    stats = _file_stats(df)
    sample = None
    if not stats.required_columns_missing:
        sample = _provider_sample(sample_path, df, patient_id)
    return InspectionResult(
        path=sample_path,
        file_stats=stats,
        provider_sample=sample,
    )


def _format_range(value: tuple[str, str] | None) -> str:
    if value is None:
        return "n/a"
    return f"{value[0]} ~ {value[1]}"


def _format_source_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "n/a"
    return ", ".join(f"{key}={value}" for key, value in counts.items())


def print_report(result: InspectionResult) -> None:
    stats = result.file_stats
    print(f"[FILE] {result.path}")
    print(f"rows={stats.rows}  cols={stats.cols}")
    if stats.required_columns_missing:
        print(
            "required columns: MISSING "
            f"({', '.join(stats.required_columns_missing)})"
        )
    else:
        print(
            "required columns: OK "
            f"({', '.join(stats.required_columns_present)})"
        )
    print(f"source counts: {_format_source_counts(stats.source_counts)}")
    print(f"start_date range: {_format_range(stats.start_date_range)}")
    print(f"end_date range: {_format_range(stats.end_date_range)}")
    print(f"unique patients: {stats.unique_patient_count}")
    print(f"full duplicates: {stats.full_duplicate_count}")
    print(f"output key duplicates: {stats.output_key_duplicate_count}")

    if result.provider_sample is None:
        return
    sample = result.provider_sample
    print()
    print(f"[PROVIDER SAMPLE] patient={sample.patient_label}")
    print(
        f"found: {sample.found}  rows: {sample.rows}  "
        f"unique_drugs: {sample.unique_drug_count}"
    )
    print(f"date_range: {_format_range(sample.date_range)}")
    print(f"schema: {'OK' if sample.schema_ok else 'FAILED'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect a daily parquet history sample without printing patient IDs.",
    )
    parser.add_argument("path")
    parser.add_argument(
        "--patient-id",
        default=None,
        help="Patient ID to sample internally. The value is not printed.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_inspection(args.path, patient_id=args.patient_id)
    print_report(result)
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
