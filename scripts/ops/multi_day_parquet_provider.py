"""Multi-day parquet-backed prescription history provider for ops validation."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Sequence

import pandas as pd
import pyarrow.parquet as pq

LOGGER = logging.getLogger(__name__)

REQUIRED_SOURCE_COLUMNS = (
    "patient_id",
    "edi_code",
    "start_date",
    "end_date",
    "total_days",
    "source",
)
OUTPUT_COLUMNS = (
    "patient_id",
    "drug_code",
    "prescription_date",
    "end_date",
    "total_days",
    "source",
)


@dataclass
class MultiDayParquetHistoryProvider:
    """Read records_YYYYMMDD parquet files as a window-aware history source."""

    raw_dir: str | Path
    date_format: str = "%Y%m%d"
    file_prefix: str = "records_"
    file_suffix: str = ".parquet"
    extra_columns: Sequence[str] = ()
    deduplicate_keys: bool = True
    _files_by_date: dict[date, Path] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.raw_dir = Path(self.raw_dir)
        self.extra_columns = tuple(dict.fromkeys(str(col) for col in self.extra_columns))
        self._files_by_date = self._build_file_index()

    @property
    def available_dates(self) -> tuple[date, ...]:
        return tuple(sorted(self._files_by_date))

    def get_history(
        self,
        patient_id: str,
        reference_date: date,
        lookback_days: int = 60,
    ) -> pd.DataFrame:
        return self.get_history_batch(
            [patient_id],
            reference_date=reference_date,
            lookback_days=lookback_days,
        )

    def get_history_batch(
        self,
        patient_ids: Sequence[str],
        reference_date: date,
        lookback_days: int = 60,
    ) -> pd.DataFrame:
        if lookback_days < 0:
            raise ValueError("lookback_days must be non-negative")

        patient_id_set = {str(patient_id) for patient_id in patient_ids}
        if not patient_id_set:
            return self._empty_history()

        chunks: list[pd.DataFrame] = []
        for path in self._paths_for_window(reference_date, lookback_days):
            chunk = self._read_filtered_file(path, patient_id_set)
            if not chunk.empty:
                chunks.append(chunk)

        if not chunks:
            return self._empty_history()

        return self._deduplicate_and_sort(pd.concat(chunks, ignore_index=True))

    def _build_file_index(self) -> dict[date, Path]:
        files_by_date: dict[date, Path] = {}
        pattern = f"{self.file_prefix}*{self.file_suffix}"
        for path in sorted(self.raw_dir.glob(pattern)):
            parsed = self._date_from_path(path)
            if parsed is None:
                LOGGER.warning("Skipping parquet file with unparsable date: %s", path.name)
                continue
            files_by_date[parsed] = path
        return files_by_date

    def _date_from_path(self, path: Path) -> date | None:
        name = path.name
        if not name.startswith(self.file_prefix) or not name.endswith(self.file_suffix):
            return None
        raw_date = name[len(self.file_prefix) : -len(self.file_suffix)]
        try:
            return datetime.strptime(raw_date, self.date_format).date()
        except ValueError:
            return None

    def _paths_for_window(self, reference_date: date, lookback_days: int) -> list[Path]:
        start_date = reference_date - timedelta(days=lookback_days)
        selected = [
            path
            for file_date, path in sorted(self._files_by_date.items())
            if start_date <= file_date <= reference_date
        ]
        if len(selected) > 90:
            LOGGER.warning(
                "Selected %d parquet files for one history window; consider narrowing lookback_days",
                len(selected),
            )
        return selected

    def _read_filtered_file(self, path: Path, patient_ids: set[str]) -> pd.DataFrame:
        self._validate_required_columns(path)
        raw = pd.read_parquet(path, columns=list(self._source_columns()))
        mask = raw["patient_id"].astype(str).isin(patient_ids)
        if not mask.any():
            return self._empty_history()
        return self._normalize(raw.loc[mask].copy(), path)

    def _validate_required_columns(self, path: Path) -> None:
        available = set(pq.ParquetFile(path).schema_arrow.names)
        missing = [column for column in self._source_columns() if column not in available]
        if missing:
            raise ValueError(f"{path.name}: missing columns {missing}")

    def _normalize(self, df: pd.DataFrame, path: Path) -> pd.DataFrame:
        prescription_dates = pd.to_datetime(df["start_date"], errors="coerce")
        end_dates = pd.to_datetime(df["end_date"], errors="coerce")
        if prescription_dates.isna().any():
            raise ValueError(f"{path.name}: start_date contains invalid values")
        if end_dates.isna().any():
            raise ValueError(f"{path.name}: end_date contains invalid values")

        normalized = pd.DataFrame({
            "patient_id": df["patient_id"].astype(str),
            "drug_code": df["edi_code"].astype(str).str.strip(),
            "prescription_date": prescription_dates.dt.date,
            "end_date": end_dates.dt.date,
            "total_days": pd.to_numeric(df["total_days"], errors="raise").astype("int64"),
            "source": df["source"].astype(str),
        })
        for column in self.extra_columns:
            normalized[column] = df[column]
        return normalized.loc[:, list(self._output_columns())]

    def _deduplicate_and_sort(self, df: pd.DataFrame) -> pd.DataFrame:
        deduped = df.drop_duplicates(subset=list(self._output_columns()), keep="first").copy()
        if not self.deduplicate_keys:
            return deduped.sort_values(
                ["patient_id", "prescription_date"],
                kind="mergesort",
            ).reset_index(drop=True)
        deduped["_source_priority"] = (deduped["source"] == "T30").astype(int)
        deduped = deduped.sort_values(
            ["patient_id", "drug_code", "prescription_date", "_source_priority"],
            ascending=[True, True, True, False],
            kind="mergesort",
        )
        deduped = deduped.drop_duplicates(
            subset=["patient_id", "drug_code", "prescription_date"],
            keep="first",
        )
        deduped = deduped.drop(columns=["_source_priority"])
        return deduped.sort_values(
            ["patient_id", "prescription_date"],
            kind="mergesort",
        ).reset_index(drop=True)

    def _empty_history(self) -> pd.DataFrame:
        return pd.DataFrame(columns=list(self._output_columns()))

    def _source_columns(self) -> tuple[str, ...]:
        return tuple(REQUIRED_SOURCE_COLUMNS) + tuple(
            column for column in self.extra_columns if column not in REQUIRED_SOURCE_COLUMNS
        )

    def _output_columns(self) -> tuple[str, ...]:
        return tuple(OUTPUT_COLUMNS) + tuple(
            column for column in self.extra_columns if column not in OUTPUT_COLUMNS
        )
