"""Single-pass raw parquet loader for full-cohort ops dataset builds."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterator, Sequence

import pandas as pd
import pyarrow.parquet as pq

REQUIRED_COLUMNS = (
    "patient_id",
    "edi_code",
    "start_date",
)


@dataclass
class FullCohortHistoryLoader:
    raw_dir: str | Path
    date_format: str = "%Y%m%d"
    file_prefix: str = "records_"
    file_suffix: str = ".parquet"
    extra_columns: Sequence[str] = ()
    last_loaded_file_count: int = field(default=0, init=False)
    _files_by_date: dict[date, Path] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.raw_dir = Path(self.raw_dir)
        self.extra_columns = tuple(dict.fromkeys(str(column) for column in self.extra_columns))
        self._files_by_date = self._build_file_index()

    @property
    def available_dates(self) -> tuple[date, ...]:
        return tuple(sorted(self._files_by_date))

    def load_window(
        self,
        *,
        reference_date: date,
        lookback_days: int = 60,
        patient_ids: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        if lookback_days < 0:
            raise ValueError("lookback_days must be non-negative")
        patient_id_set = None if patient_ids is None else {str(patient_id) for patient_id in patient_ids}
        chunks: list[pd.DataFrame] = []
        paths = self._paths_for_window(reference_date, lookback_days)
        self.last_loaded_file_count = len(paths)
        for path in paths:
            self._validate_columns(path)
            raw = pd.read_parquet(path, columns=list(self._source_columns()))
            if patient_id_set is not None:
                mask = raw["patient_id"].astype(str).isin(patient_id_set)
                if not mask.any():
                    continue
                raw = raw.loc[mask].copy()
            normalized = self._normalize(raw, path)
            if not normalized.empty:
                chunks.append(normalized)
        if not chunks:
            return self._empty_history()
        return pd.concat(chunks, ignore_index=True).sort_values(
            ["patient_id", "prescription_date"],
            kind="mergesort",
        ).reset_index(drop=True)

    def _build_file_index(self) -> dict[date, Path]:
        files_by_date: dict[date, Path] = {}
        for path in sorted(self.raw_dir.glob(f"{self.file_prefix}*{self.file_suffix}")):
            parsed = self._date_from_path(path)
            if parsed is not None:
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
        return [
            path
            for file_date, path in sorted(self._files_by_date.items())
            if start_date <= file_date <= reference_date
        ]

    def _validate_columns(self, path: Path) -> None:
        available = set(pq.ParquetFile(path).schema_arrow.names)
        missing = [column for column in self._source_columns() if column not in available]
        if missing:
            raise ValueError(f"{path.name}: missing columns {missing}")

    def _normalize(self, df: pd.DataFrame, path: Path) -> pd.DataFrame:
        if df.empty:
            return self._empty_history()
        prescription_dates = pd.to_datetime(df["start_date"], errors="coerce")
        if prescription_dates.isna().any():
            raise ValueError(f"{path.name}: start_date contains invalid values")
        normalized = pd.DataFrame({
            "patient_id": df["patient_id"].astype(str),
            "drug_code": df["edi_code"].astype(str).str.strip(),
            "prescription_date": prescription_dates.dt.date,
        })
        for column in self.extra_columns:
            normalized[column] = df[column]
        return normalized.loc[:, list(self._output_columns())]

    def _empty_history(self) -> pd.DataFrame:
        return pd.DataFrame(columns=list(self._output_columns()))

    def _source_columns(self) -> tuple[str, ...]:
        return tuple(REQUIRED_COLUMNS) + tuple(
            column for column in self.extra_columns if column not in REQUIRED_COLUMNS
        )

    def _output_columns(self) -> tuple[str, ...]:
        return ("patient_id", "drug_code", "prescription_date") + tuple(
            column for column in self.extra_columns if column not in {"patient_id", "drug_code", "prescription_date"}
        )


def iter_patient_batches(
    histories: pd.DataFrame,
    patient_ids: Sequence[str],
    *,
    batch_size: int = 5000,
) -> Iterator[tuple[list[str], pd.DataFrame]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    patient_id_list = [str(patient_id) for patient_id in patient_ids]
    histories = histories.copy()
    if not histories.empty:
        histories["patient_id"] = histories["patient_id"].astype(str)
    for start in range(0, len(patient_id_list), batch_size):
        batch_ids = patient_id_list[start : start + batch_size]
        if histories.empty:
            batch_histories = histories.copy()
        else:
            batch_histories = histories[histories["patient_id"].isin(set(batch_ids))].copy()
        yield batch_ids, batch_histories
