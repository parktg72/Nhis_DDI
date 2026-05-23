"""Parquet-backed prescription history provider for offline DL verification."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from scripts.datasets.contracts import DL_DATASET_REQUIRED_COLUMNS
from serving.hana_history import validate_history_frame


@dataclass
class ParquetHistoryProvider:
    """Read a static daily prescription snapshot as DL history events.

    This provider is an ops verification helper for local parquet samples. It
    intentionally does not apply reference_date/lookback_days filtering because
    daily extracts such as records_20241001.parquet are static snapshots, not a
    queryable date-range source.
    """

    path: str | Path

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self._history_df: pd.DataFrame | None = None

    def fetch_patient_history(
        self,
        patient_id: str,
        reference_date: date,
        lookback_days: int,
    ) -> pd.DataFrame:
        del reference_date, lookback_days
        history = self._load_history()
        result = history.loc[history["patient_id"].astype(str) == str(patient_id)].copy()
        if result.empty:
            result = pd.DataFrame(columns=DL_DATASET_REQUIRED_COLUMNS)
        validate_history_frame(result, context="parquet history")
        return result.reset_index(drop=True)

    def _load_history(self) -> pd.DataFrame:
        if self._history_df is None:
            raw = pd.read_parquet(self.path)
            self._history_df = self._normalize(raw)
        return self._history_df

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        required = ["patient_id", "edi_code", "start_date"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"parquet history missing columns: {missing}")

        normalized = df.drop_duplicates().copy()
        prescription_dates = pd.to_datetime(
            normalized["start_date"],
            errors="coerce",
        )
        if prescription_dates.isna().any():
            bad = normalized.loc[prescription_dates.isna(), "start_date"].head(3).tolist()
            raise ValueError(f"parquet history start_date contains invalid values: {bad}")

        history = pd.DataFrame({
            "patient_id": normalized["patient_id"].astype(str),
            "drug_code": normalized["edi_code"].astype(str).str.strip(),
            "prescription_date": prescription_dates.dt.strftime("%Y%m%d"),
        })
        history = history.drop_duplicates(
            subset=list(DL_DATASET_REQUIRED_COLUMNS),
            keep="first",
        )
        validate_history_frame(history, context="parquet history")
        return history.sort_values(["prescription_date", "drug_code"]).reset_index(drop=True)
