"""HANA prescription-history boundary for operational DL serving.

The concrete HANA extractor is injected instead of imported here. That keeps
serving history contracts importable in lightweight test/runtime environments
where HANA or ETL dependencies are not installed.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Protocol

import pandas as pd

from scripts.datasets.contracts import (
    DL_DATASET_REQUIRED_COLUMNS,
    validate_lookback_days,
    validate_required_columns,
)


@dataclass(frozen=True)
class HistoryWindow:
    """Closed date window used for patient prescription history lookup."""

    patient_id: str
    reference_date: date
    lookback_days: int
    start_date: date
    end_date: date

    @classmethod
    def from_reference(
        cls,
        patient_id: str,
        reference_date: date,
        lookback_days: int,
    ) -> "HistoryWindow":
        lookback_value = validate_lookback_days(lookback_days)
        end_date = reference_date
        start_date = end_date - timedelta(days=lookback_value - 1)
        return cls(
            patient_id=str(patient_id),
            reference_date=end_date,
            lookback_days=lookback_value,
            start_date=start_date,
            end_date=end_date,
        )


class HANAHistoryProvider(Protocol):
    """Fetch normalized DL prescription events for one patient."""

    def fetch_patient_history(
        self,
        patient_id: str,
        reference_date: date,
        lookback_days: int,
    ) -> pd.DataFrame:
        ...


def validate_history_frame(df: pd.DataFrame, context: str = "history") -> pd.DataFrame:
    """Validate the minimal DL serving history schema."""
    validate_required_columns(df.columns, DL_DATASET_REQUIRED_COLUMNS)
    return df


def _history_dates(series: pd.Series) -> pd.Series:
    values = series.astype(str).str.strip()
    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.isna().any():
        bad = values[parsed.isna()].head(3).tolist()
        raise ValueError(f"prescription_date contains invalid values: {bad}")
    return parsed.dt.date


def _sort_history(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.reset_index(drop=True)
    return df.sort_values(["prescription_date", "drug_code"]).reset_index(drop=True)


class InMemoryHANAHistoryProvider:
    """Small test/fallback provider for already-normalized DL history frames."""

    def __init__(self, history_df: pd.DataFrame) -> None:
        self._history_df = validate_history_frame(
            history_df.copy(),
            context="in-memory history",
        )

    def fetch_patient_history(
        self,
        patient_id: str,
        reference_date: date,
        lookback_days: int,
    ) -> pd.DataFrame:
        window = HistoryWindow.from_reference(patient_id, reference_date, lookback_days)
        df = self._history_df
        dates = _history_dates(df["prescription_date"])
        mask = (
            (df["patient_id"].astype(str) == window.patient_id)
            & (dates >= window.start_date)
            & (dates <= window.end_date)
        )
        result = df.loc[mask].copy()
        validate_history_frame(result, context="in-memory history result")
        return _sort_history(result)


class HANAExtractorHistoryProvider:
    """Normalize T30/T60 rows from an injected HANAExtractor-like object."""

    def __init__(self, extractor) -> None:
        self._extractor = extractor

    def fetch_patient_history(
        self,
        patient_id: str,
        reference_date: date,
        lookback_days: int,
    ) -> pd.DataFrame:
        window = HistoryWindow.from_reference(patient_id, reference_date, lookback_days)
        patient_ids = [window.patient_id]
        t30 = self._extractor.fetch_t30_by_date(
            window.start_date,
            window.end_date,
            patient_ids=patient_ids,
        )
        t60 = self._extractor.fetch_t60_by_date(
            window.start_date,
            window.end_date,
            patient_ids=patient_ids,
        )
        frames = [
            self._events_from_source(t30, "t30"),
            self._events_from_source(t60, "t60"),
        ]
        history = pd.concat(frames, ignore_index=True)
        validate_history_frame(history, context="hana extractor history")
        return _sort_history(history)

    def _events_from_source(self, df: pd.DataFrame, source: str) -> pd.DataFrame:
        cols = self._extractor.cols[source]
        out_cols = [
            "patient_id",
            "drug_code",
            "prescription_date",
            "source",
            "edi_code",
            "total_days",
        ]
        if df.empty:
            return pd.DataFrame(columns=out_cols)

        patient_col = cols["patient_id"]
        date_col = cols["start_date"]
        edi_col = cols.get("edi_code")
        total_days_col = cols.get("total_days")

        # 학습↔서빙 인코더 정합(약물코드 네임스페이스): DL drug_vocab 은 EDI 코드
        # (MCARE_DIV_CD) 빈도로 구축되고, 학습 history 로더(full_cohort_history_loader/
        # multi_day_parquet_provider)도 drug_code 컬럼에 edi_code 값을 담는다. 따라서 서빙
        # 도 drug_code 를 EDI 네임스페이스로 채워야 dl_predictor._encode_history 가 vocab 에
        # 정확히 매칭된다. WK_COMPN_CD/GNL_NM_CD(주성분/일반명코드)를 쓰면 거의 전부 OOV.
        # edi_col 미설정/부재는 silent 빈 인코딩을 막기 위해 에러로 처리한다.
        if not edi_col:
            raise ValueError(f"{source} history config has no edi_code (MCARE_DIV_CD) mapping")
        required = [patient_col, date_col, edi_col]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"{source} history missing columns: {missing}")

        edi_code = df[edi_col].astype(str).str.strip()
        events = pd.DataFrame({
            "patient_id": df[patient_col].astype(str),
            "drug_code": edi_code,
            "prescription_date": df[date_col].astype(str),
            "source": source.upper(),
        })
        events["edi_code"] = edi_code
        events["total_days"] = (
            df[total_days_col] if total_days_col and total_days_col in df.columns else None
        )
        return events[out_cols]
