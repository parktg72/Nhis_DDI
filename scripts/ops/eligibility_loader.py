"""Load dense eligibility-derived features for sparse research datasets."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Sequence

import pandas as pd


DEFAULT_DEMOGRAPHICS_FEATURE = (0.0, 0.5)
REQUIRED_DEMOGRAPHICS_COLUMNS = {"patient_id", "byear", "age", "sex_type"}


def load_demographics(
    raw_dir: str | Path,
    patient_ids: Sequence[str],
    *,
    reference_date: date,
) -> dict[str, tuple[float, float]]:
    """Return age and sex scalar features keyed by normalized patient id.

    Features are ``(age_years / 100, sex_type_1_flag)``. Missing patients use
    ``(0.0, 0.5)`` so cohort row counts do not change when demographics are
    added.
    """
    path = Path(raw_dir) / "eligibility_demographics.parquet"
    if not path.exists():
        raise FileNotFoundError(f"eligibility demographics file not found: {path}")

    df = pd.read_parquet(path)
    missing_columns = REQUIRED_DEMOGRAPHICS_COLUMNS - set(df.columns)
    if missing_columns:
        raise ValueError(
            "eligibility_demographics.parquet missing required columns: "
            f"{sorted(missing_columns)}"
        )

    normalized = df.copy()
    normalized["patient_id"] = normalized["patient_id"].astype(str).str.strip()
    normalized = normalized.drop_duplicates(subset=["patient_id"], keep="first")

    by_patient = {
        row.patient_id: (
            _age_feature(row.byear, row.age, reference_date),
            _sex_type_1_flag(row.sex_type),
        )
        for row in normalized.itertuples(index=False)
        if row.patient_id
    }
    return {
        str(patient_id).strip(): by_patient.get(str(patient_id).strip(), DEFAULT_DEMOGRAPHICS_FEATURE)
        for patient_id in patient_ids
    }


def _age_feature(byear: object, age: object, reference_date: date) -> float:
    year = _to_float(byear)
    if year is not None and 1900 <= year <= reference_date.year:
        age_years = reference_date.year - int(year)
    else:
        age_value = _to_float(age)
        age_years = int(age_value) if age_value is not None else 0
    age_years = min(max(age_years, 0), 120)
    return round(age_years / 100.0, 6)


def _sex_type_1_flag(value: object) -> float:
    if pd.isna(value):
        return 0.5
    normalized = str(value).strip()
    if normalized == "1":
        return 1.0
    if normalized == "2":
        return 0.0
    return 0.5


def _to_float(value: object) -> float | None:
    if pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
