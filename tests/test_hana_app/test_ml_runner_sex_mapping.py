from datetime import date

import pytest

from hana_app.core.ml_runner import FEATURE_COLS, _patient_features_to_row
from scripts.etl.models import PatientFeatures


def _make(sex: str | None) -> PatientFeatures:
    return PatientFeatures(
        patient_id="P001",
        window_start=date(2024, 7, 1),
        window_end=date(2024, 9, 28),
        sex=sex,
    )


@pytest.mark.parametrize(
    ("raw_sex", "expected"),
    [("1", 1.0), ("2", 0.0), (None, 0.5), ("", 0.5), ("9", 0.5)],
)
def test_patient_features_sex_mapping(raw_sex, expected):
    row = _patient_features_to_row(_make(sex=raw_sex))

    assert row["sex_type"] == raw_sex
    assert row["sex_m"] == expected
    assert isinstance(row["sex_m"], float)


def test_feature_cols_order_is_unchanged():
    assert FEATURE_COLS == [
        "drug_count",
        "drug_count_7d",
        "institution_count",
        "ddi_contraindicated",
        "ddi_major",
        "ddi_moderate",
        "ddi_minor",
        "triple_whammy",
        "qt_risk_count",
        "dup_same_ingredient",
        "dup_atc5",
        "dup_atc4",
        "dup_atc3",
        "dup_efmdc",
        "has_high_risk_drug",
        "has_renal_risk_drug",
        "has_hepatic_risk_drug",
        "cyp_risk_score",
        "cyp_max_enzyme_risk",
        "cyp_high_risk_pairs",
        "age",
        "sex_m",
    ]
