from __future__ import annotations

from datetime import date

import pandas as pd


def test_load_demographics_uses_reference_year_age_and_sex_type_1_flag(tmp_path) -> None:
    from scripts.ops.eligibility_loader import load_demographics

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    pd.DataFrame({
        "patient_id": [" P1 ", "P2"],
        "byear": [1980, 1950],
        "age": [44, 74],
        "sex_type": ["1", "2"],
        "addr_cd": ["11110", "22220"],
    }).to_parquet(raw_dir / "eligibility_demographics.parquet", index=False)

    features = load_demographics(raw_dir, ["P1", "P2"], reference_date=date(2024, 12, 31))

    assert features["P1"] == (0.44, 1.0)
    assert features["P2"] == (0.74, 0.0)


def test_load_demographics_uses_null_safe_defaults(tmp_path) -> None:
    from scripts.ops.eligibility_loader import load_demographics

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    pd.DataFrame({
        "patient_id": ["P1"],
        "byear": [None],
        "age": [57],
        "sex_type": [None],
        "addr_cd": [None],
    }).to_parquet(raw_dir / "eligibility_demographics.parquet", index=False)

    features = load_demographics(raw_dir, ["P1", "P_missing"], reference_date=date(2024, 12, 31))

    assert features["P1"] == (0.57, 0.5)
    assert features["P_missing"] == (0.0, 0.5)


def test_load_demographics_rejects_missing_required_columns(tmp_path) -> None:
    from scripts.ops.eligibility_loader import load_demographics

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    pd.DataFrame({"patient_id": ["P1"], "age": [30]}).to_parquet(
        raw_dir / "eligibility_demographics.parquet",
        index=False,
    )

    try:
        load_demographics(raw_dir, ["P1"], reference_date=date(2024, 12, 31))
    except ValueError as exc:
        assert "eligibility_demographics.parquet missing required columns" in str(exc)
    else:
        raise AssertionError("expected ValueError")
