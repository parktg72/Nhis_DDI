from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.etl.drug_master import DrugMaster


def _write_drug_master_parquet(path: Path) -> None:
    pd.DataFrame(
        [
            {
                "ingr_code": "146801ATB",
                "is_combo": False,
                "ingr_name_raw": "diphenylhydantoin sodium   0.1g",
                "components": "diphenylhydantoin",
                "ingr_count": 1,
            },
            {
                "ingr_code": "229705ATR",
                "is_combo": False,
                "ingr_name_raw": "sodium valproate   0.3g",
                "components": "sodium valproate",
                "ingr_count": 1,
            },
        ]
    ).to_parquet(path, index=False)


def _write_ddi_matrix(path: Path) -> None:
    pd.DataFrame(
        [
            {
                "drug_a_name": "phenytoin",
                "drug_a_id": "DB00252",
                "drug_b_name": "warfarin",
                "drug_b_id": "D000669",
                "severity": "Major",
                "description": "fixture",
                "source": "fixture",
            },
            {
                "drug_a_name": "valproic acid",
                "drug_a_id": "D000535",
                "drug_b_name": "warfarin",
                "drug_b_id": "D000669",
                "severity": "Major",
                "description": "fixture",
                "source": "fixture",
            },
            {
                "drug_a_name": "loxoprofen",
                "drug_a_id": "DB09212",
                "drug_b_name": "warfarin",
                "drug_b_id": "D000669",
                "severity": "Major",
                "description": "fixture",
                "source": "fixture",
            },
            {
                "drug_a_name": "fimasartan",
                "drug_a_id": "DB09279",
                "drug_b_name": "warfarin",
                "drug_b_id": "D000669",
                "severity": "Major",
                "description": "fixture",
                "source": "fixture",
            },
        ]
    ).to_parquet(path, index=False)


def test_load_parquet_applies_synonyms_after_ddi_index_is_available(tmp_path: Path):
    master_path = tmp_path / "hira_drug_master.parquet"
    ddi_path = tmp_path / "ddi_matrix_final.parquet"
    _write_drug_master_parquet(master_path)
    _write_ddi_matrix(ddi_path)

    master = DrugMaster.load_parquet(master_path, ddi_path)

    assert master.get_ddi_id("diphenylhydantoin") == "DB00252"
    assert master.get_ddi_ids("146801ATB") == ["DB00252"]
    assert master.get_ddi_id("sodium valproate") == "D000535"
    assert master.get_ddi_ids("229705ATR") == ["D000535"]


def test_load_parquet_repairs_stale_cached_components_from_raw_names(tmp_path: Path):
    master_path = tmp_path / "hira_drug_master.parquet"
    ddi_path = tmp_path / "ddi_matrix_final.parquet"
    pd.DataFrame(
        [
            {
                "ingr_code": "186101ATB",
                "is_combo": False,
                "ingr_name_raw": "loxoprofen sodium hydrate (as loxoprofen sodium   60mg)",
                "components": "loxoprofen sodium",
                "ingr_count": 1,
            },
            {
                "ingr_code": "716100ATB",
                "is_combo": True,
                "ingr_name_raw": "S-amlodipine besylate (as S-amlodipine   2.5mg), fimasartan potassium trihydrate (as fimasartan potassium   60mg)",
                "components": "s-amlodipine|fimasartan potassium",
                "ingr_count": 2,
            },
        ]
    ).to_parquet(master_path, index=False)
    _write_ddi_matrix(ddi_path)

    master = DrugMaster.load_parquet(master_path, ddi_path)

    assert master.get_components("186101ATB") == ["loxoprofen"]
    assert master.get_ddi_ids("186101ATB") == ["DB09212"]
    assert master.get_components("716100ATB") == ["s-amlodipine", "fimasartan"]
    assert master.get_ddi_ids("716100ATB") == ["DB09279"]


def test_load_parquet_does_not_repair_when_reparsed_ids_drop_current_ids(tmp_path: Path):
    master_path = tmp_path / "hira_drug_master.parquet"
    ddi_path = tmp_path / "ddi_matrix_final.parquet"
    pd.DataFrame(
        [
            {
                "ingr_code": "SAFE001",
                "is_combo": True,
                "ingr_name_raw": "loxoprofen sodium hydrate (as loxoprofen sodium   60mg), fimasartan potassium trihydrate (as fimasartan potassium   60mg), warfarin   5mg",
                "components": "phenytoin|valproic acid",
                "ingr_count": 2,
            },
        ]
    ).to_parquet(master_path, index=False)
    _write_ddi_matrix(ddi_path)

    master = DrugMaster.load_parquet(master_path, ddi_path)

    assert master.get_components("SAFE001") == ["phenytoin", "valproic acid"]
    assert master.get_ddi_ids("SAFE001") == ["DB00252", "D000535"]


def test_load_parquet_does_not_create_literal_nan_component(tmp_path: Path):
    master_path = tmp_path / "hira_drug_master.parquet"
    ddi_path = tmp_path / "ddi_matrix_final.parquet"
    pd.DataFrame(
        [
            {
                "ingr_code": "BLANK001",
                "is_combo": False,
                "ingr_name_raw": "",
                "components": float("nan"),
                "ingr_count": 0,
            },
        ]
    ).to_parquet(master_path, index=False)
    _write_ddi_matrix(ddi_path)

    master = DrugMaster.load_parquet(master_path, ddi_path)

    assert master.get_components("BLANK001") == []
    assert master.get_ddi_ids("BLANK001") == []


def test_load_parquet_skips_repair_if_reparse_would_drop_unrepresented_component(tmp_path: Path):
    master_path = tmp_path / "hira_drug_master.parquet"
    ddi_path = tmp_path / "ddi_matrix_final.parquet"
    pd.DataFrame(
        [
            {
                "ingr_code": "DROP001",
                "is_combo": True,
                "ingr_name_raw": "phenytoin   100mg, loxoprofen sodium hydrate (as loxoprofen sodium   60mg)",
                "components": "clinically important unmapped|phenytoin",
                "ingr_count": 2,
            },
        ]
    ).to_parquet(master_path, index=False)
    _write_ddi_matrix(ddi_path)

    master = DrugMaster.load_parquet(master_path, ddi_path)

    assert master.get_components("DROP001") == ["clinically important unmapped", "phenytoin"]
    assert master.get_ddi_ids("DROP001") == ["DB00252"]


def test_load_parquet_without_ddi_matrix_keeps_cached_components(tmp_path: Path):
    master_path = tmp_path / "hira_drug_master.parquet"
    missing_ddi_path = tmp_path / "missing_ddi_matrix.parquet"
    pd.DataFrame(
        [
            {
                "ingr_code": "186101ATB",
                "is_combo": False,
                "ingr_name_raw": "loxoprofen sodium hydrate (as loxoprofen sodium   60mg)",
                "components": "loxoprofen sodium",
                "ingr_count": 1,
            },
        ]
    ).to_parquet(master_path, index=False)

    master = DrugMaster.load_parquet(master_path, missing_ddi_path)

    assert master.get_components("186101ATB") == ["loxoprofen sodium"]
    assert master.get_ddi_ids("186101ATB") == []
