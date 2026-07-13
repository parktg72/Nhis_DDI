from __future__ import annotations

import gc
import sys
import weakref
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.etl import drug_ontology as ontology_module
from scripts.etl.drug_master import DrugMaster
from scripts.etl.drug_ontology import (
    DDIMatrixLookup,
    DrugOntology,
    clear_lookup_cache,
    get_lookup,
    higher_severity,
    severity_rank,
)
from scripts.etl.models import DrugOverlapPair


def _matrix(rows: list[dict[str, str | None]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _master() -> DrugMaster:
    master = DrugMaster()
    master._code_to_components = {
        "WK_A": ["alpha"],
        "WK_B": ["beta"],
        "WK_C": ["gamma"],
        "WK_COMBO": ["alpha", "gamma"],
    }
    master._name_to_ddi_id = {
        "alpha": "D_A",
        "beta": "D_B",
        "gamma": "D_C",
    }
    return master


def _pair(a: str, b: str) -> DrugOverlapPair:
    return DrugOverlapPair(
        patient_id="P1",
        drug_a_wk_compn=a,
        drug_a_edi=None,
        drug_a_atc=None,
        drug_a_name=None,
        drug_b_wk_compn=b,
        drug_b_edi=None,
        drug_b_atc=None,
        drug_b_name=None,
        overlap_start=date(2024, 1, 1),
        overlap_end=date(2024, 1, 2),
        overlap_days=2,
        window_start=date(2024, 1, 1),
        window_end=date(2024, 3, 30),
    )


def setup_function() -> None:
    clear_lookup_cache()


def test_severity_rank_and_higher_severity_order_known_values() -> None:
    assert severity_rank("Contraindicated") > severity_rank("Major")
    assert severity_rank("Major") > severity_rank("Moderate")
    assert severity_rank("Moderate") > severity_rank("Minor")
    assert severity_rank(None) == 0
    assert severity_rank("Unknown") == 0

    assert higher_severity("Minor", "Major") == "Major"
    assert higher_severity(None, "Moderate") == "Moderate"
    assert higher_severity("Contraindicated", "Major") == "Contraindicated"
    assert higher_severity(None, None) is None


def test_ddi_matrix_lookup_is_symmetric() -> None:
    lookup = DDIMatrixLookup.from_matrix(
        _matrix([
            {"drug_a_id": "D_A", "drug_b_id": "D_B", "severity": "Major"},
        ])
    )

    assert lookup.severity("D_A", "D_B") == "Major"
    assert lookup.severity("D_B", "D_A") == "Major"


def test_ddi_matrix_duplicate_pair_keeps_highest_severity() -> None:
    lookup = DDIMatrixLookup.from_matrix(
        _matrix([
            {"drug_a_id": "D_A", "drug_b_id": "D_B", "severity": "Minor"},
            {"drug_a_id": "D_B", "drug_b_id": "D_A", "severity": "Contraindicated"},
            {"drug_a_id": "D_A", "drug_b_id": "D_B", "severity": "Moderate"},
        ])
    )

    assert lookup.severity("D_A", "D_B") == "Contraindicated"


def test_ddi_matrix_lookup_skips_blank_ids() -> None:
    lookup = DDIMatrixLookup.from_matrix(
        _matrix([
            {"drug_a_id": "D_A", "drug_b_id": "", "severity": "Major"},
            {"drug_a_id": None, "drug_b_id": "D_B", "severity": "Major"},
            {"drug_a_id": "D_A", "drug_b_id": "D_B", "severity": "Minor"},
        ])
    )

    assert lookup.severity("D_A", "") is None
    assert lookup.severity("D_A", "D_B") == "Minor"


def test_missing_required_columns_yields_empty_lookup() -> None:
    lookup = DDIMatrixLookup.from_matrix(
        pd.DataFrame([{"drug_a_id": "D_A", "severity": "Major"}])
    )

    assert lookup.severity("D_A", "D_B") is None


def test_best_severity_uses_component_cross_product() -> None:
    lookup = DDIMatrixLookup.from_matrix(
        _matrix([
            {"drug_a_id": "D_A", "drug_b_id": "D_X", "severity": "Minor"},
            {"drug_a_id": "D_C", "drug_b_id": "D_X", "severity": "Major"},
            {"drug_a_id": "D_C", "drug_b_id": "D_Y", "severity": "Moderate"},
        ])
    )

    assert lookup.best_severity(["D_A", "D_C"], ["D_X", "D_Y"]) == "Major"
    assert lookup.best_severity(["D_A"], ["D_Z"]) is None


def test_get_lookup_uses_new_matrix_after_cache_clear() -> None:
    first = _matrix([
        {"drug_a_id": "D_A", "drug_b_id": "D_B", "severity": "Minor"},
    ])
    second = _matrix([
        {"drug_a_id": "D_A", "drug_b_id": "D_B", "severity": "Major"},
    ])

    assert get_lookup(first).severity("D_A", "D_B") == "Minor"
    clear_lookup_cache()

    assert get_lookup(second).severity("D_A", "D_B") == "Major"


def test_get_lookup_ignores_stale_id_cache_entry_with_dead_weakref() -> None:
    stale_matrix = _matrix([
        {"drug_a_id": "D_A", "drug_b_id": "D_B", "severity": "Minor"},
    ])
    fresh_matrix = _matrix([
        {"drug_a_id": "D_A", "drug_b_id": "D_B", "severity": "Major"},
    ])
    stale_ref = weakref.ref(stale_matrix)
    stale_lookup = DDIMatrixLookup.from_matrix(stale_matrix)
    ontology_module._lookup_cache[id(fresh_matrix)] = SimpleNamespace(
        frame_ref=stale_ref,
        lookup=stale_lookup,
    )

    del stale_matrix
    gc.collect()

    assert stale_ref() is None
    lookup = get_lookup(fresh_matrix)
    assert lookup.severity("D_A", "D_B") == "Major"
    assert ontology_module._lookup_cache[id(fresh_matrix)].frame_ref() is fresh_matrix


def test_drug_ontology_pair_severity_and_unmapped_wk() -> None:
    ontology = DrugOntology(
        _master(),
        _matrix([
            {"drug_a_id": "D_A", "drug_b_id": "D_B", "severity": "Major"},
            {"drug_a_id": "D_C", "drug_b_id": "D_B", "severity": "Moderate"},
        ]),
    )

    assert ontology.components("WK_A") == ["alpha"]
    assert ontology.ddi_ids("WK_COMBO") == ["D_A", "D_C"]
    assert ontology.pair_severity("WK_A", "WK_B") == "Major"
    assert ontology.pair_severity("WK_UNKNOWN", "WK_B") is None


def test_drug_ontology_pair_severities_preserves_pair_identity() -> None:
    pair = _pair("WK_COMBO", "WK_B")
    missing = _pair("WK_UNKNOWN", "WK_B")
    ontology = DrugOntology(
        _master(),
        _matrix([
            {"drug_a_id": "D_A", "drug_b_id": "D_B", "severity": "Minor"},
            {"drug_a_id": "D_C", "drug_b_id": "D_B", "severity": "Major"},
        ]),
    )

    result = ontology.pair_severities([pair, missing])

    assert result == [(pair, "Major")]
    assert result[0][0] is pair
