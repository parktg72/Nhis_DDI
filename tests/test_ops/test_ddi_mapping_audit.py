from __future__ import annotations

from datetime import date

import pandas as pd


class _FakeMaster:
    def __init__(self, mapping: dict[str, list[str]]) -> None:
        self.mapping = mapping

    def get_ddi_ids(self, wk_compn_cd: str) -> list[str]:
        return self.mapping.get(str(wk_compn_cd), [])


def test_canonical_pair_sorts_ids() -> None:
    from scripts.ops.ddi_mapping_audit import canonical_pair

    assert canonical_pair("D000718", "D000669") == ("D000669", "D000718")


def test_patient_label_detects_contraindicated_d_code_pair() -> None:
    from scripts.ops.ddi_mapping_audit import label_ddi_contraindication

    histories = pd.DataFrame([
        {"patient_id": "P1", "wk_compn_cd": "WK_A", "prescription_date": date(2024, 11, 1), "end_date": date(2024, 11, 3)},
        {"patient_id": "P1", "wk_compn_cd": "WK_B", "prescription_date": date(2024, 11, 2), "end_date": date(2024, 11, 4)},
        {"patient_id": "P2", "wk_compn_cd": "WK_A", "prescription_date": date(2024, 11, 1), "end_date": date(2024, 11, 1)},
        {"patient_id": "P2", "wk_compn_cd": "WK_C", "prescription_date": date(2024, 11, 10), "end_date": date(2024, 11, 11)},
    ])
    master = _FakeMaster({"WK_A": ["D001"], "WK_B": ["D002"], "WK_C": ["D999"]})

    result = label_ddi_contraindication(
        ["P1", "P2", "P3"],
        histories,
        master=master,
        contraindicated_pairs={("D001", "D002")},
    )

    assert result.labels == {"P1": 1, "P2": 0, "P3": 0}
    assert result.label_positive == 1
    assert result.hit_pair_counts == {"P1": 1, "P2": 0, "P3": 0}
    assert result.mapped_patients == 2
    assert result.overlap_positive_patients == 1


def test_pair_namespace_ignores_db_ids_for_hira_pairs() -> None:
    from scripts.ops.ddi_mapping_audit import label_ddi_contraindication

    histories = pd.DataFrame([
        {"patient_id": "P1", "wk_compn_cd": "WK_A"},
        {"patient_id": "P1", "wk_compn_cd": "WK_B"},
    ])
    master = _FakeMaster({"WK_A": ["DB001"], "WK_B": ["D002"]})

    result = label_ddi_contraindication(
        ["P1"],
        histories,
        master=master,
        contraindicated_pairs={("D001", "D002")},
    )

    assert result.labels == {"P1": 0}
    assert result.db_code_count == 1
    assert result.d_code_count == 1


def test_audit_report_includes_mapping_and_dominance_metrics() -> None:
    from scripts.ops.ddi_mapping_audit import run_ddi_mapping_audit

    class Provider:
        def get_history_batch(self, patient_ids, reference_date: date, lookback_days: int = 60):
            del reference_date, lookback_days
            return pd.DataFrame([
                {"patient_id": patient_ids[0], "wk_compn_cd": "WK_A", "prescription_date": date(2024, 11, 1), "end_date": date(2024, 11, 3)},
                {"patient_id": patient_ids[0], "wk_compn_cd": "WK_B", "prescription_date": date(2024, 11, 2), "end_date": date(2024, 11, 4)},
                {"patient_id": patient_ids[1], "wk_compn_cd": "WK_A", "prescription_date": date(2024, 11, 1), "end_date": date(2024, 11, 1)},
                {"patient_id": patient_ids[1], "wk_compn_cd": "WK_C", "prescription_date": date(2024, 11, 10), "end_date": date(2024, 11, 11)},
                {"patient_id": patient_ids[2], "wk_compn_cd": "WK_MISSING", "prescription_date": date(2024, 11, 2), "end_date": date(2024, 11, 2)},
            ])

    master = _FakeMaster({"WK_A": ["D001"], "WK_B": ["D002"], "WK_C": ["DB003"]})

    report = run_ddi_mapping_audit(
        provider=Provider(),
        patient_ids=["P1", "P2", "P3", "P4"],
        reference_date=date(2024, 11, 30),
        lookback_days=60,
        master=master,
        contraindicated_pairs={("D001", "D002")},
    )

    assert report["n_patients"] == 4
    assert report["label_positive"] == 1
    assert report["label_positive_rate_pct"] == 25.0
    assert report["any_mapped_patients"] == 2
    assert report["mapped_patients"] == 2
    assert report["zero_mapped_patients"] == 2
    assert report["mapping_coverage_pct"] == 50.0
    assert report["wk_coverage_pct"] == 75.0
    assert report["unmapped_drug_row_count"] == 1
    assert report["unmapped_drug_row_rate_pct"] == 20.0
    assert report["top_hit_pairs"] == [{"drug_a_id": "D001", "drug_b_id": "D002", "patient_count": 1}]
    assert report["top_pair_dominance_pct"] == 100.0
    assert report["temporal_overlap_available"] is True
    assert report["overlap_positive_patients"] == 1


def test_audit_can_exclude_dominant_drug_ids() -> None:
    from scripts.ops.ddi_mapping_audit import run_ddi_mapping_audit

    class Provider:
        def get_history_batch(self, patient_ids, reference_date: date, lookback_days: int = 60):
            del reference_date, lookback_days
            return pd.DataFrame([
                {"patient_id": patient_ids[0], "wk_compn_cd": "WK_A"},
                {"patient_id": patient_ids[0], "wk_compn_cd": "WK_B"},
                {"patient_id": patient_ids[1], "wk_compn_cd": "WK_A"},
                {"patient_id": patient_ids[1], "wk_compn_cd": "WK_C"},
            ])

    master = _FakeMaster({"WK_A": ["D001"], "WK_B": ["D002"], "WK_C": ["D003"]})

    report = run_ddi_mapping_audit(
        provider=Provider(),
        patient_ids=["P1", "P2"],
        reference_date=date(2024, 11, 30),
        lookback_days=60,
        master=master,
        contraindicated_pairs={("D001", "D002"), ("D001", "D003")},
        excluded_drug_ids={"D001"},
    )

    assert report["excluded_drug_ids"] == ["D001"]
    assert report["excluded_pair_count"] == 2
    assert report["active_contraindicated_pair_count"] == 0
    assert report["label_positive"] == 0
