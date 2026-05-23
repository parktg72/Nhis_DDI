from __future__ import annotations

from datetime import date

import pandas as pd


def _history(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_known_adr_icd10_returns_label_1() -> None:
    from scripts.ops.label_audit import assign_adr_label_from_sick_code

    result = assign_adr_label_from_sick_code(_history([{"sick_code": "K92.1"}]))

    assert result == 1


def test_no_adr_icd10_returns_label_0() -> None:
    from scripts.ops.label_audit import assign_adr_label_from_sick_code

    result = assign_adr_label_from_sick_code(_history([{"sick_code": "J00"}]))

    assert result == 0


def test_empty_sick_codes_returns_label_0() -> None:
    from scripts.ops.label_audit import assign_adr_label_from_sick_code

    result = assign_adr_label_from_sick_code(
        _history([{"sick_code": None}, {"sick_code": ""}, {"sick_code": float("nan")}]),
    )

    assert result == 0


def test_partial_icd10_prefix_match() -> None:
    from scripts.ops.label_audit import assign_adr_label_from_sick_code

    dotted = assign_adr_label_from_sick_code(_history([{"sick_code": "K92.1"}]))
    undotted = assign_adr_label_from_sick_code(_history([{"sick_code": "K921"}]))

    assert dotted == 1
    assert undotted == 1


def test_sick_code_matching_is_case_and_space_insensitive() -> None:
    from scripts.ops.label_audit import assign_adr_label_from_sick_code

    result = assign_adr_label_from_sick_code(_history([{"sick_code": " k92.1 "}]))

    assert result == 1


def test_batch_label_distribution() -> None:
    from scripts.ops.label_audit import label_patient_histories

    histories = pd.DataFrame([
        {"patient_id": "P1", "sick_code": "K92.1"},
        {"patient_id": "P2", "sick_code": "J00"},
        {"patient_id": "P3", "sick_code": "N17"},
        {"patient_id": "P4", "sick_code": None},
    ])

    result = label_patient_histories(["P1", "P2", "P3", "P4"], histories)

    assert result.labels == {"P1": 1, "P2": 0, "P3": 1, "P4": 0}
    assert result.label_positive == 2
    assert result.icd10_type_counts["bleeding"] == 1
    assert result.icd10_type_counts["acute_kidney_injury"] == 1
    assert result.null_sick_code_count == 1


class _FakeProvider:
    def get_history_batch(self, patient_ids, reference_date: date, lookback_days: int = 60):
        del reference_date, lookback_days
        return pd.DataFrame([
            {"patient_id": patient_ids[0], "sick_code": "K92.1"},
            {"patient_id": patient_ids[1], "sick_code": "J00"},
        ])


def test_run_label_audit_report_has_aggregate_metrics() -> None:
    from scripts.ops.label_audit import run_label_audit

    report = run_label_audit(
        provider=_FakeProvider(),
        patient_ids=["P1", "P2"],
        reference_date=date(2024, 11, 30),
        lookback_days=60,
    )

    assert report["n_patients"] == 2
    assert report["label_positive"] == 1
    assert report["label_positive_rate_pct"] == 50.0
    assert report["null_sick_code_rate_pct"] == 0.0
    assert report["icd10_type_counts"]["bleeding"] == 1
