from __future__ import annotations

from datetime import date

import pandas as pd


def _history(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_same_class_distinct_drugs_counts_as_duplication() -> None:
    from scripts.ops.therapeutic_duplication_label import duplication_class_count

    result = duplication_class_count(
        _history([
            {"efmdc_clsf_no": "114", "drug_code": "D1"},
            {"efmdc_clsf_no": "114", "drug_code": "D2"},
        ]),
    )

    assert result == 1


def test_same_class_same_drug_repeated_is_not_duplication() -> None:
    from scripts.ops.therapeutic_duplication_label import duplication_class_count

    result = duplication_class_count(
        _history([
            {"efmdc_clsf_no": "114", "drug_code": "D1"},
            {"efmdc_clsf_no": "114", "drug_code": "D1"},
        ]),
    )

    assert result == 0


def test_null_and_blank_classes_are_skipped() -> None:
    from scripts.ops.therapeutic_duplication_label import duplication_class_count

    result = duplication_class_count(
        _history([
            {"efmdc_clsf_no": None, "drug_code": "D1"},
            {"efmdc_clsf_no": "", "drug_code": "D2"},
            {"efmdc_clsf_no": " ", "drug_code": "D3"},
            {"efmdc_clsf_no": "114", "drug_code": "D4"},
        ]),
    )

    assert result == 0


def test_multiple_duplicate_classes_counted() -> None:
    from scripts.ops.therapeutic_duplication_label import duplication_class_count

    result = duplication_class_count(
        _history([
            {"efmdc_clsf_no": "114", "drug_code": "D1"},
            {"efmdc_clsf_no": "114", "drug_code": "D2"},
            {"efmdc_clsf_no": "396", "drug_code": "D3"},
            {"efmdc_clsf_no": "396", "drug_code": "D4"},
            {"efmdc_clsf_no": "222", "drug_code": "D5"},
        ]),
    )

    assert result == 2


def test_label_respects_min_duplicate_classes_threshold() -> None:
    from scripts.ops.therapeutic_duplication_label import (
        assign_therapeutic_duplication_label,
    )

    history = _history([
        {"efmdc_clsf_no": "114", "drug_code": "D1"},
        {"efmdc_clsf_no": "114", "drug_code": "D2"},
    ])

    assert assign_therapeutic_duplication_label(history, min_duplicate_classes=1) == 1
    assert assign_therapeutic_duplication_label(history, min_duplicate_classes=2) == 0


def test_batch_labels_and_evaluable_patient_metrics() -> None:
    from scripts.ops.therapeutic_duplication_label import label_therapeutic_duplication

    histories = pd.DataFrame([
        {"patient_id": "P1", "efmdc_clsf_no": "114", "drug_code": "D1"},
        {"patient_id": "P1", "efmdc_clsf_no": "114", "drug_code": "D2"},
        {"patient_id": "P2", "efmdc_clsf_no": "396", "drug_code": "D3"},
        {"patient_id": "P3", "efmdc_clsf_no": None, "drug_code": "D4"},
    ])

    result = label_therapeutic_duplication(
        ["P1", "P2", "P3", "P4"],
        histories,
        min_duplicate_classes=1,
    )

    assert result.labels == {"P1": 1, "P2": 0, "P3": 0, "P4": 0}
    assert result.label_positive == 1
    assert result.duplication_class_counts == {"P1": 1, "P2": 0, "P3": 0, "P4": 0}
    assert result.evaluable_patient_count == 2
    assert result.null_efmdc_row_count == 1


class _FakeProvider:
    def get_history_batch(self, patient_ids, reference_date: date, lookback_days: int = 60):
        del reference_date, lookback_days
        return pd.DataFrame([
            {"patient_id": patient_ids[0], "efmdc_clsf_no": "114", "drug_code": "D1"},
            {"patient_id": patient_ids[0], "efmdc_clsf_no": "114", "drug_code": "D2"},
            {"patient_id": patient_ids[0], "efmdc_clsf_no": "396", "drug_code": "D3"},
            {"patient_id": patient_ids[0], "efmdc_clsf_no": "396", "drug_code": "D4"},
            {"patient_id": patient_ids[1], "efmdc_clsf_no": "114", "drug_code": "D1"},
            {"patient_id": patient_ids[2], "efmdc_clsf_no": None, "drug_code": "D9"},
        ])


def test_run_audit_report_has_required_metrics() -> None:
    from scripts.ops.therapeutic_duplication_label import (
        run_therapeutic_duplication_audit,
    )

    report = run_therapeutic_duplication_audit(
        provider=_FakeProvider(),
        patient_ids=["P1", "P2", "P3", "P4"],
        reference_date=date(2024, 11, 30),
        lookback_days=60,
        min_duplicate_classes=1,
    )

    assert report["n_patients"] == 4
    assert report["label_positive"] == 1
    assert report["label_positive_rate_pct"] == 25.0
    assert report["evaluable_patient_count"] == 2
    assert report["label_positive_rate_evaluable_pct"] == 50.0
    assert report["duplication_class_count_percentiles"]["max"] == 2.0
    assert report["patients_with_n_dup_classes_dist"] == {"0": 3, "1": 0, "2": 1, "3_plus": 0}
    assert report["top_duplicated_efmdc_classes"] == [{"efmdc_clsf_no": "114", "patient_class_count": 1}, {"efmdc_clsf_no": "396", "patient_class_count": 1}]
    assert report["top_null_drug_codes"] == [{"drug_code": "D9", "row_count": 1}]


def test_threshold_sensitivity_recommends_highest_threshold_in_target_range() -> None:
    from scripts.ops.therapeutic_duplication_label import (
        run_threshold_sensitivity_audit,
    )

    report = run_threshold_sensitivity_audit(
        provider=_FakeProvider(),
        patient_ids=["P1", "P2", "P3", "P4"],
        reference_date=date(2024, 11, 30),
        lookback_days=60,
        thresholds=[1, 2, 3],
        target_positive_rate_range=(10.0, 30.0),
    )

    assert report["recommended_threshold"] == 2
    assert report["threshold_results"] == [
        {"threshold": 1, "label_positive": 1, "label_positive_rate_pct": 25.0, "label_positive_rate_evaluable_pct": 50.0},
        {"threshold": 2, "label_positive": 1, "label_positive_rate_pct": 25.0, "label_positive_rate_evaluable_pct": 50.0},
        {"threshold": 3, "label_positive": 0, "label_positive_rate_pct": 0.0, "label_positive_rate_evaluable_pct": 0.0},
    ]
