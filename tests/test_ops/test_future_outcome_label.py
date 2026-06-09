from __future__ import annotations

from datetime import date

import pandas as pd


def _hist(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_onset_positive_when_oct_below_threshold_and_nov_reaches_threshold() -> None:
    from scripts.ops.future_outcome_label import label_future_multi_institution_onset

    oct_histories = _hist([
        {"patient_id": "P1", "institution_id": "A"},
        {"patient_id": "P1", "institution_id": "B"},
    ])
    nov_histories = _hist([
        {"patient_id": "P1", "institution_id": "A"},
        {"patient_id": "P1", "institution_id": "B"},
        {"patient_id": "P1", "institution_id": "C"},
    ])

    result = label_future_multi_institution_onset(
        ["P1"],
        oct_histories,
        nov_histories,
        threshold=3,
    )

    assert result.labels == {"P1": 1}
    assert result.label_positive == 1
    assert result.clean_onset_positive == 0
    assert result.escalation_positive == 1
    assert result.onset_eligible_n == 1


def test_negative_when_oct_below_threshold_and_nov_stays_below_threshold() -> None:
    from scripts.ops.future_outcome_label import label_future_multi_institution_onset

    result = label_future_multi_institution_onset(
        ["P1"],
        _hist([{"patient_id": "P1", "institution_id": "A"}]),
        _hist([{"patient_id": "P1", "institution_id": "A"}]),
        threshold=3,
    )

    assert result.labels == {"P1": 0}
    assert result.label_positive == 0
    assert result.n_evaluable == 1
    assert result.n_censored == 0


def test_missing_nov_records_are_censored_not_negative() -> None:
    from scripts.ops.future_outcome_label import label_future_multi_institution_onset

    result = label_future_multi_institution_onset(
        ["P1"],
        _hist([{"patient_id": "P1", "institution_id": "A"}]),
        _hist([]),
        threshold=3,
    )

    assert result.labels == {}
    assert result.n_censored == 1
    assert result.n_evaluable == 0
    assert result.censored_patient_ids == ["P1"]


def test_oct_at_threshold_is_persistence_excluded() -> None:
    from scripts.ops.future_outcome_label import label_future_multi_institution_onset

    oct_histories = _hist([
        {"patient_id": "P1", "institution_id": "A"},
        {"patient_id": "P1", "institution_id": "B"},
        {"patient_id": "P1", "institution_id": "C"},
    ])
    nov_histories = _hist([
        {"patient_id": "P1", "institution_id": "A"},
        {"patient_id": "P1", "institution_id": "B"},
        {"patient_id": "P1", "institution_id": "C"},
    ])

    result = label_future_multi_institution_onset(
        ["P1"],
        oct_histories,
        nov_histories,
        threshold=3,
    )

    assert result.labels == {}
    assert result.persistence_cohort_size == 1
    assert result.persistence_rate_pct == 100.0
    assert result.persistence_excluded_count == 1


def test_oct_without_history_is_observability_excluded() -> None:
    from scripts.ops.future_outcome_label import label_future_multi_institution_onset

    result = label_future_multi_institution_onset(
        ["P1"],
        _hist([]),
        _hist([{"patient_id": "P1", "institution_id": "A"}]),
        threshold=3,
    )

    assert result.labels == {}
    assert result.oct_history_zero_excluded == 1
    assert result.n_evaluable == 0


def test_null_blank_and_duplicate_institutions_are_counted_like_multi_institution_label() -> None:
    from scripts.ops.future_outcome_label import label_future_multi_institution_onset

    oct_histories = _hist([
        {"patient_id": "P1", "institution_id": None},
        {"patient_id": "P1", "institution_id": ""},
        {"patient_id": "P1", "institution_id": " "},
        {"patient_id": "P1", "institution_id": "A"},
        {"patient_id": "P1", "institution_id": "A"},
    ])
    nov_histories = _hist([
        {"patient_id": "P1", "institution_id": "A"},
        {"patient_id": "P1", "institution_id": "B"},
    ])

    result = label_future_multi_institution_onset(
        ["P1"],
        oct_histories,
        nov_histories,
        threshold=2,
    )

    assert result.oct_institution_counts == {"P1": 1}
    assert result.nov_institution_counts == {"P1": 2}
    assert result.labels == {"P1": 1}


def test_threshold_parameter_changes_boundary_behavior() -> None:
    from scripts.ops.future_outcome_label import label_future_multi_institution_onset

    oct_histories = _hist([
        {"patient_id": "P1", "institution_id": "A"},
        {"patient_id": "P1", "institution_id": "B"},
    ])
    nov_histories = _hist([
        {"patient_id": "P1", "institution_id": "A"},
        {"patient_id": "P1", "institution_id": "B"},
        {"patient_id": "P1", "institution_id": "C"},
    ])

    threshold_3 = label_future_multi_institution_onset(["P1"], oct_histories, nov_histories, threshold=3)
    threshold_2 = label_future_multi_institution_onset(["P1"], oct_histories, nov_histories, threshold=2)

    assert threshold_3.labels == {"P1": 1}
    assert threshold_2.labels == {}
    assert threshold_2.persistence_cohort_size == 1


def test_funnel_counts_are_conserved() -> None:
    from scripts.ops.future_outcome_label import label_future_multi_institution_onset

    oct_histories = _hist([
        {"patient_id": "P_pos", "institution_id": "A"},
        {"patient_id": "P_neg", "institution_id": "A"},
        {"patient_id": "P_cens", "institution_id": "A"},
        {"patient_id": "P_persist", "institution_id": "A"},
        {"patient_id": "P_persist", "institution_id": "B"},
    ])
    nov_histories = _hist([
        {"patient_id": "P_pos", "institution_id": "A"},
        {"patient_id": "P_pos", "institution_id": "B"},
        {"patient_id": "P_neg", "institution_id": "A"},
        {"patient_id": "P_persist", "institution_id": "A"},
    ])

    result = label_future_multi_institution_onset(
        ["P_pos", "P_neg", "P_cens", "P_persist", "P_no_oct"],
        oct_histories,
        nov_histories,
        threshold=2,
    )

    assert result.n_patients == 5
    assert (
        result.n_evaluable
        + result.n_censored
        + result.persistence_excluded_count
        + result.oct_history_zero_excluded
    ) == result.n_patients


class _WindowProvider:
    def __init__(self, oct_histories: pd.DataFrame, nov_histories: pd.DataFrame) -> None:
        self.oct_histories = oct_histories
        self.nov_histories = nov_histories

    def get_history_batch(self, patient_ids, reference_date: date, lookback_days: int = 29):
        del lookback_days
        source = self.oct_histories if reference_date == date(2024, 10, 31) else self.nov_histories
        return source[source["patient_id"].isin(patient_ids)].copy()


def test_run_audit_report_has_required_fields() -> None:
    from scripts.ops.future_outcome_label import run_future_outcome_label_audit

    provider = _WindowProvider(
        _hist([
            {"patient_id": "P1", "institution_id": "A"},
            {"patient_id": "P2", "institution_id": "A"},
        ]),
        _hist([
            {"patient_id": "P1", "institution_id": "A"},
            {"patient_id": "P1", "institution_id": "B"},
            {"patient_id": "P2", "institution_id": "A"},
        ]),
    )

    report = run_future_outcome_label_audit(
        provider,
        ["P1", "P2", "P3"],
        feature_reference_date=date(2024, 10, 31),
        outcome_reference_date=date(2024, 11, 30),
        lookback_days=29,
        threshold=2,
    )

    assert report["label_source"] == "future_multi_institution_onset"
    assert report["threshold"] == 2
    assert report["n_feature_cohort"] == 3
    assert report["label_positive"] == 1
    assert report["n_censored"] == 0
    assert report["oct_history_zero_excluded"] == 1
    assert report["label_positive_rate_pct"] == 50.0
    assert report["label_semantics"] == "positive when oct_institution_count < T and nov_institution_count >= T"
    assert "escalation" in report["onset_type_note"]
    assert "no_third_month_caveat" not in report
    assert report["temporal_holdout_status"] == (
        "dataset finalized at 6 months (2024-07..12); Jan 2025 / Gate 5A acquisition cancelled; "
        "Nov→Dec future-onset holdout remains frozen (parked, no unlock planned)"
    )
