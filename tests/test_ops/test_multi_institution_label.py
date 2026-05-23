from __future__ import annotations

from datetime import date

import pandas as pd


def _history(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_two_distinct_institutions_returns_label_0() -> None:
    from scripts.ops.multi_institution_label import assign_multi_institution_label

    result = assign_multi_institution_label(
        _history([
            {"institution_id": "H001"},
            {"institution_id": "H002"},
        ]),
        threshold=3,
    )

    assert result == 0


def test_three_distinct_institutions_returns_label_1() -> None:
    from scripts.ops.multi_institution_label import assign_multi_institution_label

    result = assign_multi_institution_label(
        _history([
            {"institution_id": "H001"},
            {"institution_id": "H002"},
            {"institution_id": "H003"},
        ]),
        threshold=3,
    )

    assert result == 1


def test_duplicate_institution_counted_once() -> None:
    from scripts.ops.multi_institution_label import institution_count

    result = institution_count(
        _history([
            {"institution_id": "H001"},
            {"institution_id": "H001"},
            {"institution_id": "H002"},
        ]),
    )

    assert result == 2


def test_null_institution_skipped() -> None:
    from scripts.ops.multi_institution_label import institution_count

    result = institution_count(
        _history([
            {"institution_id": "H001"},
            {"institution_id": None},
            {"institution_id": ""},
            {"institution_id": float("nan")},
        ]),
    )

    assert result == 1


def test_batch_label_distribution_and_percentiles() -> None:
    from scripts.ops.multi_institution_label import label_patient_histories

    histories = pd.DataFrame([
        {"patient_id": "P1", "institution_id": "H001"},
        {"patient_id": "P1", "institution_id": "H002"},
        {"patient_id": "P1", "institution_id": "H003"},
        {"patient_id": "P2", "institution_id": "H001"},
        {"patient_id": "P2", "institution_id": "H001"},
        {"patient_id": "P3", "institution_id": None},
    ])

    result = label_patient_histories(["P1", "P2", "P3", "P4"], histories, threshold=3)

    assert result.labels == {"P1": 1, "P2": 0, "P3": 0, "P4": 0}
    assert result.label_positive == 1
    assert result.institution_counts == {"P1": 3, "P2": 1, "P3": 0, "P4": 0}
    assert result.null_institution_count == 1
    assert result.institution_count_percentiles["p50"] == 0.5
    assert result.institution_count_percentiles["p90"] == 2.4
    assert result.institution_count_percentiles["p95"] == 2.7
    assert result.institution_count_percentiles["p99"] == 2.94
    assert result.institution_count_percentiles["max"] == 3.0


class _FakeProvider:
    def get_history_batch(self, patient_ids, reference_date: date, lookback_days: int = 60):
        del reference_date, lookback_days
        return pd.DataFrame([
            {"patient_id": patient_ids[0], "institution_id": "H001"},
            {"patient_id": patient_ids[0], "institution_id": "H002"},
            {"patient_id": patient_ids[0], "institution_id": "H003"},
            {"patient_id": patient_ids[1], "institution_id": "H001"},
            {"patient_id": patient_ids[1], "institution_id": None},
        ])


def test_run_multi_institution_audit_report_has_required_metrics() -> None:
    from scripts.ops.multi_institution_label import run_multi_institution_audit

    report = run_multi_institution_audit(
        provider=_FakeProvider(),
        patient_ids=["P1", "P2"],
        reference_date=date(2024, 11, 30),
        lookback_days=60,
        threshold=3,
    )

    assert report["n_patients"] == 2
    assert report["label_positive"] == 1
    assert report["label_positive_rate_pct"] == 50.0
    assert report["institution_count_percentiles"]["p50"] == 2.0
    assert report["institution_count_percentiles"]["max"] == 3.0
    assert report["null_institution_count"] == 1
    assert report["null_institution_rate_pct"] == 20.0
    assert "multi-institution" in report["label_semantics"]


def test_threshold_sensitivity_audit_reuses_counts_for_multiple_thresholds() -> None:
    from scripts.ops.multi_institution_label import run_threshold_sensitivity_audit

    report = run_threshold_sensitivity_audit(
        provider=_FakeProvider(),
        patient_ids=["P1", "P2"],
        reference_date=date(2024, 11, 30),
        lookback_days=60,
        thresholds=[2, 3, 4],
        target_positive_rate_range=(10.0, 60.0),
    )

    assert report["thresholds"] == [2, 3, 4]
    assert report["recommended_threshold"] == 3
    assert report["threshold_results"] == [
        {"threshold": 2, "label_positive": 1, "label_positive_rate_pct": 50.0},
        {"threshold": 3, "label_positive": 1, "label_positive_rate_pct": 50.0},
        {"threshold": 4, "label_positive": 0, "label_positive_rate_pct": 0.0},
    ]
    assert report["institution_count_percentiles"]["p50"] == 2.0


def test_provider_window_boundary_preserves_institutions(tmp_path) -> None:
    from scripts.ops.multi_day_parquet_provider import MultiDayParquetHistoryProvider
    from scripts.ops.multi_institution_label import institution_count

    def row(patient_id: str, edi_code: str, start_date: str, institution_id: str) -> dict:
        return {
            "patient_id": patient_id,
            "edi_code": edi_code,
            "start_date": start_date,
            "end_date": start_date,
            "total_days": 1,
            "source": "T30",
            "institution_id": institution_id,
        }

    pd.DataFrame([row("P1", "D1", "2024-10-01", "H001")]).to_parquet(
        tmp_path / "records_20241001.parquet",
        index=False,
    )
    pd.DataFrame([row("P1", "D2", "2024-10-02", "H002")]).to_parquet(
        tmp_path / "records_20241002.parquet",
        index=False,
    )
    pd.DataFrame([row("P1", "D3", "2024-10-03", "H003")]).to_parquet(
        tmp_path / "records_20241003.parquet",
        index=False,
    )

    history = MultiDayParquetHistoryProvider(
        tmp_path,
        extra_columns=["institution_id"],
        deduplicate_keys=False,
    ).get_history(
        "P1",
        reference_date=date(2024, 10, 2),
        lookback_days=1,
    )

    assert history["institution_id"].tolist() == ["H001", "H002"]
    assert institution_count(history) == 2
