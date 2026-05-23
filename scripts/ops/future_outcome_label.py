"""Future multi-institution onset labels from separated feature/outcome windows."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, timedelta
import json
from pathlib import Path
import sys
from typing import Sequence

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.ops.multi_day_parquet_provider import MultiDayParquetHistoryProvider


FUTURE_MULTI_INSTITUTION_THRESHOLD = 6


@dataclass(frozen=True)
class FutureOutcomeLabelResult:
    labels: dict[str, int]
    label_positive: int
    n_patients: int
    n_evaluable: int
    n_censored: int
    onset_eligible_n: int
    persistence_cohort_size: int
    persistence_rate_pct: float
    persistence_excluded_count: int
    oct_history_zero_excluded: int
    clean_onset_positive: int
    escalation_positive: int
    oct_institution_counts: dict[str, int]
    nov_institution_counts: dict[str, int]
    oct_history_rows: dict[str, int]
    nov_history_rows: dict[str, int]
    censored_patient_ids: list[str]
    threshold: int


def label_future_multi_institution_onset(
    patient_ids: Sequence[str],
    oct_histories: pd.DataFrame,
    nov_histories: pd.DataFrame,
    *,
    threshold: int = FUTURE_MULTI_INSTITUTION_THRESHOLD,
) -> FutureOutcomeLabelResult:
    if threshold < 1:
        raise ValueError("threshold must be positive")

    patient_id_list = [str(patient_id) for patient_id in patient_ids]
    oct_df = _normalize_histories(oct_histories)
    nov_df = _normalize_histories(nov_histories)
    oct_counts = _institution_counts(oct_df)
    nov_counts = _institution_counts(nov_df)
    oct_rows = _row_counts(oct_df)
    nov_rows = _row_counts(nov_df)

    labels: dict[str, int] = {}
    censored_patient_ids: list[str] = []
    persistence_denominator = 0
    persistence_positive = 0
    oct_history_zero_excluded = 0
    clean_onset_positive = 0
    escalation_positive = 0
    onset_eligible = 0

    for patient_id in patient_id_list:
        oct_row_count = oct_rows.get(patient_id, 0)
        nov_row_count = nov_rows.get(patient_id, 0)
        oct_count = oct_counts.get(patient_id, 0)
        nov_count = nov_counts.get(patient_id, 0)

        if oct_row_count == 0:
            oct_history_zero_excluded += 1
            continue
        if oct_count >= threshold:
            if nov_row_count > 0:
                persistence_denominator += 1
                if nov_count >= threshold:
                    persistence_positive += 1
            continue

        onset_eligible += 1
        if nov_row_count == 0:
            censored_patient_ids.append(patient_id)
            continue

        label = 1 if nov_count >= threshold else 0
        labels[patient_id] = label
        if label:
            if oct_count == 0:
                clean_onset_positive += 1
            else:
                escalation_positive += 1

    label_positive = sum(labels.values())
    return FutureOutcomeLabelResult(
        labels=labels,
        label_positive=label_positive,
        n_patients=len(patient_id_list),
        n_evaluable=len(labels),
        n_censored=len(censored_patient_ids),
        onset_eligible_n=onset_eligible,
        persistence_cohort_size=persistence_denominator,
        persistence_rate_pct=_pct(persistence_positive, persistence_denominator),
        persistence_excluded_count=sum(
            1 for patient_id in patient_id_list
            if oct_rows.get(patient_id, 0) > 0 and oct_counts.get(patient_id, 0) >= threshold
        ),
        oct_history_zero_excluded=oct_history_zero_excluded,
        clean_onset_positive=clean_onset_positive,
        escalation_positive=escalation_positive,
        oct_institution_counts={patient_id: oct_counts.get(patient_id, 0) for patient_id in patient_id_list},
        nov_institution_counts={patient_id: nov_counts.get(patient_id, 0) for patient_id in patient_id_list},
        oct_history_rows={patient_id: oct_rows.get(patient_id, 0) for patient_id in patient_id_list},
        nov_history_rows={patient_id: nov_rows.get(patient_id, 0) for patient_id in patient_id_list},
        censored_patient_ids=censored_patient_ids,
        threshold=threshold,
    )


def run_future_outcome_label_audit(
    provider,
    patient_ids: Sequence[str],
    *,
    feature_reference_date: date,
    outcome_reference_date: date,
    lookback_days: int = 29,
    threshold: int = FUTURE_MULTI_INSTITUTION_THRESHOLD,
) -> dict:
    patient_id_list = [str(patient_id) for patient_id in patient_ids]
    oct_histories = provider.get_history_batch(
        patient_id_list,
        reference_date=feature_reference_date,
        lookback_days=lookback_days,
    )
    nov_histories = provider.get_history_batch(
        patient_id_list,
        reference_date=outcome_reference_date,
        lookback_days=lookback_days,
    )
    result = label_future_multi_institution_onset(
        patient_id_list,
        oct_histories,
        nov_histories,
        threshold=threshold,
    )
    return _audit_report(
        result,
        feature_reference_date=feature_reference_date,
        outcome_reference_date=outcome_reference_date,
        lookback_days=lookback_days,
        oct_histories=oct_histories,
        nov_histories=nov_histories,
    )


def run_raw_future_outcome_label_audit(
    raw_dir: str | Path,
    *,
    n_patients: int | None = None,
    feature_reference_date: date = date(2024, 10, 31),
    outcome_reference_date: date = date(2024, 11, 30),
    lookback_days: int = 29,
    threshold: int = FUTURE_MULTI_INSTITUTION_THRESHOLD,
) -> dict:
    raw_path = Path(raw_dir)
    patient_ids = _sample_patient_ids(raw_path, feature_reference_date, n_patients)
    provider = MultiDayParquetHistoryProvider(
        raw_path,
        extra_columns=["institution_id"],
        deduplicate_keys=False,
    )
    return run_future_outcome_label_audit(
        provider,
        patient_ids,
        feature_reference_date=feature_reference_date,
        outcome_reference_date=outcome_reference_date,
        lookback_days=lookback_days,
        threshold=threshold,
    )


def write_report(report: dict, output_dir: str | Path) -> tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    json_path = output_path / "future_outcome_label_audit.json"
    md_path = output_path / "future_outcome_label_audit.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_markdown_report(report), encoding="utf-8")
    return json_path, md_path


def _audit_report(
    result: FutureOutcomeLabelResult,
    *,
    feature_reference_date: date,
    outcome_reference_date: date,
    lookback_days: int,
    oct_histories: pd.DataFrame,
    nov_histories: pd.DataFrame,
) -> dict:
    return {
        "label_source": "future_multi_institution_onset",
        "feature_reference_date": feature_reference_date.isoformat(),
        "outcome_reference_date": outcome_reference_date.isoformat(),
        "lookback_days": lookback_days,
        "feature_window": _window_dict(feature_reference_date, lookback_days),
        "outcome_window": _window_dict(outcome_reference_date, lookback_days),
        "threshold": result.threshold,
        "n_feature_cohort": result.n_patients,
        "n_evaluable": result.n_evaluable,
        "n_censored": result.n_censored,
        "censoring_rate_pct": _pct(result.n_censored, result.n_patients),
        "onset_eligible_n": result.onset_eligible_n,
        "onset_eligible_censoring_rate_pct": _pct(result.n_censored, result.onset_eligible_n),
        "label_positive": result.label_positive,
        "label_positive_rate_pct": _pct(result.label_positive, result.n_evaluable),
        "clean_onset_positive": result.clean_onset_positive,
        "escalation_positive": result.escalation_positive,
        "persistence_cohort_size": result.persistence_cohort_size,
        "persistence_rate_pct": result.persistence_rate_pct,
        "persistence_excluded_count": result.persistence_excluded_count,
        "oct_history_zero_excluded": result.oct_history_zero_excluded,
        "oct_history_rows_total": int(len(oct_histories)),
        "nov_history_rows_total": int(len(nov_histories)),
        "oct_institution_count_percentiles": _percentiles(result.oct_institution_counts.values()),
        "nov_institution_count_percentiles": _percentiles(result.nov_institution_counts.values()),
        "label_semantics": "positive when oct_institution_count < T and nov_institution_count >= T",
        "onset_type_note": (
            "Positive cases are escalation when oct_institution_count is 1..T-1 and nov_institution_count >= T. "
            "Clean de-novo onset with oct_institution_count == 0 may be structurally rare or absent under strict "
            "oct_history_rows >= 1 observability."
        ),
        "censoring_policy": "patients with zero Nov outcome-window rows are excluded, not treated as negative",
        "no_third_month_caveat": "2024-12 Raw is unavailable; this audit supports single-period feasibility only",
    }


def _normalize_histories(histories: pd.DataFrame) -> pd.DataFrame:
    if histories.empty:
        return pd.DataFrame(columns=["patient_id", "institution_id"])
    normalized = histories.copy()
    normalized["patient_id"] = normalized["patient_id"].astype(str)
    if "institution_id" not in normalized.columns:
        normalized["institution_id"] = None
    return normalized


def _institution_counts(histories: pd.DataFrame) -> dict[str, int]:
    if histories.empty:
        return {}
    df = histories[["patient_id", "institution_id"]].copy()
    df["patient_id"] = df["patient_id"].astype(str)
    df = df.dropna(subset=["institution_id"])
    df["institution_id"] = df["institution_id"].astype(str).str.strip()
    df = df[df["institution_id"] != ""]
    if df.empty:
        return {}
    return df.groupby("patient_id")["institution_id"].nunique().to_dict()


def _row_counts(histories: pd.DataFrame) -> dict[str, int]:
    if histories.empty:
        return {}
    return histories["patient_id"].astype(str).value_counts().to_dict()


def _window_dict(reference_date: date, lookback_days: int) -> dict[str, str]:
    start = reference_date - timedelta(days=lookback_days)
    return {"start": start.isoformat(), "end": reference_date.isoformat()}


def _sample_patient_ids(raw_dir: Path, reference_date: date, n_patients: int | None) -> list[str]:
    path = raw_dir / f"records_{reference_date.strftime('%Y%m%d')}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"reference records file not found: {path}")
    df = pd.read_parquet(path, columns=["patient_id"])
    patient_ids = df["patient_id"].dropna().astype(str).drop_duplicates().tolist()
    if n_patients is not None:
        return patient_ids[:n_patients]
    return patient_ids


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y%m%d").date()


def _percentiles(values: Sequence[int]) -> dict[str, float]:
    value_list = list(values)
    if not value_list:
        return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    series = pd.Series(value_list, dtype="float64")
    return {
        "p50": round(float(series.quantile(0.50)), 4),
        "p90": round(float(series.quantile(0.90)), 4),
        "p95": round(float(series.quantile(0.95)), 4),
        "p99": round(float(series.quantile(0.99)), 4),
        "max": round(float(series.max()), 4),
    }


def _pct(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round((numerator / denominator) * 100, 4)


def _markdown_report(report: dict) -> str:
    return "\n".join([
        "# Future Outcome Label Audit",
        "",
        f"- label_source: {report['label_source']}",
        f"- feature_window: {report['feature_window']['start']}..{report['feature_window']['end']}",
        f"- outcome_window: {report['outcome_window']['start']}..{report['outcome_window']['end']}",
        f"- threshold: {report['threshold']}",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| n_feature_cohort | {report['n_feature_cohort']} |",
        f"| n_evaluable | {report['n_evaluable']} |",
        f"| n_censored | {report['n_censored']} |",
        f"| censoring_rate_pct | {report['censoring_rate_pct']} |",
        f"| label_positive | {report['label_positive']} |",
        f"| label_positive_rate_pct | {report['label_positive_rate_pct']} |",
        f"| clean_onset_positive | {report['clean_onset_positive']} |",
        f"| escalation_positive | {report['escalation_positive']} |",
        f"| persistence_cohort_size | {report['persistence_cohort_size']} |",
        f"| persistence_rate_pct | {report['persistence_rate_pct']} |",
        "",
    ])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit future multi-institution onset labels.")
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--output-dir", default="data/reports/future_outcome")
    parser.add_argument("--n-patients", type=int, default=None)
    parser.add_argument("--feature-reference-date", default="20241031")
    parser.add_argument("--outcome-reference-date", default="20241130")
    parser.add_argument("--lookback-days", type=int, default=29)
    parser.add_argument("--threshold", type=int, default=FUTURE_MULTI_INSTITUTION_THRESHOLD)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_raw_future_outcome_label_audit(
        args.raw_dir,
        n_patients=args.n_patients,
        feature_reference_date=_parse_date(args.feature_reference_date),
        outcome_reference_date=_parse_date(args.outcome_reference_date),
        lookback_days=args.lookback_days,
        threshold=args.threshold,
    )
    json_path, md_path = write_report(report, args.output_dir)
    print(f"[OK] wrote {json_path}")
    print(f"[OK] wrote {md_path}")
    print(f"label_positive_rate_pct={report['label_positive_rate_pct']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
