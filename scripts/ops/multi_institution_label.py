"""Multi-institution (다기관) proxy label for polypharmacy fragmentation risk."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime
import json
from pathlib import Path
import sys
from typing import Sequence

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.ops.multi_day_parquet_provider import MultiDayParquetHistoryProvider


MULTI_INSTITUTION_THRESHOLD = 3  # mirrors clinical_rules.py FRAG trigger


@dataclass(frozen=True)
class MultiInstitutionLabelResult:
    labels: dict[str, int]
    label_positive: int
    institution_counts: dict[str, int]
    institution_count_percentiles: dict[str, float]
    null_institution_count: int
    threshold: int


def institution_count(history_df: pd.DataFrame) -> int:
    if history_df.empty or "institution_id" not in history_df.columns:
        return 0
    values = history_df["institution_id"].dropna().astype(str).str.strip()
    values = values[values != ""]
    return int(values.nunique())


def assign_multi_institution_label(
    history_df: pd.DataFrame,
    threshold: int = MULTI_INSTITUTION_THRESHOLD,
) -> int:
    return 1 if institution_count(history_df) >= threshold else 0


def count_distinct_institutions(histories: pd.DataFrame) -> dict[str, int]:
    if histories.empty or "institution_id" not in histories.columns:
        return {}
    df = histories[["patient_id", "institution_id"]].copy()
    df["patient_id"] = df["patient_id"].astype(str)
    df = df.dropna(subset=["institution_id"])
    df = df[df["institution_id"].astype(str).str.strip() != ""]
    if df.empty:
        return {}
    return (
        df.groupby("patient_id")["institution_id"]
        .nunique()
        .to_dict()
    )


def label_multi_institution(
    patient_ids: Sequence[str],
    histories: pd.DataFrame,
    threshold: int = MULTI_INSTITUTION_THRESHOLD,
) -> MultiInstitutionLabelResult:
    patient_id_list = [str(pid) for pid in patient_ids]
    counts = count_distinct_institutions(histories)
    labels: dict[str, int] = {}
    institution_counts: dict[str, int] = {}
    for pid in patient_id_list:
        n = counts.get(pid, 0)
        institution_counts[pid] = n
        labels[pid] = 1 if n >= threshold else 0
    return MultiInstitutionLabelResult(
        labels=labels,
        label_positive=sum(labels.values()),
        institution_counts=institution_counts,
        institution_count_percentiles=_institution_count_percentiles(
            list(institution_counts.values()),
        ),
        null_institution_count=_null_institution_count(histories),
        threshold=threshold,
    )


def label_patient_histories(
    patient_ids: Sequence[str],
    histories: pd.DataFrame,
    threshold: int = MULTI_INSTITUTION_THRESHOLD,
) -> MultiInstitutionLabelResult:
    return label_multi_institution(patient_ids, histories, threshold=threshold)


def run_multi_institution_audit(
    provider,
    patient_ids: Sequence[str],
    reference_date: date,
    lookback_days: int = 60,
    threshold: int = MULTI_INSTITUTION_THRESHOLD,
) -> dict:
    patient_id_list = [str(pid) for pid in patient_ids]
    histories = provider.get_history_batch(
        patient_id_list,
        reference_date=reference_date,
        lookback_days=lookback_days,
    )
    result = label_multi_institution(patient_id_list, histories, threshold=threshold)
    row_count = len(histories)
    counts_list = list(result.institution_counts.values())
    return {
        "reference_date": reference_date.isoformat(),
        "lookback_days": lookback_days,
        "n_patients": len(patient_id_list),
        "history_rows": row_count,
        "threshold": threshold,
        "label_positive": result.label_positive,
        "label_positive_rate_pct": _pct(result.label_positive, len(patient_id_list)),
        "institution_count_mean": round(sum(counts_list) / len(counts_list), 4) if counts_list else 0.0,
        "institution_count_max": max(counts_list) if counts_list else 0,
        "institution_count_percentiles": result.institution_count_percentiles,
        "null_institution_count": result.null_institution_count,
        "null_institution_rate_pct": _pct(result.null_institution_count, row_count),
        "label_semantics": "multi-institution proxy: distinct institution_id count >= threshold within lookback window",
    }


def run_threshold_sensitivity_audit(
    provider,
    patient_ids: Sequence[str],
    reference_date: date,
    lookback_days: int = 60,
    thresholds: Sequence[int] = (4, 5, 6, 7, 8, 9, 10),
    target_positive_rate_range: tuple[float, float] = (10.0, 25.0),
) -> dict:
    patient_id_list = [str(pid) for pid in patient_ids]
    threshold_list = sorted({int(threshold) for threshold in thresholds})
    histories = provider.get_history_batch(
        patient_id_list,
        reference_date=reference_date,
        lookback_days=lookback_days,
    )
    counts = label_multi_institution(
        patient_id_list,
        histories,
        threshold=threshold_list[0] if threshold_list else MULTI_INSTITUTION_THRESHOLD,
    ).institution_counts
    count_values = list(counts.values())
    threshold_results = [
        _threshold_summary(count_values, threshold, len(patient_id_list))
        for threshold in threshold_list
    ]
    return {
        "reference_date": reference_date.isoformat(),
        "lookback_days": lookback_days,
        "n_patients": len(patient_id_list),
        "history_rows": len(histories),
        "thresholds": threshold_list,
        "target_positive_rate_range": {
            "min_pct": target_positive_rate_range[0],
            "max_pct": target_positive_rate_range[1],
        },
        "recommended_threshold": _recommended_threshold(
            threshold_results,
            target_positive_rate_range,
        ),
        "institution_count_percentiles": _institution_count_percentiles(count_values),
        "threshold_results": threshold_results,
        "null_institution_count": _null_institution_count(histories),
        "null_institution_rate_pct": _pct(_null_institution_count(histories), len(histories)),
        "label_semantics": "multi-institution threshold sensitivity: distinct institution_id count >= threshold within lookback window",
    }


def run_label_audit(
    provider,
    patient_ids: Sequence[str],
    reference_date: date,
    lookback_days: int = 60,
    threshold: int = MULTI_INSTITUTION_THRESHOLD,
) -> dict:
    return run_multi_institution_audit(
        provider,
        patient_ids,
        reference_date=reference_date,
        lookback_days=lookback_days,
        threshold=threshold,
    )


def run_raw_threshold_sensitivity_audit(
    raw_dir: str | Path,
    *,
    n_patients: int = 1000,
    reference_date: date | None = None,
    lookback_days: int = 60,
    thresholds: Sequence[int] = (4, 5, 6, 7, 8, 9, 10),
    target_positive_rate_range: tuple[float, float] = (10.0, 25.0),
) -> dict:
    raw_path = Path(raw_dir)
    resolved_date = reference_date or _latest_records_date(raw_path)
    patient_ids = _sample_patient_ids(raw_path, resolved_date, n_patients)
    provider = MultiDayParquetHistoryProvider(
        raw_path,
        extra_columns=["institution_id"],
        deduplicate_keys=False,
    )
    return run_threshold_sensitivity_audit(
        provider,
        patient_ids,
        reference_date=resolved_date,
        lookback_days=lookback_days,
        thresholds=thresholds,
        target_positive_rate_range=target_positive_rate_range,
    )


def run_raw_multi_institution_audit(
    raw_dir: str | Path,
    *,
    n_patients: int = 1000,
    reference_date: date | None = None,
    lookback_days: int = 60,
    threshold: int = MULTI_INSTITUTION_THRESHOLD,
) -> dict:
    raw_path = Path(raw_dir)
    resolved_date = reference_date or _latest_records_date(raw_path)
    patient_ids = _sample_patient_ids(raw_path, resolved_date, n_patients)
    provider = MultiDayParquetHistoryProvider(
        raw_path,
        extra_columns=["institution_id"],
        deduplicate_keys=False,
    )
    return run_multi_institution_audit(
        provider,
        patient_ids,
        reference_date=resolved_date,
        lookback_days=lookback_days,
        threshold=threshold,
    )


def write_report(report: dict, output_dir: str | Path) -> tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    json_path = output_path / "multi_institution_label_report.json"
    md_path = output_path / "multi_institution_label_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_markdown_report(report), encoding="utf-8")
    return json_path, md_path


def write_threshold_sensitivity_report(report: dict, output_dir: str | Path) -> tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    json_path = output_path / "multi_institution_threshold_sensitivity.json"
    md_path = output_path / "multi_institution_threshold_sensitivity.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_threshold_markdown_report(report), encoding="utf-8")
    return json_path, md_path


def _sample_patient_ids(raw_dir: Path, reference_date: date, n_patients: int) -> list[str]:
    path = raw_dir / f"records_{reference_date.strftime('%Y%m%d')}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"reference records file not found: {path}")
    import pandas as pd
    df = pd.read_parquet(path, columns=["patient_id"])
    return df["patient_id"].dropna().astype(str).drop_duplicates().head(n_patients).tolist()


def _latest_records_date(raw_dir: Path) -> date:
    dates = []
    for path in raw_dir.glob("records_*.parquet"):
        try:
            dates.append(datetime.strptime(path.stem.removeprefix("records_"), "%Y%m%d").date())
        except ValueError:
            continue
    if not dates:
        raise FileNotFoundError(f"no records_YYYYMMDD.parquet files found in {raw_dir}")
    return max(dates)


def _parse_date(value: str | None) -> date | None:
    if value is None:
        return None
    return datetime.strptime(value, "%Y%m%d").date()


def _pct(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round((numerator / denominator) * 100, 4)


def _threshold_summary(counts: list[int], threshold: int, denominator: int) -> dict:
    positives = sum(1 for count in counts if count >= threshold)
    return {
        "threshold": threshold,
        "label_positive": positives,
        "label_positive_rate_pct": _pct(positives, denominator),
    }


def _recommended_threshold(
    threshold_results: list[dict],
    target_positive_rate_range: tuple[float, float],
) -> int | None:
    min_pct, max_pct = target_positive_rate_range
    candidates = [
        result
        for result in threshold_results
        if min_pct <= result["label_positive_rate_pct"] <= max_pct
    ]
    if candidates:
        return int(max(candidates, key=lambda result: result["threshold"])["threshold"])
    return None


def _institution_count_percentiles(counts: list[int]) -> dict[str, float]:
    if not counts:
        return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    series = pd.Series(counts, dtype="float64")
    return {
        "p50": round(float(series.quantile(0.50)), 4),
        "p90": round(float(series.quantile(0.90)), 4),
        "p95": round(float(series.quantile(0.95)), 4),
        "p99": round(float(series.quantile(0.99)), 4),
        "max": round(float(series.max()), 4),
    }


def _null_institution_count(histories: pd.DataFrame) -> int:
    if histories.empty or "institution_id" not in histories.columns:
        return 0
    return int(histories["institution_id"].isna().sum())


def _markdown_report(report: dict) -> str:
    percentiles = report["institution_count_percentiles"]
    return "\n".join([
        "# Multi-Institution Label Report",
        "",
        f"- reference_date: {report['reference_date']}",
        f"- lookback_days: {report['lookback_days']}",
        f"- n_patients: {report['n_patients']}",
        f"- threshold: {report['threshold']}",
        f"- label_semantics: {report['label_semantics']}",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| label_positive | {report['label_positive']} |",
        f"| label_positive_rate_pct | {report['label_positive_rate_pct']} |",
        f"| institution_count_mean | {report['institution_count_mean']} |",
        f"| institution_count_max | {report['institution_count_max']} |",
        f"| institution_count_p50 | {percentiles['p50']} |",
        f"| institution_count_p90 | {percentiles['p90']} |",
        f"| institution_count_p95 | {percentiles['p95']} |",
        f"| institution_count_p99 | {percentiles['p99']} |",
        f"| null_institution_count | {report['null_institution_count']} |",
        f"| null_institution_rate_pct | {report['null_institution_rate_pct']} |",
        "",
    ])


def _threshold_markdown_report(report: dict) -> str:
    lines = [
        "# Multi-Institution Threshold Sensitivity",
        "",
        f"- reference_date: {report['reference_date']}",
        f"- lookback_days: {report['lookback_days']}",
        f"- n_patients: {report['n_patients']}",
        f"- history_rows: {report['history_rows']}",
        f"- recommended_threshold: {report['recommended_threshold']}",
        f"- label_semantics: {report['label_semantics']}",
        "",
        "| threshold | label_positive | label_positive_rate_pct |",
        "|---:|---:|---:|",
    ]
    for result in report["threshold_results"]:
        lines.append(
            f"| {result['threshold']} | {result['label_positive']} | {result['label_positive_rate_pct']} |",
        )
    lines.extend([
        "",
        "## Institution Count Percentiles",
        "",
        "| percentile | value |",
        "|---|---:|",
    ])
    for key, value in report["institution_count_percentiles"].items():
        lines.append(f"| {key} | {value} |")
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit multi-institution proxy labels.")
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--output-dir", default="data/vocab")
    parser.add_argument("--n-patients", type=int, default=1000)
    parser.add_argument("--reference-date", default=None)
    parser.add_argument("--lookback-days", type=int, default=60)
    parser.add_argument("--threshold", type=int, default=MULTI_INSTITUTION_THRESHOLD)
    parser.add_argument("--thresholds", default=None)
    parser.add_argument("--target-positive-rate-min", type=float, default=10.0)
    parser.add_argument("--target-positive-rate-max", type=float, default=25.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.thresholds:
        report = run_raw_threshold_sensitivity_audit(
            args.raw_dir,
            n_patients=args.n_patients,
            reference_date=_parse_date(args.reference_date),
            lookback_days=args.lookback_days,
            thresholds=_parse_thresholds(args.thresholds),
            target_positive_rate_range=(
                args.target_positive_rate_min,
                args.target_positive_rate_max,
            ),
        )
        json_path, md_path = write_threshold_sensitivity_report(report, args.output_dir)
        print(f"[OK] wrote {json_path}")
        print(f"[OK] wrote {md_path}")
        print(f"recommended_threshold={report['recommended_threshold']}")
        return 0
    report = run_raw_multi_institution_audit(
        args.raw_dir,
        n_patients=args.n_patients,
        reference_date=_parse_date(args.reference_date),
        lookback_days=args.lookback_days,
        threshold=args.threshold,
    )
    json_path, md_path = write_report(report, args.output_dir)
    print(f"[OK] wrote {json_path}")
    print(f"[OK] wrote {md_path}")
    print(f"label_positive_rate_pct={report['label_positive_rate_pct']}")
    return 0


def _parse_thresholds(value: str) -> list[int]:
    thresholds = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not thresholds:
        raise ValueError("thresholds must contain at least one integer")
    return thresholds


if __name__ == "__main__":
    raise SystemExit(main())
