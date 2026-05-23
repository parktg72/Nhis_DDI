"""Audit therapeutic duplication labels from efmdc_clsf_no classes."""
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


THERAPEUTIC_DUP_THRESHOLD = 1


@dataclass(frozen=True)
class TherapeuticDupLabelResult:
    labels: dict[str, int]
    label_positive: int
    duplication_class_counts: dict[str, int]
    null_efmdc_row_count: int
    evaluable_patient_count: int
    threshold: int


def duplication_class_count(
    history_df: pd.DataFrame,
    *,
    class_col: str = "efmdc_clsf_no",
    drug_col: str = "drug_code",
) -> int:
    if history_df.empty or class_col not in history_df.columns or drug_col not in history_df.columns:
        return 0
    counts = _distinct_drug_counts_by_class(history_df, class_col=class_col, drug_col=drug_col)
    return int((counts >= 2).sum())


def assign_therapeutic_duplication_label(
    history_df: pd.DataFrame,
    *,
    min_duplicate_classes: int = THERAPEUTIC_DUP_THRESHOLD,
) -> int:
    return 1 if duplication_class_count(history_df) >= min_duplicate_classes else 0


def label_therapeutic_duplication(
    patient_ids: Sequence[str],
    histories: pd.DataFrame,
    *,
    min_duplicate_classes: int = THERAPEUTIC_DUP_THRESHOLD,
) -> TherapeuticDupLabelResult:
    patient_id_list = [str(patient_id) for patient_id in patient_ids]
    if histories.empty:
        histories = pd.DataFrame(columns=["patient_id", "efmdc_clsf_no", "drug_code"])
    else:
        histories = histories.copy()
        histories["patient_id"] = histories["patient_id"].astype(str)
    grouped = {patient_id: group for patient_id, group in histories.groupby("patient_id")}

    labels: dict[str, int] = {}
    duplication_counts: dict[str, int] = {}
    evaluable = 0
    for patient_id in patient_id_list:
        patient_history = grouped.get(patient_id, pd.DataFrame(columns=histories.columns))
        count = duplication_class_count(patient_history)
        duplication_counts[patient_id] = count
        labels[patient_id] = 1 if count >= min_duplicate_classes else 0
        if _has_valid_efmdc(patient_history):
            evaluable += 1

    return TherapeuticDupLabelResult(
        labels=labels,
        label_positive=sum(labels.values()),
        duplication_class_counts=duplication_counts,
        null_efmdc_row_count=_null_efmdc_row_count(histories),
        evaluable_patient_count=evaluable,
        threshold=min_duplicate_classes,
    )


def run_therapeutic_duplication_audit(
    provider,
    patient_ids: Sequence[str],
    reference_date: date,
    lookback_days: int = 60,
    min_duplicate_classes: int = THERAPEUTIC_DUP_THRESHOLD,
) -> dict:
    patient_id_list = [str(patient_id) for patient_id in patient_ids]
    histories = provider.get_history_batch(
        patient_id_list,
        reference_date=reference_date,
        lookback_days=lookback_days,
    )
    result = label_therapeutic_duplication(
        patient_id_list,
        histories,
        min_duplicate_classes=min_duplicate_classes,
    )
    return _base_report(
        patient_id_list,
        histories,
        result,
        reference_date=reference_date,
        lookback_days=lookback_days,
    )


def run_threshold_sensitivity_audit(
    provider,
    patient_ids: Sequence[str],
    reference_date: date,
    lookback_days: int = 60,
    thresholds: Sequence[int] = (1, 2, 3, 4, 5),
    target_positive_rate_range: tuple[float, float] = (10.0, 25.0),
) -> dict:
    patient_id_list = [str(patient_id) for patient_id in patient_ids]
    threshold_list = sorted({int(threshold) for threshold in thresholds})
    histories = provider.get_history_batch(
        patient_id_list,
        reference_date=reference_date,
        lookback_days=lookback_days,
    )
    result = label_therapeutic_duplication(patient_id_list, histories)
    threshold_results = [
        _threshold_summary(
            list(result.duplication_class_counts.values()),
            threshold,
            len(patient_id_list),
            result.evaluable_patient_count,
        )
        for threshold in threshold_list
    ]
    report = _base_report(
        patient_id_list,
        histories,
        result,
        reference_date=reference_date,
        lookback_days=lookback_days,
    )
    report.update({
        "thresholds": threshold_list,
        "target_positive_rate_range": {
            "min_pct": target_positive_rate_range[0],
            "max_pct": target_positive_rate_range[1],
        },
        "recommended_threshold": _recommended_threshold(
            threshold_results,
            target_positive_rate_range,
        ),
        "threshold_results": threshold_results,
    })
    return report


def run_raw_therapeutic_duplication_audit(
    raw_dir: str | Path,
    *,
    n_patients: int = 1000,
    reference_date: date | None = None,
    lookback_days: int = 60,
    min_duplicate_classes: int = THERAPEUTIC_DUP_THRESHOLD,
) -> dict:
    raw_path = Path(raw_dir)
    resolved_date = reference_date or _latest_records_date(raw_path)
    patient_ids = _sample_patient_ids(raw_path, resolved_date, n_patients)
    provider = MultiDayParquetHistoryProvider(
        raw_path,
        extra_columns=["efmdc_clsf_no"],
        deduplicate_keys=False,
    )
    return run_therapeutic_duplication_audit(
        provider,
        patient_ids,
        reference_date=resolved_date,
        lookback_days=lookback_days,
        min_duplicate_classes=min_duplicate_classes,
    )


def run_raw_threshold_sensitivity_audit(
    raw_dir: str | Path,
    *,
    n_patients: int = 1000,
    reference_date: date | None = None,
    lookback_days: int = 60,
    thresholds: Sequence[int] = (1, 2, 3, 4, 5),
    target_positive_rate_range: tuple[float, float] = (10.0, 25.0),
) -> dict:
    raw_path = Path(raw_dir)
    resolved_date = reference_date or _latest_records_date(raw_path)
    patient_ids = _sample_patient_ids(raw_path, resolved_date, n_patients)
    provider = MultiDayParquetHistoryProvider(
        raw_path,
        extra_columns=["efmdc_clsf_no"],
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


def write_report(report: dict, output_dir: str | Path, *, sensitivity: bool = False) -> tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    stem = (
        "therapeutic_duplication_threshold_sensitivity"
        if sensitivity
        else "therapeutic_duplication_label_report"
    )
    json_path = output_path / f"{stem}.json"
    md_path = output_path / f"{stem}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_markdown_report(report), encoding="utf-8")
    return json_path, md_path


def _base_report(
    patient_ids: Sequence[str],
    histories: pd.DataFrame,
    result: TherapeuticDupLabelResult,
    *,
    reference_date: date,
    lookback_days: int,
) -> dict:
    row_count = len(histories)
    count_values = list(result.duplication_class_counts.values())
    return {
        "reference_date": reference_date.isoformat(),
        "lookback_days": lookback_days,
        "n_patients": len(patient_ids),
        "history_rows": row_count,
        "min_duplicate_classes": result.threshold,
        "label_positive": result.label_positive,
        "label_positive_rate_pct": _pct(result.label_positive, len(patient_ids)),
        "evaluable_patient_count": result.evaluable_patient_count,
        "label_positive_rate_evaluable_pct": _pct(
            result.label_positive,
            result.evaluable_patient_count,
        ),
        "null_efmdc_row_count": result.null_efmdc_row_count,
        "null_efmdc_row_rate_pct": _pct(result.null_efmdc_row_count, row_count),
        "duplication_class_count_percentiles": _percentiles(count_values),
        "patients_with_n_dup_classes_dist": _dup_class_distribution(count_values),
        "max_distinct_drug_per_class_percentiles": _max_distinct_drug_percentiles(histories, patient_ids),
        "top_duplicated_efmdc_classes": _top_duplicated_classes(histories),
        "top_null_drug_codes": _top_null_drug_codes(histories),
        "label_semantics": "therapeutic duplication proxy: same efmdc_clsf_no with distinct drug_code count >= 2; label positive when duplicate class count >= threshold",
    }


def _valid_rows(
    df: pd.DataFrame,
    *,
    class_col: str = "efmdc_clsf_no",
    drug_col: str = "drug_code",
) -> pd.DataFrame:
    if df.empty or class_col not in df.columns or drug_col not in df.columns:
        return pd.DataFrame(columns=list(df.columns) if not df.empty else [class_col, drug_col])
    valid = df.copy()
    valid[class_col] = valid[class_col].where(valid[class_col].notna(), None)
    valid = valid.dropna(subset=[class_col])
    valid[class_col] = valid[class_col].astype(str).str.strip()
    valid[drug_col] = valid[drug_col].astype(str).str.strip()
    valid = valid[(valid[class_col] != "") & (valid[drug_col] != "")]
    return valid


def _distinct_drug_counts_by_class(
    df: pd.DataFrame,
    *,
    class_col: str = "efmdc_clsf_no",
    drug_col: str = "drug_code",
) -> pd.Series:
    valid = _valid_rows(df, class_col=class_col, drug_col=drug_col)
    if valid.empty:
        return pd.Series(dtype="int64")
    return valid.groupby(class_col)[drug_col].nunique()


def _has_valid_efmdc(df: pd.DataFrame) -> bool:
    if df.empty or "efmdc_clsf_no" not in df.columns:
        return False
    values = df["efmdc_clsf_no"].dropna().astype(str).str.strip()
    return bool((values != "").any())


def _null_efmdc_row_count(df: pd.DataFrame) -> int:
    if df.empty or "efmdc_clsf_no" not in df.columns:
        return 0
    return int(df["efmdc_clsf_no"].isna().sum())


def _percentiles(values: Sequence[int | float]) -> dict[str, float]:
    if not values:
        return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    series = pd.Series(values, dtype="float64")
    return {
        "p50": round(float(series.quantile(0.50)), 4),
        "p90": round(float(series.quantile(0.90)), 4),
        "p95": round(float(series.quantile(0.95)), 4),
        "p99": round(float(series.quantile(0.99)), 4),
        "max": round(float(series.max()), 4),
    }


def _dup_class_distribution(values: Sequence[int]) -> dict[str, int]:
    return {
        "0": sum(1 for value in values if value == 0),
        "1": sum(1 for value in values if value == 1),
        "2": sum(1 for value in values if value == 2),
        "3_plus": sum(1 for value in values if value >= 3),
    }


def _top_duplicated_classes(histories: pd.DataFrame, limit: int = 10) -> list[dict]:
    valid = _valid_rows(histories)
    if valid.empty or "patient_id" not in valid.columns:
        return []
    counts = valid.groupby(["patient_id", "efmdc_clsf_no"])["drug_code"].nunique()
    duplicated = counts[counts >= 2].reset_index()
    if duplicated.empty:
        return []
    top = duplicated["efmdc_clsf_no"].value_counts().head(limit)
    return [
        {"efmdc_clsf_no": str(key), "patient_class_count": int(value)}
        for key, value in top.items()
    ]


def _top_null_drug_codes(histories: pd.DataFrame, limit: int = 10) -> list[dict]:
    if histories.empty or "efmdc_clsf_no" not in histories.columns or "drug_code" not in histories.columns:
        return []
    null_rows = histories[histories["efmdc_clsf_no"].isna()].copy()
    if null_rows.empty:
        return []
    top = null_rows["drug_code"].astype(str).str.strip().value_counts().head(limit)
    return [{"drug_code": str(key), "row_count": int(value)} for key, value in top.items()]


def _max_distinct_drug_percentiles(histories: pd.DataFrame, patient_ids: Sequence[str]) -> dict[str, float]:
    patient_id_list = [str(patient_id) for patient_id in patient_ids]
    valid = _valid_rows(histories)
    if valid.empty or "patient_id" not in valid.columns:
        return _percentiles([0 for _ in patient_id_list])
    valid["patient_id"] = valid["patient_id"].astype(str)
    per_class = valid.groupby(["patient_id", "efmdc_clsf_no"])["drug_code"].nunique()
    per_patient = per_class.groupby(level=0).max().to_dict()
    return _percentiles([int(per_patient.get(patient_id, 0)) for patient_id in patient_id_list])


def _threshold_summary(
    counts: Sequence[int],
    threshold: int,
    denominator: int,
    evaluable_denominator: int,
) -> dict:
    positives = sum(1 for count in counts if count >= threshold)
    return {
        "threshold": threshold,
        "label_positive": positives,
        "label_positive_rate_pct": _pct(positives, denominator),
        "label_positive_rate_evaluable_pct": _pct(positives, evaluable_denominator),
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


def _sample_patient_ids(raw_dir: Path, reference_date: date, n_patients: int) -> list[str]:
    path = raw_dir / f"records_{reference_date.strftime('%Y%m%d')}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"reference records file not found: {path}")
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


def _parse_thresholds(value: str) -> list[int]:
    thresholds = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not thresholds:
        raise ValueError("thresholds must contain at least one integer")
    return thresholds


def _pct(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round((numerator / denominator) * 100, 4)


def _markdown_report(report: dict) -> str:
    lines = [
        "# Therapeutic Duplication Label Report",
        "",
        f"- reference_date: {report['reference_date']}",
        f"- lookback_days: {report['lookback_days']}",
        f"- n_patients: {report['n_patients']}",
        f"- history_rows: {report['history_rows']}",
        f"- label_semantics: {report['label_semantics']}",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| label_positive | {report['label_positive']} |",
        f"| label_positive_rate_pct | {report['label_positive_rate_pct']} |",
        f"| evaluable_patient_count | {report['evaluable_patient_count']} |",
        f"| label_positive_rate_evaluable_pct | {report['label_positive_rate_evaluable_pct']} |",
        f"| null_efmdc_row_count | {report['null_efmdc_row_count']} |",
        f"| null_efmdc_row_rate_pct | {report['null_efmdc_row_rate_pct']} |",
    ]
    if "threshold_results" in report:
        lines.extend(["", "## Threshold Results", "", "| threshold | positive | positive_rate_pct | evaluable_rate_pct |", "|---:|---:|---:|---:|"])
        for result in report["threshold_results"]:
            lines.append(
                f"| {result['threshold']} | {result['label_positive']} | {result['label_positive_rate_pct']} | {result['label_positive_rate_evaluable_pct']} |",
            )
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit therapeutic duplication labels.")
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--output-dir", default="data/vocab")
    parser.add_argument("--n-patients", type=int, default=1000)
    parser.add_argument("--reference-date", default=None)
    parser.add_argument("--lookback-days", type=int, default=60)
    parser.add_argument("--min-duplicate-classes", type=int, default=THERAPEUTIC_DUP_THRESHOLD)
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
        json_path, md_path = write_report(report, args.output_dir, sensitivity=True)
        print(f"[OK] wrote {json_path}")
        print(f"[OK] wrote {md_path}")
        print(f"recommended_threshold={report['recommended_threshold']}")
        return 0

    report = run_raw_therapeutic_duplication_audit(
        args.raw_dir,
        n_patients=args.n_patients,
        reference_date=_parse_date(args.reference_date),
        lookback_days=args.lookback_days,
        min_duplicate_classes=args.min_duplicate_classes,
    )
    json_path, md_path = write_report(report, args.output_dir)
    print(f"[OK] wrote {json_path}")
    print(f"[OK] wrote {md_path}")
    print(f"label_positive_rate_pct={report['label_positive_rate_pct']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
