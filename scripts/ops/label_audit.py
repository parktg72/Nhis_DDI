"""Audit simple ADR proxy labels from raw sick_code co-occurrence."""
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

from labeling.adr_labeler import ADR_ICD10_MAP
from scripts.ops.multi_day_parquet_provider import MultiDayParquetHistoryProvider


@dataclass(frozen=True)
class LabelAuditResult:
    labels: dict[str, int]
    label_positive: int
    icd10_type_counts: dict[str, int]
    null_sick_code_count: int


def assign_adr_label_from_sick_code(
    history_df: pd.DataFrame,
    icd10_map: dict[str, list[str]] = ADR_ICD10_MAP,
) -> int:
    return 1 if matched_adr_types(history_df, icd10_map) else 0


def matched_adr_types(
    history_df: pd.DataFrame,
    icd10_map: dict[str, list[str]] = ADR_ICD10_MAP,
) -> set[str]:
    if history_df.empty or "sick_code" not in history_df.columns:
        return set()
    codes = _normalized_sick_codes(history_df["sick_code"])
    matched: set[str] = set()
    for code in codes:
        normalized_code = _normalize_icd10(code)
        for adr_type, patterns in icd10_map.items():
            if any(normalized_code.startswith(_normalize_icd10(pattern)) for pattern in patterns):
                matched.add(adr_type)
    return matched


def label_patient_histories(
    patient_ids: Sequence[str],
    histories: pd.DataFrame,
    icd10_map: dict[str, list[str]] = ADR_ICD10_MAP,
) -> LabelAuditResult:
    patient_id_list = [str(patient_id) for patient_id in patient_ids]
    if histories.empty:
        histories = pd.DataFrame(columns=["patient_id", "sick_code"])
    else:
        histories = histories.copy()
        histories["patient_id"] = histories["patient_id"].astype(str)
    grouped = {patient_id: group for patient_id, group in histories.groupby("patient_id")}

    labels: dict[str, int] = {}
    type_counts = {adr_type: 0 for adr_type in icd10_map}
    for patient_id in patient_id_list:
        patient_history = grouped.get(patient_id, pd.DataFrame(columns=histories.columns))
        types = matched_adr_types(patient_history, icd10_map)
        labels[patient_id] = 1 if types else 0
        for adr_type in types:
            type_counts[adr_type] += 1

    return LabelAuditResult(
        labels=labels,
        label_positive=sum(labels.values()),
        icd10_type_counts={key: value for key, value in type_counts.items() if value},
        null_sick_code_count=_null_sick_code_count(histories),
    )


def run_label_audit(
    provider,
    patient_ids: Sequence[str],
    reference_date: date,
    lookback_days: int = 60,
) -> dict:
    patient_id_list = [str(patient_id) for patient_id in patient_ids]
    histories = provider.get_history_batch(
        patient_id_list,
        reference_date=reference_date,
        lookback_days=lookback_days,
    )
    result = label_patient_histories(patient_id_list, histories)
    row_count = len(histories)
    return {
        "reference_date": reference_date.isoformat(),
        "lookback_days": lookback_days,
        "n_patients": len(patient_id_list),
        "history_rows": row_count,
        "label_positive": result.label_positive,
        "label_positive_rate_pct": _pct(result.label_positive, len(patient_id_list)),
        "null_sick_code_count": result.null_sick_code_count,
        "null_sick_code_rate_pct": _pct(result.null_sick_code_count, row_count),
        "icd10_type_counts": result.icd10_type_counts,
        "label_semantics": "sick_code ADR co-occurrence proxy, not retrospective ADR causality",
    }


def run_raw_label_audit(
    raw_dir: str | Path,
    *,
    n_patients: int = 1000,
    reference_date: date | None = None,
    lookback_days: int = 60,
) -> dict:
    raw_path = Path(raw_dir)
    resolved_reference_date = reference_date or _latest_records_date(raw_path)
    patient_ids = _sample_patient_ids(raw_path, resolved_reference_date, n_patients)
    provider = MultiDayParquetHistoryProvider(
        raw_path,
        extra_columns=["sick_code"],
        deduplicate_keys=False,
    )
    return run_label_audit(
        provider,
        patient_ids,
        reference_date=resolved_reference_date,
        lookback_days=lookback_days,
    )


def write_label_audit_report(report: dict, output_dir: str | Path) -> tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    json_path = output_path / "label_audit_report.json"
    md_path = output_path / "label_audit_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_markdown_report(report), encoding="utf-8")
    return json_path, md_path


def _normalized_sick_codes(values: pd.Series) -> list[str]:
    cleaned = values.dropna().astype(str).str.strip()
    codes: list[str] = []
    for value in cleaned.tolist():
        if not value:
            continue
        for part in value.replace(";", ",").split(","):
            part = part.strip()
            if part:
                codes.append(part)
    return codes


def _normalize_icd10(value: str) -> str:
    return str(value).upper().replace(".", "").strip()


def _null_sick_code_count(df: pd.DataFrame) -> int:
    if df.empty or "sick_code" not in df.columns:
        return 0
    return int(df["sick_code"].isna().sum())


def _pct(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round((numerator / denominator) * 100, 4)


def _sample_patient_ids(raw_dir: Path, reference_date: date, n_patients: int) -> list[str]:
    path = raw_dir / f"records_{reference_date.strftime('%Y%m%d')}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"reference records file not found: {path}")
    df = pd.read_parquet(path, columns=["patient_id"])
    return (
        df["patient_id"].dropna().astype(str).drop_duplicates().head(n_patients).tolist()
    )


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


def _markdown_report(report: dict) -> str:
    lines = [
        "# Label Audit Report",
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
        f"| null_sick_code_count | {report['null_sick_code_count']} |",
        f"| null_sick_code_rate_pct | {report['null_sick_code_rate_pct']} |",
        "",
        "## ADR Type Counts",
        "",
    ]
    for key, value in report["icd10_type_counts"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit sick_code ADR proxy labels.")
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--output-dir", default="data/vocab")
    parser.add_argument("--n-patients", type=int, default=1000)
    parser.add_argument("--reference-date", default=None)
    parser.add_argument("--lookback-days", type=int, default=60)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_raw_label_audit(
        args.raw_dir,
        n_patients=args.n_patients,
        reference_date=_parse_date(args.reference_date),
        lookback_days=args.lookback_days,
    )
    json_path, md_path = write_label_audit_report(report, args.output_dir)
    print(f"[OK] wrote {json_path}")
    print(f"[OK] wrote {md_path}")
    print(f"label_positive_rate_pct={report['label_positive_rate_pct']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
