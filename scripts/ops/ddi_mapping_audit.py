"""Audit HIRA DUR contraindicated DDI labels from wk_compn_cd histories."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime
import json
from pathlib import Path
import sys
from typing import Iterable, Sequence

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.etl.drug_master import DrugMaster
from scripts.ops.multi_day_parquet_provider import MultiDayParquetHistoryProvider


@dataclass(frozen=True)
class DDIMappingAuditResult:
    labels: dict[str, int]
    label_positive: int
    any_mapped_patients: int
    mapped_patients: int
    zero_mapped_patients: int
    hit_pair_counts: dict[str, int]
    candidate_pair_counts: dict[str, int]
    top_hit_pairs: list[dict]
    top_pair_dominance_pct: float
    d_code_count: int
    db_code_count: int
    mapped_wk_code_count: int
    unique_wk_code_count: int
    unmapped_drug_row_count: int
    patients_with_unmapped_wk_count: int
    overlap_positive_patients: int
    temporal_overlap_available: bool


def canonical_pair(a: str, b: str) -> tuple[str, str]:
    left, right = str(a).strip(), str(b).strip()
    return (left, right) if left <= right else (right, left)


def load_contraindicated_pairs(path: str | Path) -> set[tuple[str, str]]:
    df = pd.read_parquet(path)
    if {"drug_a_code", "drug_b_code"}.issubset(df.columns):
        left_col, right_col = "drug_a_code", "drug_b_code"
    elif {"INGR_CODE", "MIXTURE_INGR_CODE"}.issubset(df.columns):
        left_col, right_col = "INGR_CODE", "MIXTURE_INGR_CODE"
    else:
        raise ValueError(f"unsupported DDI pair schema: {list(df.columns)}")

    pairs: set[tuple[str, str]] = set()
    for left, right in df[[left_col, right_col]].dropna().itertuples(index=False):
        left_id, right_id = str(left).strip(), str(right).strip()
        if left_id and right_id and left_id != right_id:
            pairs.add(canonical_pair(left_id, right_id))
    return pairs


def label_ddi_contraindication(
    patient_ids: Sequence[str],
    histories: pd.DataFrame,
    *,
    master,
    contraindicated_pairs: set[tuple[str, str]],
) -> DDIMappingAuditResult:
    patient_id_list = [str(patient_id) for patient_id in patient_ids]
    normalized = _normalize_histories(histories)
    wk_mapping = _build_wk_mapping(normalized, master)
    pair_universe = {drug_id for pair in contraindicated_pairs for drug_id in pair}

    labels: dict[str, int] = {}
    hit_pair_counts: dict[str, int] = {}
    candidate_pair_counts: dict[str, int] = {}
    top_pair_counter: Counter[tuple[str, str]] = Counter()
    any_mapped_patients = 0
    mapped_patients = 0
    patients_with_unmapped = 0
    overlap_positive_patients = 0
    temporal_overlap_available = _has_temporal_columns(normalized)

    grouped = {patient_id: group for patient_id, group in normalized.groupby("patient_id")}
    for patient_id in patient_id_list:
        patient_history = grouped.get(patient_id, pd.DataFrame(columns=normalized.columns))
        patient_hits = _patient_hit_pairs(patient_history, wk_mapping, contraindicated_pairs)
        d_ids = _patient_d_ids(patient_history, wk_mapping, pair_universe)
        candidate_pair_counts[patient_id] = max((len(d_ids) * (len(d_ids) - 1)) // 2, 0)
        hit_pair_counts[patient_id] = len(patient_hits)
        labels[patient_id] = 1 if patient_hits else 0
        if d_ids:
            mapped_patients += 1
        if _patient_has_any_mapped_id(patient_history, wk_mapping):
            any_mapped_patients += 1
        if _patient_has_unmapped_wk(patient_history, wk_mapping):
            patients_with_unmapped += 1
        if temporal_overlap_available and _patient_has_overlap_hit(
            patient_history,
            wk_mapping,
            contraindicated_pairs,
        ):
            overlap_positive_patients += 1
        for hit_pair in patient_hits:
            top_pair_counter[hit_pair] += 1

    label_positive = sum(labels.values())
    mapped_ids = {mapped_id for ids in wk_mapping.values() for mapped_id in ids}
    unique_wk_codes = _unique_wk_codes(normalized)
    mapped_wk_codes = [wk for wk in unique_wk_codes if wk_mapping.get(wk)]
    unmapped_row_count = _unmapped_drug_row_count(normalized, wk_mapping)

    return DDIMappingAuditResult(
        labels=labels,
        label_positive=label_positive,
        any_mapped_patients=any_mapped_patients,
        mapped_patients=mapped_patients,
        zero_mapped_patients=len(patient_id_list) - mapped_patients,
        hit_pair_counts=hit_pair_counts,
        candidate_pair_counts=candidate_pair_counts,
        top_hit_pairs=_top_hit_pairs(top_pair_counter),
        top_pair_dominance_pct=_top_pair_dominance(top_pair_counter, label_positive),
        d_code_count=sum(1 for mapped_id in mapped_ids if _is_d_code(mapped_id)),
        db_code_count=sum(1 for mapped_id in mapped_ids if mapped_id.startswith("DB")),
        mapped_wk_code_count=len(mapped_wk_codes),
        unique_wk_code_count=len(unique_wk_codes),
        unmapped_drug_row_count=unmapped_row_count,
        patients_with_unmapped_wk_count=patients_with_unmapped,
        overlap_positive_patients=overlap_positive_patients,
        temporal_overlap_available=temporal_overlap_available,
    )


def run_ddi_mapping_audit(
    provider,
    patient_ids: Sequence[str],
    reference_date: date,
    lookback_days: int = 60,
    *,
    master,
    contraindicated_pairs: set[tuple[str, str]],
    excluded_drug_ids: Iterable[str] = (),
) -> dict:
    patient_id_list = [str(patient_id) for patient_id in patient_ids]
    excluded_ids = {str(value).strip() for value in excluded_drug_ids if str(value).strip()}
    active_pairs = _exclude_pairs(contraindicated_pairs, excluded_ids)
    histories = provider.get_history_batch(
        patient_id_list,
        reference_date=reference_date,
        lookback_days=lookback_days,
    )
    result = label_ddi_contraindication(
        patient_id_list,
        histories,
        master=master,
        contraindicated_pairs=active_pairs,
    )
    pair_universe = {drug_id for pair in active_pairs for drug_id in pair}
    d_overlap = _mapped_d_overlap(histories, master, pair_universe)
    row_count = len(histories)
    return {
        "reference_date": reference_date.isoformat(),
        "lookback_days": lookback_days,
        "n_patients": len(patient_id_list),
        "history_rows": row_count,
        "contraindicated_pair_count": len(contraindicated_pairs),
        "excluded_drug_ids": sorted(excluded_ids),
        "excluded_pair_count": len(contraindicated_pairs) - len(active_pairs),
        "active_contraindicated_pair_count": len(active_pairs),
        "label_positive": result.label_positive,
        "label_positive_rate_pct": _pct(result.label_positive, len(patient_id_list)),
        "any_mapped_patients": result.any_mapped_patients,
        "any_mapping_coverage_pct": _pct(result.any_mapped_patients, len(patient_id_list)),
        "mapped_patients": result.mapped_patients,
        "zero_mapped_patients": result.zero_mapped_patients,
        "mapping_coverage_pct": _pct(result.mapped_patients, len(patient_id_list)),
        "unique_wk_code_count": result.unique_wk_code_count,
        "mapped_wk_code_count": result.mapped_wk_code_count,
        "wk_coverage_pct": _pct(result.mapped_wk_code_count, result.unique_wk_code_count),
        "d_code_count": result.d_code_count,
        "db_code_count": result.db_code_count,
        "d_code_overlap_count": len(d_overlap),
        "d_code_overlap_pct": _pct(len(d_overlap), len(pair_universe)),
        "d_code_only_overlap_pct": _pct(len(d_overlap), result.d_code_count),
        "unmapped_drug_row_count": result.unmapped_drug_row_count,
        "unmapped_drug_row_rate_pct": _pct(result.unmapped_drug_row_count, row_count),
        "patients_with_unmapped_wk_count": result.patients_with_unmapped_wk_count,
        "patients_with_unmapped_wk_rate_pct": _pct(
            result.patients_with_unmapped_wk_count,
            len(patient_id_list),
        ),
        "candidate_pair_count_percentiles": _percentiles(result.candidate_pair_counts.values()),
        "hit_pair_count_percentiles": _percentiles(result.hit_pair_counts.values()),
        "top_hit_pairs": result.top_hit_pairs,
        "top_pair_dominance_pct": result.top_pair_dominance_pct,
        "temporal_overlap_available": result.temporal_overlap_available,
        "overlap_positive_patients": result.overlap_positive_patients,
        "overlap_positive_rate_pct": _pct(result.overlap_positive_patients, len(patient_id_list)),
        "label_semantics": "same-window HIRA DUR contraindicated DDI proxy: patient has at least one contraindicated D-code pair mapped from wk_compn_cd within lookback window",
        "limitations": [
            "Not a future clinical outcome label; it is a same-window co-prescription proxy.",
            "DrugMaster can map some ingredients to DrugBank DB IDs, but HIRA DUR contraindicated pairs use D-code IDs.",
            "Temporal overlap is only evaluated when prescription_date and end_date are present.",
        ],
    }


def run_raw_ddi_mapping_audit(
    raw_dir: str | Path,
    *,
    n_patients: int = 1000,
    reference_date: date | None = None,
    lookback_days: int = 60,
    master_path: str | Path = "data/processed/hira_drug_master.parquet",
    ddi_matrix_path: str | Path = "data/processed/ddi_matrix_final.parquet",
    contraindicated_pairs_path: str | Path = "data/dur/dur_ddi_contraindicated_std.parquet",
    excluded_drug_ids: Iterable[str] = (),
) -> dict:
    raw_path = Path(raw_dir)
    resolved_date = reference_date or _latest_records_date(raw_path)
    patient_ids = _sample_patient_ids(raw_path, resolved_date, n_patients)
    provider = MultiDayParquetHistoryProvider(
        raw_path,
        extra_columns=["wk_compn_cd"],
        deduplicate_keys=False,
    )
    master = DrugMaster.load_parquet(master_path, ddi_matrix_path)
    pairs = load_contraindicated_pairs(contraindicated_pairs_path)
    return run_ddi_mapping_audit(
        provider,
        patient_ids,
        reference_date=resolved_date,
        lookback_days=lookback_days,
        master=master,
        contraindicated_pairs=pairs,
        excluded_drug_ids=excluded_drug_ids,
    )


def write_report(report: dict, output_dir: str | Path) -> tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    json_path = output_path / "ddi_mapping_audit_report.json"
    md_path = output_path / "ddi_mapping_audit_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_markdown_report(report), encoding="utf-8")
    return json_path, md_path


def _normalize_histories(histories: pd.DataFrame) -> pd.DataFrame:
    columns = list(histories.columns) if not histories.empty else ["patient_id", "wk_compn_cd"]
    if histories.empty:
        return pd.DataFrame(columns=columns)
    normalized = histories.copy()
    normalized["patient_id"] = normalized["patient_id"].astype(str)
    normalized["wk_compn_cd"] = normalized["wk_compn_cd"].where(normalized["wk_compn_cd"].notna(), "")
    normalized["wk_compn_cd"] = normalized["wk_compn_cd"].astype(str).str.strip()
    return normalized


def _exclude_pairs(
    contraindicated_pairs: set[tuple[str, str]],
    excluded_drug_ids: set[str],
) -> set[tuple[str, str]]:
    if not excluded_drug_ids:
        return contraindicated_pairs
    return {
        pair for pair in contraindicated_pairs
        if pair[0] not in excluded_drug_ids and pair[1] not in excluded_drug_ids
    }


def _build_wk_mapping(histories: pd.DataFrame, master) -> dict[str, tuple[str, ...]]:
    mapping: dict[str, tuple[str, ...]] = {}
    for wk_code in _unique_wk_codes(histories):
        ids = tuple(dict.fromkeys(str(value).strip() for value in master.get_ddi_ids(wk_code) if str(value).strip()))
        mapping[wk_code] = ids
    return mapping


def _unique_wk_codes(histories: pd.DataFrame) -> list[str]:
    if histories.empty or "wk_compn_cd" not in histories.columns:
        return []
    values = histories["wk_compn_cd"].dropna().astype(str).str.strip()
    return sorted({value for value in values if value})


def _patient_d_ids(
    patient_history: pd.DataFrame,
    wk_mapping: dict[str, tuple[str, ...]],
    pair_universe: set[str],
) -> set[str]:
    d_ids: set[str] = set()
    if patient_history.empty or "wk_compn_cd" not in patient_history.columns:
        return d_ids
    for wk_code in patient_history["wk_compn_cd"].dropna().astype(str).str.strip():
        for mapped_id in wk_mapping.get(wk_code, ()):
            if _is_d_code(mapped_id) and mapped_id in pair_universe:
                d_ids.add(mapped_id)
    return d_ids


def _patient_hit_pairs(
    patient_history: pd.DataFrame,
    wk_mapping: dict[str, tuple[str, ...]],
    contraindicated_pairs: set[tuple[str, str]],
) -> set[tuple[str, str]]:
    pair_universe = {drug_id for pair in contraindicated_pairs for drug_id in pair}
    d_ids = sorted(_patient_d_ids(patient_history, wk_mapping, pair_universe))
    hits: set[tuple[str, str]] = set()
    for idx, left in enumerate(d_ids):
        for right in d_ids[idx + 1:]:
            pair = canonical_pair(left, right)
            if pair in contraindicated_pairs:
                hits.add(pair)
    return hits


def _patient_has_overlap_hit(
    patient_history: pd.DataFrame,
    wk_mapping: dict[str, tuple[str, ...]],
    contraindicated_pairs: set[tuple[str, str]],
) -> bool:
    mapped_rows = _mapped_patient_rows(patient_history, wk_mapping)
    for idx, left in enumerate(mapped_rows):
        for right in mapped_rows[idx + 1:]:
            if not _date_ranges_overlap(left["start"], left["end"], right["start"], right["end"]):
                continue
            for left_id in left["d_ids"]:
                for right_id in right["d_ids"]:
                    if left_id != right_id and canonical_pair(left_id, right_id) in contraindicated_pairs:
                        return True
    return False


def _mapped_patient_rows(
    patient_history: pd.DataFrame,
    wk_mapping: dict[str, tuple[str, ...]],
) -> list[dict]:
    rows: list[dict] = []
    if patient_history.empty or not _has_temporal_columns(patient_history):
        return rows
    for row in patient_history.itertuples(index=False):
        row_dict = row._asdict()
        wk_code = str(row_dict.get("wk_compn_cd", "")).strip()
        d_ids = [mapped_id for mapped_id in wk_mapping.get(wk_code, ()) if _is_d_code(mapped_id)]
        if not d_ids:
            continue
        rows.append({
            "start": pd.to_datetime(row_dict["prescription_date"]).date(),
            "end": pd.to_datetime(row_dict["end_date"]).date(),
            "d_ids": d_ids,
        })
    return rows


def _has_temporal_columns(histories: pd.DataFrame) -> bool:
    return {"prescription_date", "end_date"}.issubset(histories.columns)


def _date_ranges_overlap(left_start: date, left_end: date, right_start: date, right_end: date) -> bool:
    return left_start <= right_end and right_start <= left_end


def _patient_has_unmapped_wk(
    patient_history: pd.DataFrame,
    wk_mapping: dict[str, tuple[str, ...]],
) -> bool:
    if patient_history.empty or "wk_compn_cd" not in patient_history.columns:
        return False
    for wk_code in patient_history["wk_compn_cd"].dropna().astype(str).str.strip():
        if wk_code and not wk_mapping.get(wk_code):
            return True
    return False


def _patient_has_any_mapped_id(
    patient_history: pd.DataFrame,
    wk_mapping: dict[str, tuple[str, ...]],
) -> bool:
    if patient_history.empty or "wk_compn_cd" not in patient_history.columns:
        return False
    for wk_code in patient_history["wk_compn_cd"].dropna().astype(str).str.strip():
        if wk_code and wk_mapping.get(wk_code):
            return True
    return False


def _unmapped_drug_row_count(histories: pd.DataFrame, wk_mapping: dict[str, tuple[str, ...]]) -> int:
    if histories.empty or "wk_compn_cd" not in histories.columns:
        return 0
    count = 0
    for wk_code in histories["wk_compn_cd"].dropna().astype(str).str.strip():
        if wk_code and not wk_mapping.get(wk_code):
            count += 1
    return count


def _mapped_d_overlap(histories: pd.DataFrame, master, pair_universe: set[str]) -> set[str]:
    mapped: set[str] = set()
    for wk_code in _unique_wk_codes(histories):
        for mapped_id in master.get_ddi_ids(wk_code):
            mapped_id = str(mapped_id).strip()
            if _is_d_code(mapped_id) and mapped_id in pair_universe:
                mapped.add(mapped_id)
    return mapped


def _is_d_code(mapped_id: str) -> bool:
    return mapped_id.startswith("D") and not mapped_id.startswith("DB")


def _top_hit_pairs(counter: Counter[tuple[str, str]], limit: int = 10) -> list[dict]:
    return [
        {"drug_a_id": left, "drug_b_id": right, "patient_count": int(count)}
        for (left, right), count in counter.most_common(limit)
    ]


def _top_pair_dominance(counter: Counter[tuple[str, str]], positive_patients: int) -> float:
    if positive_patients == 0 or not counter:
        return 0.0
    return _pct(int(counter.most_common(1)[0][1]), positive_patients)


def _percentiles(values: Iterable[int | float]) -> dict[str, float]:
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


def _parse_excluded_drug_ids(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _pct(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round((numerator / denominator) * 100, 4)


def _markdown_report(report: dict) -> str:
    lines = [
        "# DDI Mapping Audit Report",
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
        f"| mapped_patients | {report['mapped_patients']} |",
        f"| mapping_coverage_pct | {report['mapping_coverage_pct']} |",
        f"| wk_coverage_pct | {report['wk_coverage_pct']} |",
        f"| d_code_overlap_pct | {report['d_code_overlap_pct']} |",
        f"| unmapped_drug_row_rate_pct | {report['unmapped_drug_row_rate_pct']} |",
        f"| top_pair_dominance_pct | {report['top_pair_dominance_pct']} |",
        f"| overlap_positive_rate_pct | {report['overlap_positive_rate_pct']} |",
    ]
    if report.get("top_hit_pairs"):
        lines.extend(["", "## Top Hit Pairs", "", "| drug_a_id | drug_b_id | patient_count |", "|---|---|---:|"])
        for pair in report["top_hit_pairs"]:
            lines.append(f"| {pair['drug_a_id']} | {pair['drug_b_id']} | {pair['patient_count']} |")
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit HIRA DUR DDI mapping labels.")
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--output-dir", default="data/vocab")
    parser.add_argument("--n-patients", type=int, default=1000)
    parser.add_argument("--reference-date", default=None)
    parser.add_argument("--lookback-days", type=int, default=60)
    parser.add_argument("--master-path", default="data/processed/hira_drug_master.parquet")
    parser.add_argument("--ddi-matrix-path", default="data/processed/ddi_matrix_final.parquet")
    parser.add_argument("--contraindicated-pairs-path", default="data/dur/dur_ddi_contraindicated_std.parquet")
    parser.add_argument("--exclude-drug-ids", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_raw_ddi_mapping_audit(
        args.raw_dir,
        n_patients=args.n_patients,
        reference_date=_parse_date(args.reference_date),
        lookback_days=args.lookback_days,
        master_path=args.master_path,
        ddi_matrix_path=args.ddi_matrix_path,
        contraindicated_pairs_path=args.contraindicated_pairs_path,
        excluded_drug_ids=_parse_excluded_drug_ids(args.exclude_drug_ids),
    )
    json_path, md_path = write_report(report, args.output_dir)
    print(f"[OK] wrote {json_path}")
    print(f"[OK] wrote {md_path}")
    print(f"label_positive_rate_pct={report['label_positive_rate_pct']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
