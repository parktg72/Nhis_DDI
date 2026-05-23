"""Build aggregate EDI-code vocabulary audit reports from raw parquet records."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime
import json
from pathlib import Path
from typing import Sequence

import pandas as pd


DEFAULT_CUTOFFS = (1, 5, 10, 50, 100, 500, 1000)
REQUIRED_COLUMNS = ("patient_id", "edi_code", "source")


@dataclass(frozen=True)
class AuditMeta:
    date_range: tuple[str | None, str | None]
    total_files: int
    total_rows: int
    unique_patients: int
    unique_edi_codes: int


@dataclass(frozen=True)
class CodeStats:
    row_count: int
    patient_count: int


@dataclass(frozen=True)
class CutoffStats:
    cutoff: int
    vocab_size: int
    row_coverage_pct: float
    patient_coverage_pct: float


@dataclass(frozen=True)
class SourceStats:
    total_rows: int
    unique_edi_codes: int


@dataclass(frozen=True)
class TopCodeStats:
    edi_code: str
    row_count: int
    patient_count: int


@dataclass(frozen=True)
class VocabAuditResult:
    meta: AuditMeta
    cutoff_table: list[CutoffStats]
    source_split: dict[str, SourceStats]
    top20_by_frequency: list[TopCodeStats]
    code_stats: dict[str, CodeStats]

    def to_json_payload(self) -> dict:
        return {
            "meta": asdict(self.meta),
            "cutoff_table": [asdict(row) for row in self.cutoff_table],
            "source_split": {
                source: asdict(stats) for source, stats in self.source_split.items()
            },
            "top20_by_frequency": [asdict(row) for row in self.top20_by_frequency],
        }


def build_vocab_audit(
    raw_dir: str | Path,
    date_from: str | date | None = None,
    date_to: str | date | None = None,
    *,
    cutoffs: Sequence[int] = DEFAULT_CUTOFFS,
) -> VocabAuditResult:
    raw_path = Path(raw_dir)
    files = _records_files(raw_path, date_from=date_from, date_to=date_to)
    edi_rows: Counter[str] = Counter()
    source_rows: Counter[str] = Counter()
    source_codes: dict[str, set[str]] = defaultdict(set)
    patient_ids: set[str] = set()
    patient_code_pairs: set[tuple[str, str]] = set()
    total_rows = 0

    for _, path in files:
        df = pd.read_parquet(path, columns=list(REQUIRED_COLUMNS))
        df = df.dropna(subset=["patient_id", "edi_code"]).copy()
        if df.empty:
            continue

        df["patient_id"] = df["patient_id"].astype(str)
        df["edi_code"] = df["edi_code"].astype(str).str.strip()
        df["source"] = df["source"].astype(str)
        df = df.loc[df["edi_code"] != ""]
        if df.empty:
            continue

        total_rows += len(df)
        edi_rows.update(df["edi_code"].tolist())
        patient_ids.update(df["patient_id"].unique().tolist())
        patient_code_pairs.update(
            zip(df["patient_id"].tolist(), df["edi_code"].tolist(), strict=False)
        )

        source_counts = df["source"].value_counts()
        for source, count in source_counts.items():
            source_rows[str(source)] += int(count)
        for source, codes in df.groupby("source")["edi_code"]:
            source_codes[str(source)].update(codes.unique().tolist())

    code_patient_counts = _patient_counts_by_code(patient_code_pairs)
    code_stats = {
        code: CodeStats(
            row_count=count,
            patient_count=code_patient_counts.get(code, 0),
        )
        for code, count in sorted(edi_rows.items())
    }
    patient_max_freq = _patient_max_frequencies(patient_code_pairs, edi_rows)
    cutoff_table = [
        _cutoff_stats(cutoff, edi_rows, patient_max_freq, total_rows, len(patient_ids))
        for cutoff in cutoffs
    ]
    source_split = {
        source: SourceStats(
            total_rows=source_rows[source],
            unique_edi_codes=len(source_codes[source]),
        )
        for source in sorted(source_rows)
    }
    top20 = [
        TopCodeStats(
            edi_code=code,
            row_count=count,
            patient_count=code_patient_counts.get(code, 0),
        )
        for code, count in edi_rows.most_common(20)
    ]
    date_range = _date_range_for_files(files)
    return VocabAuditResult(
        meta=AuditMeta(
            date_range=date_range,
            total_files=len(files),
            total_rows=total_rows,
            unique_patients=len(patient_ids),
            unique_edi_codes=len(edi_rows),
        ),
        cutoff_table=cutoff_table,
        source_split=source_split,
        top20_by_frequency=top20,
        code_stats=code_stats,
    )


def write_audit_outputs(
    result: VocabAuditResult,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    json_path = output_path / "drug_vocab_audit.json"
    md_path = output_path / "drug_vocab_audit.md"

    json_path.write_text(
        json.dumps(result.to_json_payload(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md_path.write_text(_markdown_report(result), encoding="utf-8")
    return json_path, md_path


def _records_files(
    raw_dir: Path,
    *,
    date_from: str | date | None,
    date_to: str | date | None,
) -> list[tuple[date, Path]]:
    start = _parse_optional_date(date_from)
    end = _parse_optional_date(date_to)
    selected: list[tuple[date, Path]] = []
    for path in sorted(raw_dir.glob("records_*.parquet")):
        parsed = _date_from_records_name(path.name)
        if parsed is None:
            continue
        if start is not None and parsed < start:
            continue
        if end is not None and parsed > end:
            continue
        selected.append((parsed, path))
    return selected


def _parse_optional_date(value: str | date | None) -> date | None:
    if value is None or isinstance(value, date):
        return value
    return datetime.strptime(value, "%Y%m%d").date()


def _date_from_records_name(name: str) -> date | None:
    prefix = "records_"
    suffix = ".parquet"
    if not name.startswith(prefix) or not name.endswith(suffix):
        return None
    raw_date = name[len(prefix) : -len(suffix)]
    try:
        return datetime.strptime(raw_date, "%Y%m%d").date()
    except ValueError:
        return None


def _date_range_for_files(files: list[tuple[date, Path]]) -> tuple[str | None, str | None]:
    if not files:
        return (None, None)
    return (files[0][0].isoformat(), files[-1][0].isoformat())


def _patient_counts_by_code(patient_code_pairs: set[tuple[str, str]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for _, code in patient_code_pairs:
        counts[code] += 1
    return dict(counts)


def _patient_max_frequencies(
    patient_code_pairs: set[tuple[str, str]],
    edi_rows: Counter[str],
) -> dict[str, int]:
    max_freq: dict[str, int] = {}
    for patient_id, code in patient_code_pairs:
        freq = edi_rows[code]
        current = max_freq.get(patient_id, 0)
        if freq > current:
            max_freq[patient_id] = freq
    return max_freq


def _cutoff_stats(
    cutoff: int,
    edi_rows: Counter[str],
    patient_max_freq: dict[str, int],
    total_rows: int,
    total_patients: int,
) -> CutoffStats:
    included_codes = {code for code, count in edi_rows.items() if count >= cutoff}
    included_rows = sum(edi_rows[code] for code in included_codes)
    included_patients = sum(1 for max_freq in patient_max_freq.values() if max_freq >= cutoff)
    return CutoffStats(
        cutoff=int(cutoff),
        vocab_size=len(included_codes),
        row_coverage_pct=_pct(included_rows, total_rows),
        patient_coverage_pct=_pct(included_patients, total_patients),
    )


def _pct(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round((numerator / denominator) * 100, 4)


def _markdown_report(result: VocabAuditResult) -> str:
    lines = [
        "# Drug Vocab Audit",
        "",
        f"- date_range: {result.meta.date_range[0]} ~ {result.meta.date_range[1]}",
        f"- total_files: {result.meta.total_files}",
        f"- total_rows: {result.meta.total_rows}",
        f"- unique_patients: {result.meta.unique_patients}",
        f"- unique_edi_codes: {result.meta.unique_edi_codes}",
        "",
        "| cutoff | vocab_size | row_coverage_pct | patient_coverage_pct |",
        "|---:|---:|---:|---:|",
    ]
    for row in result.cutoff_table:
        lines.append(
            f"| {row.cutoff} | {row.vocab_size} | "
            f"{row.row_coverage_pct:.4f} | {row.patient_coverage_pct:.4f} |"
        )
    lines.extend(["", "## Source Split", ""])
    for source, stats in result.source_split.items():
        lines.append(
            f"- {source}: total_rows={stats.total_rows}, "
            f"unique_edi_codes={stats.unique_edi_codes}"
        )
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build aggregate drug vocab audit reports.")
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--output-dir", default="data/vocab")
    parser.add_argument("--date-from", default=None)
    parser.add_argument("--date-to", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = build_vocab_audit(
        args.raw_dir,
        date_from=args.date_from,
        date_to=args.date_to,
    )
    json_path, md_path = write_audit_outputs(result, args.output_dir)
    print(f"[OK] wrote {json_path}")
    print(f"[OK] wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
