"""Run multi-hot encoding smoke validation without printing patient IDs."""
from __future__ import annotations

import argparse
from datetime import date, datetime
import json
from pathlib import Path
import sys
from typing import Sequence

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.ops.multi_day_parquet_provider import MultiDayParquetHistoryProvider
from scripts.ops.multihot_encoder import encode_batch


def run_smoke(
    raw_dir: str | Path,
    vocab_path: str | Path,
    *,
    n_patients: int = 1000,
    reference_date: date | None = None,
    lookback_days: int = 60,
) -> dict:
    raw_path = Path(raw_dir)
    vocab = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
    resolved_reference_date = reference_date or _latest_records_date(raw_path)
    patient_ids = _sample_patient_ids(raw_path, resolved_reference_date, n_patients)
    provider = MultiDayParquetHistoryProvider(raw_path)
    matrix, stats = encode_batch(
        provider,
        patient_ids,
        vocab,
        reference_date=resolved_reference_date,
        lookback_days=lookback_days,
    )
    return {
        "reference_date": resolved_reference_date.isoformat(),
        "lookback_days": lookback_days,
        "requested_patients": len(patient_ids),
        "matrix_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "stats": stats,
    }


def write_smoke_report(report: dict, output_dir: str | Path) -> tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    json_path = output_path / "multihot_smoke_report.json"
    md_path = output_path / "multihot_smoke_report.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="",
    )
    md_path.write_text(_markdown_report(report), encoding="utf-8", newline="")
    return json_path, md_path


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
    stats = report["stats"]
    return "\n".join([
        "# Multi-hot Smoke Report",
        "",
        f"- reference_date: {report['reference_date']}",
        f"- lookback_days: {report['lookback_days']}",
        f"- requested_patients: {report['requested_patients']}",
        f"- matrix_shape: {report['matrix_shape']}",
        "",
        "| metric | value |",
        "|---|---:|",
        *[f"| {key} | {value} |" for key, value in stats.items()],
        "",
    ])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run multi-hot dataset smoke validation.")
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--vocab-path", required=True)
    parser.add_argument("--output-dir", default="data/vocab")
    parser.add_argument("--n-patients", type=int, default=1000)
    parser.add_argument("--reference-date", default=None)
    parser.add_argument("--lookback-days", type=int, default=60)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_smoke(
        args.raw_dir,
        args.vocab_path,
        n_patients=args.n_patients,
        reference_date=_parse_date(args.reference_date),
        lookback_days=args.lookback_days,
    )
    json_path, md_path = write_smoke_report(report, args.output_dir)
    print(f"[OK] wrote {json_path}")
    print(f"[OK] wrote {md_path}")
    print(f"matrix_shape={report['matrix_shape']}")
    print(f"unk_flag_rate_pct={report['stats']['unk_flag_rate_pct']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
