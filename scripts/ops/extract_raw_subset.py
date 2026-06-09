from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

ALLOWED_START = dt.date(2024, 7, 1)
ALLOWED_END = dt.date(2024, 12, 31)
BLOCKED_TUNING_PURPOSES = {"tuning", "ablation", "hparam", "hyperparameter"}
ALLOWED_PURPOSES = {"analysis", "locked_eval", "inference", "audit"} | BLOCKED_TUNING_PURPOSES
FILTER_COLUMNS = {
    "patient_ids": "patient_id",
    "institution_ids": "institution_id",
    "wk_compn_cd": "wk_compn_cd",
    "edi_code": "edi_code",
}
REQUIRED_RECORD_COLUMNS = {
    "patient_id",
    "institution_id",
    "bill_no",
    "wk_compn_cd",
    "edi_code",
    "gnl_nm_cd",
    "efmdc_clsf_no",
    "start_date",
    "end_date",
    "total_days",
    "dose_once",
    "dose_freq",
    "sick_code",
    "sex",
    "age_id",
    "institution_type",
    "source",
}
REQUIRED_DEMOGRAPHICS_COLUMNS = {"patient_id", "byear", "age", "sex_type", "addr_cd"}
ALLOWED_ANALYSIS_EXTS = {".json", ".md", ".txt", ".csv", ".parquet"}


class ExtractorError(Exception):
    exit_code = 2


class PolicyError(ExtractorError):
    exit_code = 3


class InputError(ExtractorError):
    exit_code = 4


class QueryError(ExtractorError):
    exit_code = 5


class SensitiveConfirmationError(ExtractorError):
    exit_code = 6


@dataclass(frozen=True)
class DateRange:
    start: dt.date
    end: dt.date


@dataclass(frozen=True)
class ExtractResult:
    manifest_path: Path
    records_path: Path
    row_count: int
    distinct_patient_count: int


def normalize_cli_path(value: str | Path) -> Path:
    text = str(value).strip().strip('"')
    match = re.match(r"^([A-Za-z]):[\\/](.*)$", text)
    if match and os.name != "nt":
        drive = match.group(1).lower()
        rest = match.group(2).replace("\\", "/")
        return Path(f"/mnt/{drive}/{rest}")
    return Path(text).expanduser()


def parse_date(value: str) -> dt.date:
    try:
        return dt.date.fromisoformat(str(value))
    except ValueError as exc:
        raise InputError(f"invalid date: {value!r}") from exc


def parse_date_range(config: dict[str, Any]) -> DateRange:
    raw = config.get("date_range") or {}
    start = parse_date(raw.get("start"))
    end = parse_date(raw.get("end"))
    if start > end:
        raise InputError("date_range.start must be <= date_range.end")
    return DateRange(start, end)


def daterange_days(date_range: DateRange) -> list[dt.date]:
    out: list[dt.date] = []
    cur = date_range.start
    while cur <= date_range.end:
        out.append(cur)
        cur += dt.timedelta(days=1)
    return out


def load_condition(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise InputError(f"cannot read config: {path}") from exc
    if path.suffix.lower() == ".json":
        try:
            loaded = json.loads(text)
        except json.JSONDecodeError as exc:
            raise InputError(f"invalid JSON config: {path}") from exc
        if not isinstance(loaded, dict):
            raise InputError("condition config must be an object")
        return loaded
    if yaml is None:
        raise InputError("YAML config needs PyYAML; use JSON or install yaml")
    try:
        loaded = yaml.safe_load(text)
    except Exception as exc:
        raise InputError(f"invalid YAML config: {path}") from exc
    if not isinstance(loaded, dict):
        raise InputError("condition config must be an object")
    return loaded


def parse_condition_json(text: str) -> dict[str, Any]:
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError as exc:
        raise InputError("invalid --condition-json") from exc
    if not isinstance(loaded, dict):
        raise InputError("--condition-json must be a JSON object")
    return loaded


def config_digest(config: dict[str, Any]) -> str:
    redacted = json.dumps(config, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(redacted.encode("utf-8")).hexdigest()


def enforce_policy(config: dict[str, Any], date_range: DateRange) -> dict[str, Any]:
    purpose = str(config.get("purpose", "analysis"))
    if purpose not in ALLOWED_PURPOSES:
        raise PolicyError(f"unsupported purpose: {purpose}")
    if date_range.start < ALLOWED_START or date_range.end > ALLOWED_END:
        raise PolicyError("date_range outside fixed Raw window 2024-07-01..2024-12-31")
    touches_frozen_holdout = date_range.end >= dt.date(2024, 12, 1)
    if purpose in BLOCKED_TUNING_PURPOSES and touches_frozen_holdout:
        raise PolicyError("tuning/ablation touching 2024-12 frozen holdout is blocked")
    if config.get("policy", {}).get("allow_future_onset"):
        raise PolicyError("future-onset track is frozen; allow_future_onset is not accepted")
    return {
        "allowed_date_window": [ALLOWED_START.isoformat(), ALLOWED_END.isoformat()],
        "requested_date_range": [date_range.start.isoformat(), date_range.end.isoformat()],
        "purpose": purpose,
        "future_onset_frozen": True,
        "jan_2025_acquired": False,
        "frozen_holdout_touched": touches_frozen_holdout,
    }


def discover_raw_parquets(raw_dir: Path, date_range: DateRange) -> list[Path]:
    raw_dir = raw_dir.resolve()
    if not raw_dir.exists():
        raise InputError(f"raw_dir missing: {raw_dir}")
    files: list[Path] = []
    missing: list[str] = []
    for day in daterange_days(date_range):
        path = raw_dir / f"records_{day:%Y%m%d}.parquet"
        if path.exists():
            files.append(path)
        else:
            missing.append(path.name)
    if missing:
        raise InputError("missing Raw daily files: " + ", ".join(missing[:10]))
    return files


def sql_literal(path: str | Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_parquet_list_sql(paths: list[Path]) -> str:
    items = ", ".join(sql_literal(p.as_posix()) for p in paths)
    return f"read_parquet([{items}])"


def validate_schema(con: Any, table_sql: str, required: set[str], label: str) -> None:
    rows = con.execute(f"DESCRIBE SELECT * FROM {table_sql} LIMIT 0").fetchall()
    cols = {str(row[0]) for row in rows}
    missing = sorted(required - cols)
    if missing:
        raise InputError(f"{label} missing required columns: {', '.join(missing)}")


def build_where_clause(filters: dict[str, Any]) -> tuple[str, list[tuple[str, str, list[str]]]]:
    clauses: list[str] = []
    tables: list[tuple[str, str, list[str]]] = []
    unknown = sorted(set(filters) - set(FILTER_COLUMNS))
    if unknown:
        raise InputError("unsupported filter keys: " + ", ".join(unknown))
    for key, column in FILTER_COLUMNS.items():
        values = filters.get(key) or []
        if values:
            if not isinstance(values, list):
                raise InputError(f"filters.{key} must be list")
            table = f"filter_{key}"
            tables.append((table, column, [str(v) for v in values]))
            clauses.append(f"CAST({column} AS VARCHAR) IN (SELECT value FROM {table})")
    return (" AND ".join(clauses) if clauses else "TRUE"), tables


def ensure_sensitive_confirmation(confirm: bool, dry_run: bool) -> None:
    if not dry_run and not confirm:
        raise SensitiveConfirmationError("--confirm-sensitive required for real Raw export")


def ensure_output_dir(out_dir: Path, overwrite: bool) -> None:
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise InputError(f"out_dir exists and is not empty; pass --overwrite: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)


def copy_analysis_files(project_root: Path, out_dir: Path, globs: list[str]) -> list[dict[str, Any]]:
    copied: list[dict[str, Any]] = []
    if not globs:
        return copied
    root = project_root.resolve()
    target_root = out_dir / "analysis"
    for pattern in globs:
        if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
            raise InputError(f"analysis glob must be relative and safe: {pattern}")
        for src in sorted(root.glob(pattern)):
            if not src.is_file():
                continue
            if src.suffix.lower() not in ALLOWED_ANALYSIS_EXTS:
                continue
            resolved = src.resolve()
            if root not in resolved.parents and resolved != root:
                raise InputError(f"analysis file escapes project root: {src}")
            rel = resolved.relative_to(root)
            dst = target_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(resolved, dst)
            copied.append({
                "relative_path": rel.as_posix(),
                "bytes": dst.stat().st_size,
                "sha256": sha256_file(dst),
            })
    return copied


def extract_subset(
    config: dict[str, Any],
    raw_dir: Path,
    out_dir: Path,
    project_root: Path,
    *,
    confirm_sensitive: bool = False,
    dry_run: bool = False,
    overwrite: bool = False,
    memory_limit: str = "512MB",
) -> ExtractResult:
    ensure_sensitive_confirmation(confirm_sensitive, dry_run)
    date_range = parse_date_range(config)
    policy = enforce_policy(config, date_range)
    raw_paths = discover_raw_parquets(raw_dir, date_range)
    ensure_output_dir(out_dir, overwrite)

    try:
        import duckdb
        import pandas as pd
    except Exception as exc:
        raise QueryError("duckdb and pandas are required") from exc

    filters = config.get("filters") or {}
    if not isinstance(filters, dict):
        raise InputError("filters must be an object")
    where_clause, filter_tables = build_where_clause(filters)
    date_clause = (
        f"TRY_CAST(start_date AS DATE) <= DATE {sql_literal(date_range.end.isoformat())} "
        f"AND TRY_CAST(end_date AS DATE) >= DATE {sql_literal(date_range.start.isoformat())}"
    )
    where_clause = f"({date_clause}) AND ({where_clause})"

    raw_sql = read_parquet_list_sql(raw_paths)
    demographics_path = raw_dir / "eligibility_demographics.parquet"
    if not demographics_path.exists():
        raise InputError(f"eligibility_demographics.parquet missing: {demographics_path}")
    demographics_sql = f"read_parquet({sql_literal(demographics_path.as_posix())})"

    records_dir = out_dir / "raw"
    records_dir.mkdir(parents=True, exist_ok=True)
    records_path = records_dir / "records_subset.parquet"
    demographics_out = records_dir / "eligibility_demographics.parquet"
    manifest_path = out_dir / "manifest.json"

    con = duckdb.connect()
    try:
        con.execute(f"SET memory_limit={sql_literal(memory_limit)}")
        validate_schema(con, raw_sql, REQUIRED_RECORD_COLUMNS, "Raw records")
        validate_schema(con, demographics_sql, REQUIRED_DEMOGRAPHICS_COLUMNS, "Demographics")
        for table, _column, values in filter_tables:
            con.register(table, pd.DataFrame({"value": values}))
        query = f"SELECT * FROM {raw_sql} WHERE {where_clause}"
        row_count = int(con.execute(f"SELECT COUNT(*) FROM ({query}) q").fetchone()[0])
        allow_empty = bool(config.get("allow_empty", False))
        if row_count == 0 and not allow_empty:
            raise QueryError("query matched zero rows; pass allow_empty: true to permit")
        distinct_patient_count = int(
            con.execute(f"SELECT COUNT(DISTINCT patient_id) FROM ({query}) q").fetchone()[0]
        )
        if not dry_run:
            con.execute(f"COPY ({query}) TO {sql_literal(records_path.as_posix())} (FORMAT PARQUET, COMPRESSION ZSTD)")
            demo_query = (
                f"SELECT d.* FROM {demographics_sql} d "
                f"INNER JOIN (SELECT DISTINCT patient_id FROM ({query}) q) p USING(patient_id)"
            )
            con.execute(
                f"COPY ({demo_query}) TO {sql_literal(demographics_out.as_posix())} "
                "(FORMAT PARQUET, COMPRESSION ZSTD)"
            )
    finally:
        con.close()

    analysis_globs = [str(x) for x in (config.get("outputs", {}) or {}).get("include_analysis_globs", [])]
    copied_analysis = [] if dry_run else copy_analysis_files(project_root, out_dir, analysis_globs)

    input_files = [
        {"name": p.name, "bytes": p.stat().st_size, "sha256": sha256_file(p)}
        for p in raw_paths
    ]
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "project": config.get("project", "MODE_11_hana"),
        "request_id": config.get("request_id"),
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "dry_run": dry_run,
        "condition_hash": config_digest(config),
        "input_raw_dir": raw_dir.as_posix(),
        "input_file_count": len(raw_paths),
        "input_files": input_files,
        "policy_decision": policy,
        "output": {
            "records_file": None if dry_run else records_path.relative_to(out_dir).as_posix(),
            "row_count": row_count,
            "distinct_patient_count": distinct_patient_count,
            "demographics_file": None if dry_run else demographics_out.relative_to(out_dir).as_posix(),
            "analysis_files": copied_analysis,
        },
        "privacy": {
            "patient_ids_logged": False,
            "row_preview_logged": False,
            "raw_patient_id_values_in_manifest": False,
        },
        "duckdb": {"version": duckdb.__version__, "memory_limit": memory_limit},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    if not dry_run:
        (out_dir / "manifest.sha256").write_text(sha256_file(manifest_path) + "  manifest.json\n", encoding="utf-8")
    return ExtractResult(manifest_path, records_path, row_count, distinct_patient_count)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract condition-matching MODE_11_hana Raw Parquet subset.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", help="JSON/YAML condition file")
    source.add_argument("--condition-json", help="Inline JSON condition object")
    parser.add_argument("--raw-dir", required=True, help="Directory containing records_YYYYMMDD.parquet")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--project-root", default=".", help="Project root for analysis globs")
    parser.add_argument("--confirm-sensitive", action="store_true", help="Required for real Raw export")
    parser.add_argument("--dry-run", action="store_true", help="Count only; no data export")
    parser.add_argument("--overwrite", action="store_true", help="Allow non-empty output dir")
    parser.add_argument("--duckdb-memory-limit", default="512MB")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = parse_condition_json(args.condition_json) if args.condition_json else load_condition(normalize_cli_path(args.config))
        result = extract_subset(
            config,
            normalize_cli_path(args.raw_dir),
            normalize_cli_path(args.out_dir),
            normalize_cli_path(args.project_root),
            confirm_sensitive=args.confirm_sensitive,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
            memory_limit=args.duckdb_memory_limit,
        )
    except ExtractorError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return exc.exit_code
    print(
        json.dumps(
            {
                "manifest": result.manifest_path.as_posix(),
                "records": None if args.dry_run else result.records_path.as_posix(),
                "row_count": result.row_count,
                "distinct_patient_count": result.distinct_patient_count,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
