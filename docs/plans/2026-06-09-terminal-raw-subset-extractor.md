# Terminal Raw Subset Extractor Implementation Plan

> **For Hermes:** Use subagent-driven-development style: small TDD tasks, spec review, quality review.

**Goal:** Build a terminal CLI that extracts only condition-matching MODE_11_hana daily Raw Parquet rows plus analysis-support files.

**Architecture:** Add a thin ops CLI backed by pure functions. Input is a JSON/YAML condition file or CLI JSON conditions. DuckDB scans selected `records_YYYYMMDD.parquet` files and writes subset Parquet plus matched `eligibility_demographics.parquet`, manifest, and optional analysis files. No HANA connection, no model tuning.

**Tech Stack:** Python 3.12, DuckDB, pandas/pyarrow, pytest.

---

## Constraints

- Authoritative repo: `/mnt/c/model/MODE_11_hana`.
- Raw data fixed: 2024-07-01 through 2024-12-31 only.
- 2025-01+ requests fail closed.
- Future-onset track remains `RESEARCH_TRACK_FROZEN`; tuning/ablation purpose touching 2024-12 fails closed.
- Do not log raw patient IDs or row previews.
- Output only under explicit `--out-dir`; never modify input Raw.

## Task 1: Condition and path helpers

**Objective:** Parse condition config, normalize Windows/WSL paths, discover selected daily Raw files, enforce date/purpose gates.

**Files:**
- Create: `scripts/ops/extract_raw_subset.py`
- Test: `tests/test_ops/test_extract_raw_subset.py`

**Tests:**
- Windows path `C:\\model\\MODE_11_hana` normalizes to `/mnt/c/model/MODE_11_hana` under WSL.
- Date range outside 2024-07..12 raises policy error.
- Purpose `tuning` touching 2024-12 raises policy error.
- Missing daily file raises input error.

## Task 2: DuckDB subset extraction

**Objective:** Export matching records from selected files using DuckDB relation API and safe validated filter DSL.

**Condition DSL v1:**
- `patient_ids`, `institution_ids`, `wk_compn_cd`, `edi_code`: lists.
- `date_range.start/end` required.
- No free-form SQL.

**Tests:**
- Synthetic 2-day parquet data filtered by patient/drug exports expected rows.
- Empty result fails unless `allow_empty=true`.
- Output date range outside rows zero.

## Task 3: Matched demographics and manifest

**Objective:** Export only matched-patient demographics and write privacy-safe manifest.

**Tests:**
- Demographics output contains only patients in extracted records.
- Manifest has counts, hashes, files, DuckDB version, policy status.
- Manifest/logs do not contain raw patient IDs from condition.

## Task 4: Analysis file copy

**Objective:** Copy allowed analysis files/globs into `analysis/`, preserving safe relative paths.

**Tests:**
- `.json/.md/.txt/.csv/.parquet` allowed.
- Traversal outside allowed roots rejected.
- Relative tree preserved.

## Task 5: CLI and verification

**Objective:** Expose `main(argv)` and command-line behavior.

**Commands:**

```bash
.venv/bin/python -m pytest tests/test_ops/test_extract_raw_subset.py -q
.venv/bin/python -m py_compile scripts/ops/extract_raw_subset.py tests/test_ops/test_extract_raw_subset.py
```

**Example:**

```bash
.venv/bin/python scripts/ops/extract_raw_subset.py \
  --config conditions/example_raw_subset.yaml \
  --raw-dir data/Raw \
  --out-dir out/raw_subset_demo \
  --confirm-sensitive
```
