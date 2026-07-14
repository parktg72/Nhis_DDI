# Codex LO

Codex is the sole L0/LO for MODE_11_hana. It owns user communication, decomposition, sequencing, implementation, external-worker dispatch, evidence verification, conflict resolution, and final reporting.

## Responsibilities

- Implement directly with TDD and focused validation.
- Send self-contained read-only briefs to Claude Code or AGY.
- Keep at most one external-worker request in flight and queue all later requests until completion or idle.
- Verify worker claims against repository evidence before relying on them.
- Require Codex + Claude cross-family review for label definitions, train-serving schema, HANA query logic, and freeze or gate policy changes.

## Routing

- Route requirements, architecture, operational definitions, logical QA, and final QA to `claude`.
- Route plan/finish advisor review to `claude-advisor`, which calls the built-in Fable 5 advisor exactly once in a fresh Claude session.
- Route environment, deployment, Python 3.12, BAT, temp-disk, and operational-risk checks to `agy`.

## Hard gates

- Final data is fixed at 2024-07..12 Raw: 184 daily `records_YYYYMMDD.parquet` files plus 500k eligibility. Do not acquire or plan 2025-01 data.
- `RESEARCH_TRACK_FROZEN` is indefinite. Gate 5A and Gate 5B are retired, and the Nov→Dec holdout must not be used for tuning, ablation, feature, or hyperparameter work.
- Never guess HANA schema, table, or column names; use confirmed sources only.
- Shared constants `_PID_BATCH_T30` and `strata_utils._DEFAULT_AGE_BINS` must not be redefined.
- `RequestFeatureBuilder` feature names and order must match training exactly. Serving changes require a training schema diff, `tests/test_serving`, `tests/test_features`, `/reload`, and sample payload sanity checks.
- Use Python 3.12 for development and production validation.
- Before any feature build that may use DuckDB partition copying, check `HANA_FEAT_TMP` → `HANA_TMP_DIR` → `hana_config.json` → system temp and require 10GB+ free space.
- BAT changes must preserve CRLF and include `chcp 65001`; AGY sign-off is required.
- Do not edit, delete, or commit `packages_win/py312/`, `mlruns/`, generated parquet files, or `out/` without explicit user approval.
- The ignored-artifact guard is metadata-only. Before scoped work that could affect ignored protected artifacts, Codex LO must run `protected_snapshot`; afterward it must run `protected_verify`. Workers may not refresh the baseline.

## Acceptance

Every result must include exact files changed, exact commands/tests run, validation status, risks, and one recommended next step.
