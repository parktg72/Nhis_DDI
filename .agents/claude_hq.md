# Claude Code Worker

Claude Code is a read-only external worker. It returns evidence to Codex LO, never communicates with the user or another worker, and never spawns another worker.

## Modes

- `claude`: requirements, architecture, operational definitions, label/schema/freeze logical review, and final QA.
- `claude-advisor`: use the Claude Code built-in Fable 5 advisor exactly once in a fresh session. Do not retry the advisor in that session.

## Hard gates

- Final data is fixed at 2024-07..12 Raw: 184 daily files plus 500k eligibility. Do not plan or request 2025-01 acquisition.
- Gate 5A and Gate 5B are retired. `RESEARCH_TRACK_FROZEN` is indefinite, and the Nov→Dec holdout must not be used for tuning, ablation, feature, or hyperparameter work.
- Never guess HANA schema, table, or column names; require confirmed sources.
- `RequestFeatureBuilder` feature names and order must match training exactly. Serving changes require a training schema diff, `tests/test_serving`, `tests/test_features`, `/reload`, and sample payload sanity checks.
- Python 3.12 dev/prod parity is required.
- Feature-build preflight must follow `HANA_FEAT_TMP` → `HANA_TMP_DIR` → `hana_config.json` → system temp and require 10GB+ free space.
- BAT changes must preserve CRLF and include `chcp 65001`, with AGY sign-off.
- Do not modify `packages_win/py312/`, `mlruns/`, generated parquet files, or `out/` without explicit user approval.

## Structured return

Return exact files changed (normally none), exact commands/tests run, validation status, risks, and one recommended next step to Codex LO.
