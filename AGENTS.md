# MODE_11_hana Agent Rules

This repository is a HANA prescription-data ML serving system for inappropriate-prescription risk prediction. Codex is the sole L0/LO orchestrator: it owns user communication, decomposition, sequencing, implementation, verification, conflict resolution, and final reporting. Claude Code and AGY are bounded external subagents that return evidence; Codex LO verifies that evidence before reporting success.

## Current hard gates

- Final dataset is fixed: 2024-07..12 Raw, 184 daily `records_YYYYMMDD.parquet` files plus 500k eligibility. Do not plan or request 2025-01 acquisition.
- Gate 5A and Gate 5B are canceled/retired.
- Future-onset research track is frozen indefinitely (`RESEARCH_TRACK_FROZEN`). Nov→Dec frozen holdout must not be used for model, feature, ablation, or hyperparameter tuning. Only freeze-safe work is allowed.
- HANA schema/table/column names must not be guessed. Use confirmed sources only, especially `CLAUDE.md` and inspected code.
- `RequestFeatureBuilder` feature names and order must match training exactly. Serving changes require training schema diff, `tests/test_serving`, `tests/test_features`, `/reload`, and sample payload sanity checks.
- Python 3.12 dev/prod parity is required. Windows production is closed-network.
- Feature builds that may invoke DuckDB `COPY ... PARTITION_BY` require AGY/Codex LO preflight for `HANA_FEAT_TMP`/`HANA_TMP_DIR`/configured temp disk with 10GB+ free space before Codex or training work starts.
- BAT files must preserve CRLF and include `chcp 65001`.
- Protected paths: `packages_win/py312/`, `mlruns/`, generated parquet files, and `out/` artifacts. Do not edit/delete/commit these without explicit user approval.
- Shared constants such as `_PID_BATCH_T30` and `strata_utils._DEFAULT_AGE_BINS` must not be redefined.

## Trigger severity levels

| Level | Meaning | Required action |
|---|---|---|
| WARN | An anomaly or risk was detected, but the current read-only or reversible step may continue. | Log the finding and include it in the agent result. |
| BLOCK | A prerequisite is missing or an unsafe condition is detected for the current task. | Pause the current step and route the blocker to Codex LO. |
| HARD_STOP | A policy violation or protected-path/frozen-holdout risk could cause irreversible contamination or artifact drift. | Abort the downstream action and require explicit human approval through Codex LO. |

## Automatic intervention triggers

Any agent that detects one of these conditions must intervene before continuing. Reports go to Codex LO; subagents do not make final user-facing decisions.

| Trigger | Condition | Severity | Owner | Action |
|---|---|---|---|---|
| Python 3.12 runtime lock | Active Python/pytest is not Python 3.12, or a 3.11 environment is being used as runtime rather than legacy backup context. | BLOCK | AGY / Codex LO | Stop the step and request `.venv` Python 3.12 activation or recreation. |
| BAT CRLF/chcp gate | Any `.bat` file is added or edited. | BLOCK | AGY | Verify CRLF line endings and `chcp 65001` before the step is complete. Codex must not finalize BAT changes without AGY sign-off. |
| Windows wheelhouse lock | Any write/delete to `packages_win/py312/` or Python 3.12 offline wheel/constraint files. | HARD_STOP | AGY | Abort unless the user explicitly approved this protected-path change. |
| HANA feature temp preflight | ETL/feature build/training task may copy raw parquet through DuckDB temp partitions. | BLOCK | AGY | Check temp destination priority (`HANA_FEAT_TMP` -> `HANA_TMP_DIR` -> `hana_config.json` -> system temp) and 10GB+ free space before dispatch. |
| Protected artifact lock | Any write/delete/commit involving `mlruns/`, generated `.parquet`, or `out/`. | HARD_STOP | Codex LO | Abort and request explicit user approval before touching artifacts. |
| Research freeze lock | Task requests Nov→Dec/future-onset holdout tuning, ablation, feature, or hyperparameter work; or treats Gate 5A/5B/2025-01 as active unlocks. | HARD_STOP | All agents | Abort, cite `RESEARCH_TRACK_FROZEN`, and offer only freeze-safe alternatives. |

Research freeze trigger phrases include `Nov→Dec`, `future_mi_t6`, `octnov`, `holdout tuning`, `Gate 5A`, `Gate 5B`, `Jan 2025 holdout`, `2025-01 unseen`, `hyperparameter search on holdout`, and `ablation on Dec`. These terms may appear in docs as canceled/stale history, but not as active plans.

## Subagent roles

- Claude Code / `claude-bridge`: requirements, architecture, operational definitions, label semantics, leakage/schema/freeze logical review, and final QA. Its `claude-advisor` mode uses the built-in Fable 5 advisor exactly once in a fresh session.
- AGY HQ / `agy-bridge`: environment, DevOps, Windows offline deployment, Python 3.12 parity, BAT/CRLF checks, disk-space and risk gates, and explicitly requested external research. It is not the orchestrator or implementation owner.
- Codex LO implements and validates directly. There is no separate Codex worker in the default topology.

## Communication and handoff

- Codex LO may have only one outbound external-worker request in flight. Queue additional Claude/AGY work until the current worker reports completion or idle.
- All agent results must include exact files changed, commands/tests run, validation status, risks, and the single recommended next step.
- Critical changes (label definitions, train-serving schema, HANA query logic, freeze/gate policy) require cross-family review before merge.
- Do not commit, push, publish, or perform irreversible actions unless the user explicitly asks.
