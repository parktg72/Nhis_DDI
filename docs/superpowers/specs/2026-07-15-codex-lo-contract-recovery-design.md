# Codex LO Contract Recovery Design

**Date:** 2026-07-15
**Status:** Approved recovery direction; updated with a read-only feature-schema safety BLOCK before implementation planning

## Decision

Resume the interrupted contract-repair work on a new branch rooted at the
current `main` (`a9c1ba6`), while preserving the existing Codex-only LO
topology.  OpenCode LO is explicitly out of scope and must not be restored.

The recovery branch is `fix/codex-lo-contract-recovery` in the isolated
worktree `.worktrees/codex-lo-contract-recovery`.

## Why this is a selective recovery

The earlier repair branch, `fix/prepush-contract-repair`, was rooted at
`def804e` and contains four later commits:

| Commit | Recovery disposition | Reason |
| --- | --- | --- |
| `a5df516` — dependency-profile contracts | Consider for transplant | Independent operational checker and regression tests. |
| `d25fadc` — lenient sunset override | Consider for transplant | Independent operational/serving fail-closed guard and tests. |
| `d8c5491` — feature-output order characterization | Consider for transplant | Synthetic contract characterization test only. |
| `1e24c93` — LO authority routing | **Never transplant** | It restores an OpenCode LO policy that conflicts with current `main` and the explicit user decision. |

Current `main` already contains the later Codex-sole-LO direction (including
commit `2075aa0`).  Replaying the old policy commit would regress that topology.

## Intended recovery topology

```text
current main (a9c1ba6; Codex is sole LO)
        |
        +-- recovery branch
              +-- assess/apply a5df516: dependency-profile contract
              +-- assess/apply d25fadc: lenient-sunset contract
              +-- assess/apply d8c5491: feature-order characterization
              `-- exclude 1e24c93 forever: OpenCode LO routing
```

Each candidate is handled independently.  A clean cherry-pick is not itself
acceptance: its changes must first be reviewed against current `main`, and any
conflict or broadened scope stops that candidate for Codex LO review.  No
automatic conflict resolution, bulk transplant, or authority-policy rewrite is
allowed.

## Bounded implementation surface

Only the following candidate paths may be considered after planning approval:

```text
scripts/ops/check_py312_drift.py
tests/test_ops/test_check_py312_drift.py
scripts/ops/check_lenient_sunset.py
serving/predictor.py
tests/test_ops/test_check_lenient_sunset.py
tests/test_serving/test_health_schema_drift.py
tests/test_serving/test_lenient_sunset.py
tests/test_contracts/test_profile_contracts.py
```

The scope deliberately excludes all `.agents/*` policy/role files from the old
authority commit.  In particular, it excludes `.agents/opencode_hq.md` and any
configuration that could make OpenCode an LO.

## Post-approval schema finding: quarantine `d8c5491`

Read-only review found that `d8c5491` would characterize a physical training
output containing `sex_male`, while the current serving allowlist and
`RequestFeatureBuilder` use `sex_m`.  The candidate does not create that
condition and its parquet boundary is fully mocked, but applying it as an
ordinary green regression test could be misread as train-serving parity
approval.

Therefore `d8c5491` is quarantined from this recovery until a separate,
cross-family train-serving schema decision is made.  The recovery may still
accept or reject the two independent operational candidates; the feature-order
candidate is recorded as a safety BLOCK, not silently transplanted.

## Safety boundaries

- No HANA query, feature-builder, label, training, model, reload, or dataset
  work is part of this recovery.
- `RESEARCH_TRACK_FROZEN` remains in force: no Nov→Dec/future-onset holdout
  work, Gate 5A/5B work, or 2025-01 acquisition.
- Do not touch protected `packages_win/py312/`, `mlruns/`, generated parquet,
  or `out/` paths.
- Do not create or edit `.bat`/`.cmd` files.
- Do not commit, push, publish, or alter global agent installation state.
- Preserve all unrelated modifications in the primary worktree.

## Runtime and baseline evidence

Before this design document was added, the recovery branch was clean and ran
with Python 3.12.  The orchestration test suite is Bash/WSL-bound by its
repository configuration:

```text
.venv_wsl/bin/python -m pytest tests/test_agents/test_codex_lo_orchestration.py -q
60 passed
```

Running the same suite through Windows `.venv/Scripts/python.exe` produced
`WinError 2` for direct execution of `.agents/adapters/call_external_agent.sh`.
This is an execution-boundary mismatch, not a branch regression: the configured
orchestration command explicitly selects `.venv_wsl/bin/python`.  Future
orchestration verification therefore uses WSL Python 3.12; Windows Python 3.12
remains required where a test or production contract explicitly targets it.

## Verification strategy

For every candidate, the future implementation plan will require:

1. inspect the candidate diff against current `main` before applying it;
2. run the candidate's focused regression tests with the appropriate Python
   3.12 runtime;
3. run the matching serving and operations checks, plus Ruff where available;
4. verify no protected artifact or authority-policy path changed;
5. obtain bounded cross-family evidence sequentially: Claude for logical
   contract/serving review and AGY for runtime, Windows parity, and risk-gate
   review; and
6. have Codex LO reconcile evidence and report results without committing.

There will be at most one outbound external-worker request in flight.  Claude,
AGY, and any advisor are evidence providers only; Codex remains the sole LO.

## Non-goals

- Restoring or configuring OpenCode LO.
- Reopening the previous pre-push plan wholesale.
- Broad dependency, wheelhouse, or constraint changes.
- Feature/model research, data generation, deployment, or release activity.

## Acceptance criteria for the later implementation

- The three technical candidates are either independently accepted with their
  tests or explicitly rejected with evidence.
- `1e24c93` is not applied and no OpenCode LO routing exists in the resulting
  branch.
- Current Codex-only LO tests and relevant focused tests pass with their
  intended Python 3.12 runtime.
- The branch stays free of protected-artifact changes and no commit or push is
  made without a separate user request.
