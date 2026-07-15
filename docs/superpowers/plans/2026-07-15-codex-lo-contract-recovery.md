# Codex LO Contract Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. The repository assigns implementation and final validation to Codex LO, so do not dispatch a separate Codex worker.

**Goal:** Recover the two independent operational contract fixes from the interrupted branch while retaining Codex as the sole LO and rejecting OpenCode LO restoration and the unsafe feature-order candidate.

**Architecture:** Use the old branch only as an immutable source of two narrow, test-first file pairs/groups. Apply each candidate to the isolated recovery worktree in working-tree form, verify its exact scope and Python 3.12 behavior, then retain it without committing. The feature-order candidate remains quarantined because its `sex_male` training output conflicts with the serving `sex_m` contract and needs a separate schema decision.

**Tech Stack:** Python 3.12 (Windows `.venv` and WSL `.venv_wsl`), pytest, `packaging` PEP 440/508 APIs, pandas test doubles, Ruff, configured Claude Fable advisor, AGY bridge, Git worktree safeguards.

---

## Fixed scope and runtime matrix

- Recovery worktree: `/mnt/c/model/mode_11_hana/.worktrees/codex-lo-contract-recovery`
- Recovery branch: `fix/codex-lo-contract-recovery`, rooted at `main` commit `a9c1ba6`
- Candidate source commits, in permitted order: `a5df516`, then `d25fadc`
- Excluded forever: `1e24c93` and every OpenCode LO policy/configuration path
- Quarantined: `d8c5491`; do not copy its test or modify feature/training/serving code in this recovery
- The repository rule overrides generic commit guidance: do not stage, commit, push, publish, reset, or delete anything.
- The direct Bash adapter suite is run with WSL Python 3.12.3. Its configured command passed: `60 passed`.
- Operational and serving tests use the Windows Python 3.12 virtual environment. The current focused baseline passed: `54 passed`.
- Do not run the historical CUDA profile: `packages_win/requirements_cuda_cu126.txt` is not confirmed to exist.
- No feature build, DuckDB invocation, HANA query, data generation, `.bat`/`.cmd` work, protected artifact work, or frozen-holdout work is authorized.

## File structure and ownership

| Path | Role in this recovery |
| --- | --- |
| `scripts/ops/check_py312_drift.py` | Candidate `a5df516` strict dependency-profile checker. |
| `tests/test_ops/test_check_py312_drift.py` | Candidate `a5df516` PEP 440/508 and lock-gap regression coverage. |
| `scripts/ops/check_lenient_sunset.py` | Candidate `d25fadc` operational sunset diagnostic. |
| `serving/predictor.py` | Candidate `d25fadc` canonical-date fail-closed predicate only. |
| `tests/test_ops/test_check_lenient_sunset.py` | Candidate `d25fadc` diagnostic regression coverage. |
| `tests/test_serving/test_health_schema_drift.py` | Candidate `d25fadc` `/health` degradation regression coverage. |
| `tests/test_serving/test_lenient_sunset.py` | Candidate `d25fadc` serving predicate regression coverage. |
| `docs/superpowers/specs/2026-07-15-codex-lo-contract-recovery-design.md` | Approved design and the newly recorded `d8c5491` safety BLOCK. |
| `docs/superpowers/plans/2026-07-15-codex-lo-contract-recovery.md` | This execution plan. |

### Task 1: Reconfirm the isolated baseline and hard gates

**Files:**

- Modify: none
- Read: the design document, current `AGENTS.md`, candidate commits, and the paths in the preceding table

- [x] **Step 1: Confirm worktree ownership and the exact allowed dirt**

Run from the recovery worktree:

```bash
git status --short --branch
git rev-parse HEAD
git rev-parse main
```

Expected: the branch points to `a9c1ba6`; only the recovery design and this plan may be untracked. If any candidate source/test path, protected path, or primary-worktree user file is dirty, stop and report a BLOCK without cleaning it.

- [x] **Step 2: Prove candidate paths do not overlap current Codex-only-LO work**

```bash
git diff --name-status def804e..main -- \
  scripts/ops/check_py312_drift.py \
  tests/test_ops/test_check_py312_drift.py \
  scripts/ops/check_lenient_sunset.py \
  serving/predictor.py \
  tests/test_ops/test_check_lenient_sunset.py \
  tests/test_serving/test_health_schema_drift.py \
  tests/test_serving/test_lenient_sunset.py
git show --format='' --name-only 1e24c93
```

Expected: the first command has no output; the second lists only authority-policy files. If a candidate overlaps a post-`def804e` main change, do not resolve it automatically.

- [x] **Step 3: Verify both Python 3.12 execution boundaries and `packaging` availability**

```bash
/mnt/c/model/mode_11_hana/.venv_wsl/bin/python -c "import packaging, sys; print(sys.version_info[:2], packaging.__version__); raise SystemExit(sys.version_info[:2] != (3, 12))"
/mnt/c/model/mode_11_hana/.venv/Scripts/python.exe -c "import packaging, sys; print(sys.version_info[:2], packaging.__version__); raise SystemExit(sys.version_info[:2] != (3, 12))"
```

Expected: both commands print `(3, 12)` and exit zero. If Windows Python cannot import `packaging`, stop: do not install packages, edit constraints, or touch the offline wheelhouse.

- [x] **Step 4: Capture an artifact guard baseline and run unchanged baselines**

```bash
PYTHONDONTWRITEBYTECODE=1 /mnt/c/model/mode_11_hana/.venv_wsl/bin/python \
  .agents/adapters/protected_artifact_guard.py snapshot --root . \
  --state /tmp/codex-lo-contract-recovery-protected-baseline.json
/mnt/c/model/mode_11_hana/.venv_wsl/bin/python -B -m pytest -p no:cacheprovider -q \
  tests/test_agents/test_codex_lo_orchestration.py
/mnt/c/model/mode_11_hana/.venv/Scripts/python.exe -m pytest -q \
  tests/test_ops/test_check_py312_drift.py \
  tests/test_ops/test_check_lenient_sunset.py \
  tests/test_serving/test_health_schema_drift.py \
  tests/test_serving/test_lenient_sunset.py \
  tests/test_contracts/test_profile_contracts.py
```

Expected: the guard creates only its `/tmp` state, the WSL adapter suite reports `60 passed`, and the Windows focused baseline reports `54 passed`. A Windows attempt to run the Bash adapter suite is not a valid baseline because it directly executes a `.sh` adapter.

### Task 2: Recover the strict dependency-profile contract from `a5df516`

**Files:**

- Modify: `scripts/ops/check_py312_drift.py`
- Modify: `tests/test_ops/test_check_py312_drift.py`
- Read only: `constraints-py312.txt`, `packages_win/requirements.txt`, `hana_app/requirements.txt`

- [x] **Step 1: Install the candidate regression test before its implementation**

```bash
git restore --source=a5df516 --worktree -- tests/test_ops/test_check_py312_drift.py
/mnt/c/model/mode_11_hana/.venv_wsl/bin/python -B -m pytest -p no:cacheprovider -q \
  tests/test_ops/test_check_py312_drift.py::test_check_drift_no_drift_when_all_declarations_match
```

Expected: RED, because current `check_drift()` accepts only one argument and the recovered test supplies the selected requirement profile. Preserve this failing working diff for the next step; do not reset or discard it.

- [x] **Step 2: Install the exact implementation and check its recovery scope**

```bash
git restore --source=a5df516 --worktree -- scripts/ops/check_py312_drift.py
git diff --name-only -- scripts/ops/check_py312_drift.py tests/test_ops/test_check_py312_drift.py
git diff --check -- scripts/ops/check_py312_drift.py tests/test_ops/test_check_py312_drift.py
git diff --exit-code a5df516 -- scripts/ops/check_py312_drift.py tests/test_ops/test_check_py312_drift.py
```

Expected: only the two named paths differ from `HEAD`, whitespace check is silent, and the final comparison is silent because the recovered content exactly matches `a5df516`. The implementation must use selected repeatable `-r/--requirement` profiles, PEP 508 parsing, canonical names, PEP 440 version checks, marker evaluation, deterministic tuple reports, and exact-lock-gap reporting.

- [x] **Step 3: Run the candidate's focused Python 3.12 regression suite**

```bash
/mnt/c/model/mode_11_hana/.venv_wsl/bin/python -B -m pytest -p no:cacheprovider -q \
  tests/test_ops/test_check_py312_drift.py
/mnt/c/model/mode_11_hana/.venv/Scripts/python.exe -m pytest -q \
  tests/test_ops/test_check_py312_drift.py
```

Expected: `35 passed` in each supported Python 3.12 environment. The tests cover malformed/unreadable/empty configuration, false markers, local versions, invalid metadata, deterministic output, and exact-lock gaps.

- [x] **Step 4: Exercise the real profiles as a diagnostic, not a green gate**

```bash
/mnt/c/model/mode_11_hana/.venv_wsl/bin/python -m scripts.ops.check_py312_drift \
  constraints-py312.txt \
  -r packages_win/requirements.txt \
  -r hana_app/requirements.txt
```

Expected: nonzero status is allowed and must be recorded exactly. Existing WSL evidence is `configuration_errors=0`, `drifts=37`, and `lock_gaps=21`; the 21 direct lock gaps are manifest-derived. Do not edit constraints, requirements, or `packages_win/py312/` to make this command green. Any follow-up lock alignment requires separate explicit user approval and AGY review.

- [x] **Step 5: Run static checks and the protected-artifact verification**

```bash
/mnt/c/Users/ptg/AppData/Local/Programs/Python/Python312/Scripts/ruff.exe check --no-cache \
  scripts/ops/check_py312_drift.py \
  tests/test_ops/test_check_py312_drift.py
PYTHONDONTWRITEBYTECODE=1 /mnt/c/model/mode_11_hana/.venv_wsl/bin/python \
  .agents/adapters/protected_artifact_guard.py verify --root . \
  --state /tmp/codex-lo-contract-recovery-protected-baseline.json
```

Expected: Ruff exits zero; the guard reports no protected-artifact change. If either fails, stop at this candidate and report the exact output.

### Task 3: Recover the canonical lenient-sunset guard from `d25fadc`

**Files:**

- Modify: `scripts/ops/check_lenient_sunset.py`
- Modify: `serving/predictor.py`
- Modify: `tests/test_ops/test_check_lenient_sunset.py`
- Modify: `tests/test_serving/test_health_schema_drift.py`
- Modify: `tests/test_serving/test_lenient_sunset.py`

- [x] **Step 1: Install the candidate tests first and prove the current bypass is RED**

```bash
git restore --source=d25fadc --worktree -- \
  tests/test_ops/test_check_lenient_sunset.py \
  tests/test_serving/test_health_schema_drift.py \
  tests/test_serving/test_lenient_sunset.py
/mnt/c/model/mode_11_hana/.venv/Scripts/python.exe -m pytest -q \
  tests/test_ops/test_check_lenient_sunset.py::test_check_sunset_noncanonical_env_date_blocks_lenient \
  tests/test_serving/test_lenient_sunset.py::TestSunsetHelper::test_noncanonical_env_date_blocks_lenient
```

Expected: RED. Before the candidate source is restored, `2027-1-1` is parsed by `strptime` and can remain lenient-active; the operations report also lacks `warning_reason`.

- [x] **Step 2: Install the two exact source files and assert the five-file boundary**

```bash
git restore --source=d25fadc --worktree -- \
  scripts/ops/check_lenient_sunset.py \
  serving/predictor.py
git diff --name-only -- \
  scripts/ops/check_lenient_sunset.py \
  serving/predictor.py \
  tests/test_ops/test_check_lenient_sunset.py \
  tests/test_serving/test_health_schema_drift.py \
  tests/test_serving/test_lenient_sunset.py
git diff --check -- \
  scripts/ops/check_lenient_sunset.py \
  serving/predictor.py \
  tests/test_ops/test_check_lenient_sunset.py \
  tests/test_serving/test_health_schema_drift.py \
  tests/test_serving/test_lenient_sunset.py
git diff --exit-code d25fadc -- \
  scripts/ops/check_lenient_sunset.py \
  serving/predictor.py \
  tests/test_ops/test_check_lenient_sunset.py \
  tests/test_serving/test_health_schema_drift.py \
  tests/test_serving/test_lenient_sunset.py
```

Expected: exactly the five named paths are restored and match `d25fadc`. In both runtime boundaries the parser must require `parsed.isoformat() == raw`; malformed or noncanonical overrides block lenient mode. `/health` must expose the raw override, report `feature_schema_lenient_allowed=false`, and become degraded when lenient is requested.

- [x] **Step 3: Run focused serving-safety regression coverage**

```bash
/mnt/c/model/mode_11_hana/.venv/Scripts/python.exe -m pytest -q \
  tests/test_ops/test_check_lenient_sunset.py \
  tests/test_serving/test_lenient_sunset.py \
  tests/test_serving/test_health_schema_drift.py
```

Expected: `39 passed`. This validates authoritative-date absence, invalid and noncanonical overrides, exact-date wording, raw `/health` visibility, and fail-closed serving behavior.

- [x] **Step 4: Run the mandatory train-serving, reload, and sample-request gates**

```bash
/mnt/c/model/mode_11_hana/.venv/Scripts/python.exe -m pytest -q \
  tests/test_features \
  tests/test_serving
/mnt/c/model/mode_11_hana/.venv/Scripts/python.exe -m pytest -q \
  tests/test_serving/test_feature_contract.py::test_training_default_feature_cols_allowed_by_serving_schema \
  tests/test_serving/test_feature_contract.py::test_builder_aligns_to_training_default_feature_cols \
  tests/test_contracts/test_serving_characterization.py::test_feature_vector_uses_bundle_name_order \
  tests/test_serving/test_feature_schema_strict.py::test_reload_hierarchical_rejects_unknown_strict \
  tests/test_contracts/test_reload_artifact_compat.py::test_reload_hierarchical_rechecks_nonempty_feature_cols \
  tests/test_serving/test_predictor.py::TestHybridPredictorDLIntegration::test_predict_attaches_dl_prediction_without_changing_final_risk
```

Expected: all collected tests pass. The diff must show no training feature-name/order change; these tests provide the schema-diff, `/reload`, and synthetic sample-request checks required for a serving safety-boundary change. Do not start a live server or load a production model artifact.

- [x] **Step 5: Run static and artifact checks for the five-file candidate**

```bash
/mnt/c/Users/ptg/AppData/Local/Programs/Python/Python312/Scripts/ruff.exe check --no-cache \
  scripts/ops/check_lenient_sunset.py \
  serving/predictor.py \
  tests/test_ops/test_check_lenient_sunset.py \
  tests/test_serving/test_health_schema_drift.py \
  tests/test_serving/test_lenient_sunset.py
PYTHONDONTWRITEBYTECODE=1 /mnt/c/model/mode_11_hana/.venv_wsl/bin/python \
  .agents/adapters/protected_artifact_guard.py verify --root . \
  --state /tmp/codex-lo-contract-recovery-protected-baseline.json
```

Expected: Ruff exits zero and the artifact guard finds no protected-path change. If a check fails, preserve the diff and route the failure to Codex LO rather than broadening the repair.

### Task 4: Quarantine the `d8c5491` physical feature-order characterization

**Files:**

- Modify: none
- Read: `d8c5491`, `scripts/features/feature_engineer.py`, `scripts/features/selector.py`, `scripts/train/dataset.py`, `serving/predictor.py`

- [x] **Step 1: Record the safety evidence without copying the candidate test**

```bash
git show --format='' --name-only d8c5491
rg -n "sex_male|sex_m|META_COLS|feature_cols" \
  scripts/features/feature_engineer.py \
  scripts/features/selector.py \
  scripts/train/dataset.py \
  serving/predictor.py
git diff --name-only -- tests/test_contracts/test_profile_contracts.py
```

Expected: `d8c5491` changes only `tests/test_contracts/test_profile_contracts.py`; the recovery diff must not include that path. The scan demonstrates that the candidate characterizes `sex_male` output while serving uses `sex_m`.

- [x] **Step 2: Treat the candidate as a BLOCK, not a test-only cleanup**

Do not run `git restore --source=d8c5491`, do not alter `FeatureEngineer`, `FeatureSelector`, `scripts/train`, or `RequestFeatureBuilder`, and do not build parquet data. Record `d8c5491` as rejected from this recovery because the train-serving schema hard gate requires a separate approved remediation design.

Expected: no code or test path changes from this candidate. The later review brief must name this as a latent schema-risk finding, not claim that the recovery fixed it.

### Task 5: Obtain sequential cross-family evidence and close the verification matrix

**Files:**

- Modify: none unless a reviewer identifies a narrowly scoped defect in the recovered `a5df516` or `d25fadc` files
- Read: the final recovery diff, the design, this plan, and test output

- [x] **Step 1: Run the complete local verification matrix**

```bash
/mnt/c/model/mode_11_hana/.venv_wsl/bin/python -B -m pytest -p no:cacheprovider -q \
  tests/test_agents/test_codex_lo_orchestration.py \
  tests/test_ops/test_check_py312_drift.py
/mnt/c/model/mode_11_hana/.venv/Scripts/python.exe -m pytest -q \
  tests/test_ops/test_check_py312_drift.py \
  tests/test_ops/test_check_lenient_sunset.py \
  tests/test_serving \
  tests/test_features \
  tests/test_contracts/test_profile_contracts.py \
  tests/test_contracts/test_serving_characterization.py \
  tests/test_contracts/test_reload_artifact_compat.py
/mnt/c/Users/ptg/AppData/Local/Programs/Python/Python312/Scripts/ruff.exe check --no-cache \
  scripts/ops/check_py312_drift.py \
  tests/test_ops/test_check_py312_drift.py \
  scripts/ops/check_lenient_sunset.py \
  serving/predictor.py \
  tests/test_ops/test_check_lenient_sunset.py \
  tests/test_serving/test_health_schema_drift.py \
  tests/test_serving/test_lenient_sunset.py
git diff --check
PYTHONDONTWRITEBYTECODE=1 /mnt/c/model/mode_11_hana/.venv_wsl/bin/python \
  .agents/adapters/protected_artifact_guard.py verify --root . \
  --state /tmp/codex-lo-contract-recovery-protected-baseline.json
```

Expected: all tests and Ruff pass; `git diff --check` is silent; the artifact guard is clean. The dependency-profile diagnostic in Task 2 remains intentionally nonzero and is reported separately, never folded into this pass condition.

- [x] **Step 2: Request one fresh Fable advisor review through `claude-advisor`**

Dispatch exactly one fresh, read-only `claude-advisor` request with this brief:

```text
Review only the recovery branch diff against main. Codex is the sole LO; OpenCode LO must not be restored. Verify (1) a5df516's strict profile checker is fail-closed and its intentional real-profile red result is not hidden; (2) d25fadc blocks noncanonical lenient sunset dates in both operations and serving without changing training features; (3) required serving schema, /reload, and sample-request tests passed; (4) d8c5491 was excluded because sex_male versus sex_m is a separate train-serving parity risk; and (5) no protected, HANA, BAT, frozen-holdout, or authority-policy path changed. Return PASS or FAIL, exact file:line findings, validation status, risks, and one next step. Do not edit files.
```

Expected: the Fable advisor is called exactly once in a fresh session. Wait for completion before any other external worker request. If it reports a defect, change only the named recovered candidate path after a new RED/GREEN cycle; otherwise retain its evidence.

- [x] **Step 3: Request the AGY runtime and risk-gate review only after Fable completes**

Dispatch exactly one read-only `agy-bridge` request with this brief:

```text
Review only the recovery branch diff and recorded command output. Verify Python 3.12 runtime parity, Windows execution of the changed checks, packaging availability without dependency installation, protected-artifact guard results, absence of BAT/HANA/DuckDB/frozen-holdout scope, and the intentionally red dependency-profile diagnostic. Confirm no Windows wheelhouse, constraints, generated parquet, mlruns, or out path changed. Return PASS or FAIL, exact evidence, risks, and one next step. Do not edit files.
```

Expected: AGY begins only after the Fable request is idle. If AGY identifies a wheelhouse, runtime, or protected-path concern, stop rather than trying to repair it without explicit approval.

- [x] **Step 4: Make the Codex LO disposition without a commit**

```bash
git diff --name-only --
git status --short --branch
```

Expected: source/test changes are limited to the two accepted candidates; documentation changes are limited to this design and plan; no `.agents` authority file, OpenCode configuration, protected artifact, generated parquet, model, or output path appears. Report the accepted `a5df516` and `d25fadc` evidence, the intentional dependency-profile red diagnostic, the rejected `d8c5491` schema risk, Fable/AGY findings, and the fact that no commit or push was made.

## Acceptance criteria

- `a5df516` and `d25fadc` pass their focused Python 3.12 tests and scope checks.
- The `d25fadc` serving guard passes the required schema, `/reload`, and sample-request test gates without live artifacts.
- The real dependency-profile diagnostic visibly reports its existing lock gaps; it is not falsely treated as passing or remediated through protected files.
- `d8c5491` is absent from the diff and reported as a separate train-serving schema BLOCK.
- Codex remains the sole LO; no OpenCode LO file or configuration is restored.
- Protected artifact guard, Ruff, and diff checks are clean, and no commit or push occurs.
