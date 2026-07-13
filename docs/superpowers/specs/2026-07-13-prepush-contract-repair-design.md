# Pre-push Contract Repair Design

**Status:** Approved by the user on 2026-07-13. Written-spec review pending.
**Owner:** OpenCode LO
**Scope:** Repair verified pre-push blockers in local commits `6cdc79a` and `6b8dfae` without changing model, serving, HANA, or protected artifacts.
**Related authority:** `AGENTS.md`, `CLAUDE.md`, `docs/superpowers/specs/2026-07-12-opencode-lo-contract-design.md`, and `docs/superpowers/specs/contracts/profile_contracts.md`.

## 1. Context

`main` is two commits ahead of `origin/main`. Full Windows Python 3.12 tests and Ruff pass, but independent pre-push review found safety gaps that the current tests do not detect:

1. `scripts/ops/check_py312_drift.py` accepts an empty constraints file, compares versions with raw string equality, and does not inspect requirements profiles.
2. Raw equality rejects the supported installed version `torch==2.11.0+cu126` against the public pin `torch==2.11.0`, despite PEP 440 compatibility.
3. The approved Phase 1 design promises requirements and constraints comparison, while the implementation checks constraints only.
4. Five active `.agents` documents still route authority and hard stops to the suspended Hermes LO instead of OpenCode LO.
5. An invalid `FEATURE_SCHEMA_LENIENT_SUNSET_DATE` fails closed but prints the false diagnosis that a future date has already passed.
6. Phase 2B requires training Parquet physical column order characterization, but the current contract test asserts constants only.

The review also raised two DL concerns that are not implementation blockers for this repair:

- DL result noninterference already has a behavioral regression test in `tests/test_serving/test_predictor.py`.
- `reload_dl()` swapping after a mocked `load() == False` is an intentionally recorded current weakness. `profile_contracts.md` reserves its production fix for future design review.

## 2. Goals

- Make dependency-profile checking fail closed and PEP 440 correct.
- Detect both requirements compatibility drift and exact-lock gaps.
- Correct the invalid sunset override diagnosis without changing its fail-closed result.
- Make OpenCode LO the only active authority in tracked agent policy documents.
- Add the missing physical output-order characterization without reading or writing a real Parquet artifact.
- Preserve all current model, feature, label, API, HANA, and reload behavior.
- Keep the two existing local commits intact and add bounded repair commits.

## 3. Non-goals

- Do not edit `constraints-py312.txt` in this repair.
- Do not edit `packages_win/py312/`, any wheel file, any BAT file, `mlruns/`, generated Parquet, or `out/`.
- Do not change `serving/`, `hana_app/`, training logic, label definitions, feature names, feature order, or model artifacts.
- Do not change `HybridPredictor.reload_dl()` behavior.
- Do not read or tune the frozen Nov to Dec holdout and do not revive Gate 5A, Gate 5B, or 2025-01 work.
- Do not push until the separate protected constraints gate is resolved or explicitly dispositioned by the user.

## 4. Approaches considered

### 4.1 Selected: strict profile and exact-lock checking

The checker receives one constraints file and one or more explicitly selected requirements files. It uses the public `packaging` APIs for PEP 440 and PEP 508 parsing. It fails on malformed or empty inputs, incompatible installed versions, missing packages, and direct requirements that have no exact lock in either constraints or a selected profile.

This approach reports the repository's current lock gaps honestly and satisfies the Python 3.12 dev/prod parity hard gate.

### 4.2 Rejected: requirements ranges plus exact constraints only

This would validate that installed versions fall inside broad requirements ranges but would allow development and production to use different versions for unpinned packages. It does not satisfy exact parity.

### 4.3 Rejected: constraints-only hotfix

This would reject empty constraints and special-case local version suffixes, but it would continue violating the approved requirements-versus-installed contract and would not model CUDA profiles accurately.

## 5. Detailed design

### 5.1 Strict Python 3.12 dependency profile checker

Files:

- Modify `scripts/ops/check_py312_drift.py`.
- Modify `tests/test_ops/test_check_py312_drift.py`.

CLI contract:

```text
python -m scripts.ops.check_py312_drift constraints-py312.txt \
  -r packages_win/requirements.txt \
  -r hana_app/requirements.txt

python -m scripts.ops.check_py312_drift constraints-py312.txt \
  -r packages_win/requirements.txt \
  -r hana_app/requirements.txt \
  -r packages_win/requirements_cuda_cu126.txt
```

At least one `-r` or `--requirement` argument is mandatory. The tool must not guess CPU, CUDA, UI, or training profiles.

Parsing and comparison rules:

1. Parse requirements with `packaging.requirements.Requirement`.
2. Normalize package names with `packaging.utils.canonicalize_name`.
3. Evaluate installed versions with `packaging.version.Version` and declared specifiers.
4. Treat an empty or comments-only constraints file as a configuration error.
5. Treat an empty or comments-only selected requirements file as a configuration error.
6. Treat malformed active lines as configuration errors with file and line context.
7. Treat an unavailable selected package as drift.
8. Require every selected direct requirement to have an exact `==` lock in the constraints file or one of the selected requirements profiles.
9. Keep the Python runtime check fixed at Python 3.12.
10. Return exit code 1 if runtime, parsing, compatibility, or lock coverage fails.

PEP 440 local-version behavior is explicit:

- Installed `2.11.0+cu126` satisfies public constraint `==2.11.0`.
- If the CUDA requirements profile declares `==2.11.0+cu126`, `+cu126` satisfies it and `+cu128` does not.

The report separates these concepts:

- runtime mismatch;
- configuration or parse errors;
- installed-version drift;
- exact-lock gaps.

The checker uses the existing public `packaging` library. The Windows `.venv` currently contains `packaging 26.2`, and the offline wheelhouse contains packaging wheels. This repair does not modify or pin that protected dependency. Making the tool dependency explicit belongs to the later protected constraints alignment.

Expected repository result after the code repair:

- Unit fixtures with complete locks pass.
- Real selected profiles expose the currently known direct-dependency lock gaps and exit 1.
- Those gaps are not downgraded to warnings to make the command green.

### 5.2 Sunset diagnostic correction

Files:

- Modify `scripts/ops/check_lenient_sunset.py`.
- Modify `tests/test_ops/test_check_lenient_sunset.py`.

`SunsetReport` records a warning reason that distinguishes at least:

- invalid override;
- unavailable authoritative date;
- active setting at or after the sunset date.

An invalid override remains blocked with exit code 1, but its message identifies the invalid environment value. It must not claim that the source default date has passed when that date is still in the future.

### 5.3 Physical training-output order characterization

File:

- Modify `tests/test_contracts/test_profile_contracts.py`.

The new test runs the existing `FeatureEngineer.run()` control flow against a controlled synthetic DataFrame while intercepting file boundaries:

- mock the input existence check;
- mock `pandas.read_parquet()` to return the controlled frame;
- use pass-through normalizer and selector collaborators;
- intercept `DataFrame.to_parquet()` and capture the columns passed to the writer;
- assert the exact returned and write-boundary column order after label and sex transformations.

The test must not create, read, delete, or commit a Parquet file. It characterizes current order for a controlled input and does not invent one universal order for every merge/profile combination.

Serving positional alignment remains covered by `test_feature_vector_uses_bundle_name_order()` in `tests/test_contracts/test_serving_characterization.py`.

### 5.4 OpenCode LO authority alignment

Files:

- Modify `.agents/agy_hq.md`.
- Modify `.agents/claude_hq.md`.
- Modify `.agents/codex_hq.md`.
- Modify `.agents/opencode_hq.md`.
- Modify `.agents/message_deferral_guide.md`.
- Create `tests/test_contracts/test_orchestration_policy.py`.

Only active authority, approval, blocker-return, and hard-stop routing language changes. All such routes point to OpenCode LO.

These historical or infrastructure identifiers remain allowed:

- `Hermes LO` in `.agents/agents_config.json` only as a suspended agent value;
- `Hermes MCP surface` in `.agents/opencode_hq.md` as an infrastructure name;
- tool, wrapper, server, and worker identifiers that contain `hermes` but do not assign authority.

The policy regression test verifies:

- `lo_configuration.lo_agent` is `OpenCode`;
- the research freeze lock remains `HARD_STOP` and owned by all agents;
- the five active documents contain no Hermes authority or routing assignment;
- only the documented historical/infrastructure occurrences remain.

Because this includes freeze and HARD_STOP routing, it is a critical policy change and requires independent Claude-family review before push.

### 5.5 DL scope decision

No production or contract change is made to `reload_dl()`.

- Keep `test_reload_dl_swaps_even_when_load_returns_false()` as an explicit characterization of the recorded current weakness.
- Rely on the existing `test_predict_attaches_dl_prediction_without_changing_final_risk()` behavior test for DL noninterference.
- A future `reload_dl()` fail-closed production change requires a separate design, serving validation, sample payload sanity, and cross-family review.

## 6. Test strategy

The repair follows test-first development for behavior changes.

### 6.1 Red tests before implementation

- empty constraints fail closed;
- comments-only constraints fail closed;
- missing requirements profile fails closed;
- empty requirements profile fails closed;
- public pin accepts the supported CUDA local version;
- exact CUDA local pin rejects the wrong CUDA suffix;
- missing selected dependency fails;
- incompatible requirements range fails;
- unpinned direct requirement produces a lock gap;
- invalid sunset override reports invalid input rather than an elapsed future date;
- tracked active agent documents fail while they still route authority to Hermes.

### 6.2 Characterization-only addition

The physical-order test records existing behavior and may pass on its first correct run. It changes no production code. Its purpose is to close an explicit Phase 2B acceptance-coverage gap.

### 6.3 Focused verification

```text
.venv/Scripts/python.exe -m pytest \
  tests/test_ops/test_check_py312_drift.py \
  tests/test_ops/test_check_lenient_sunset.py -q

.venv/Scripts/python.exe -m pytest \
  tests/test_contracts/test_profile_contracts.py \
  tests/test_contracts/test_orchestration_policy.py \
  tests/test_contracts/test_serving_characterization.py \
  tests/test_serving/test_predictor.py -q
```

### 6.4 Required final verification

- Windows Python 3.12 full pytest suite.
- Ruff check with `--no-cache`.
- `git diff --check`.
- JSON parse of `.agents/agents_config.json`.
- Read-only scan of active policy routes.
- Base and CUDA real profile checker commands, with known lock gaps reported as failures until separately resolved.
- Independent Claude-family logical/policy review.
- Independent Codex technical review.

No `/reload` call or sample serving payload is required for this repair because no serving, training, feature implementation, schema, label, or artifact behavior changes. The cross-family reviewer must confirm that conclusion.

## 7. Protected constraints follow-up gate

Strict checking currently exposes 21 direct dependencies without an exact lock after exact pins declared inside selected requirements are counted. Resolving those gaps may require changes to `constraints-py312.txt` and inspection or cleanup of offline wheel choices.

That work is explicitly outside this repair. It requires:

1. separate explicit user approval for the protected Python 3.12 constraint scope;
2. AGY environment and wheelhouse review;
3. no unapproved writes to `packages_win/py312/`;
4. Windows closed-network installation validation;
5. a new full Python 3.12 test run.

Until that gate is resolved or explicitly dispositioned, the strict real-profile command remains red and `main` is not pushed.

## 8. Commit and rollback strategy

The approved design document is committed separately. Implementation work uses bounded commits so that tooling, characterization, and policy alignment can be reviewed or reverted independently.

No existing commit is amended or rewritten. Rollback uses ordinary non-destructive revert commits if needed. No force push, hard reset, or protected artifact cleanup is allowed.

## 9. Acceptance criteria

- Empty or malformed dependency-profile inputs cannot return status `ok`.
- PEP 440 local CUDA versions are evaluated correctly.
- Requirements compatibility and exact-lock coverage are both reported.
- Invalid sunset overrides produce a truthful diagnosis and remain fail closed.
- The physical training-output write order is characterized without Parquet I/O.
- Active policy documents route BLOCK and HARD_STOP only to OpenCode LO.
- Hermes historical and infrastructure identifiers remain only in the documented allowed locations.
- No serving, model, feature, label, HANA, BAT, protected artifact, or frozen research behavior changes.
- Focused tests, full Windows Python 3.12 tests, Ruff, diff checks, and cross-family reviews pass.
- Real profile lock gaps are reported, not hidden.
- Push remains blocked until the separate protected constraints gate is resolved or explicitly dispositioned.
