# Phase 1: Minimal Tooling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add minimal check-only tooling (pytest config, ruff config, Python 3.12 drift checker, FEATURE_SCHEMA_LENIENT sunset monitor) that automatically detects contract violations, with zero production code behavior change.

**Architecture:** Standalone `pytest.ini` and `ruff.toml` at repo root (no `pyproject.toml`). Two new `scripts/ops/` checkers following the existing `validate_dl_bundle.py` pattern (frozen dataclass report, `build_parser()`, `main() -> int`). Each checker has focused `tests/test_ops/` tests using lazy imports and `tmp_path` fixtures. The sunset checker reads `_FEATURE_SCHEMA_LENIENT_SUNSET_DEFAULT` from `serving/predictor.py` via AST parsing to avoid importing serving (which requires pydantic, unavailable in this venv). Ruff is not installed in the current venv; the config file is written but validation is gated on ruff availability.

**Tech Stack:** Python 3.12 (`.venv`), pytest, stdlib only for checkers (`importlib.metadata`, `ast`, `re`, `argparse`, `dataclasses`, `pathlib`, `datetime`), ruff config file (tool not installed, validation blocked).

**Depends on:** `docs/superpowers/plans/2026-07-12-contract-baseline-inventory.md` (Plan A). Plan A captures the actual pytest baseline (collection IDs, pass/fail set, error set) at execution time. Phase 1 requires that every pre-existing node ID and pass/fail result remains unchanged after Phase 1 changes, while allowing only the approved new tests from this plan.

**Authority sources:**
- Spec: `docs/superpowers/specs/2026-07-12-opencode-lo-contract-design.md` section 7.4
- `AGENTS.md` hard gates and protected paths
- `CLAUDE.md` configured paths and environment
- OpenCode is final LO. All worker results return evidence to OpenCode for verification.

**Freeze-safe declaration:** All tasks are tooling-only. No Nov->Dec holdout tuning, no feature/label/version changes, no artifact migration, no retraining, no Gate 5A/5B activation, no 2025-01 data acquisition. `RESEARCH_TRACK_FROZEN`.

**Rollback:** Revert all Phase 1 commits; delete the 6 new files. No production code is touched, so rollback has zero production impact.

**Oracle dependency:** None. `ask_advisor_panel` unavailable and Oracle blocked; this plan has no Oracle dependency.

---

## File Structure

All files are new. No existing source files are modified.

| File | Responsibility |
|---|---|
| `pytest.ini` | Marker registration, explicit excludes. No addopts, no testpaths, no importmode change. |
| `ruff.toml` | Check-only config (`fix = false`), target py312, F401/F811/I rules, explicit excludes. TID included for import tidy rules but no banned-import config is set (no architecture enforcement claim). |
| `scripts/ops/check_py312_drift.py` | Compares `constraints-py312.txt` pins against installed versions via `importlib.metadata`. Verifies Python 3.12 runtime. Check-only, reports drift. |
| `tests/test_ops/test_check_py312_drift.py` | 7 focused tests for drift checker. |
| `scripts/ops/check_lenient_sunset.py` | Warns if `FEATURE_SCHEMA_LENIENT=1` is set after sunset date. Reads sunset default from `serving/predictor.py` via AST parsing (no serving import). |
| `tests/test_ops/test_check_lenient_sunset.py` | 11 focused tests for sunset monitor. |

**Files NOT touched:** All production code (`serving/`, `hana_app/`, `scripts/` existing, `dags/`, `rules/`), `constraints-py312.txt`, requirements files, `packages_win/py312/`, `mlruns/`, `data/`, `models/`, any `.bat` file, existing tests, `CLAUDE.md`, `AGENTS.md`, `.gitignore`.

---

## Pre-flight Checks

- [ ] **Pre-flight Step 1: Verify Python 3.12 and ruff gate**

Run:
```bash
source .venv/bin/activate
python --version
python -c "import ruff" 2>&1 || echo "ruff not installed"
git status --short
```
Expected: `Python 3.12.x`. Ruff is NOT installed (config validation is BLOCKED, not a failure). Record working-tree state. Do NOT install ruff, modify `constraints-py312.txt`, or touch `packages_win/py312/`. All commits stage ONLY the 6 Phase 1 files.

---

## Task 1: Capture pytest baseline (from Plan A)

Plan A captures the actual pytest baseline at execution time. Phase 1 does NOT hardcode any test count or error count. Phase 1 requires that every pre-existing node ID and pass/fail result remains unchanged, while allowing only the approved new tests from this plan.

- [ ] **Step 1: Capture collection and pass/fail baseline**

Run:
```bash
source .venv/bin/activate
python -m pytest --collect-only -q > /tmp/opencode/phase1_baseline_collect.txt 2>&1
python -m pytest -q --tb=no -rA > /tmp/opencode/phase1_baseline_passfail.txt 2>&1
wc -l /tmp/opencode/phase1_baseline_collect.txt /tmp/opencode/phase1_baseline_passfail.txt
```
Expected: Both files have content (nonzero line count). These are the baselines that pre-existing tests must match after Phase 1.

---

## Task 2: Create `pytest.ini`

- [ ] **Step 1: Write `pytest.ini`**

```ini
[pytest]
# Phase 1: minimal pytest config. Marker registration + explicit excludes.
# Do NOT change import mode (default prepend) or rootdir.
# Do NOT add addopts that change collection behavior.

# Contract-related markers (for future test tagging, not applied to existing tests here).
markers =
    contract: profile contract validation tests
    schema: feature schema validation tests
    drift: dependency constraint drift tests
    sunset: FEATURE_SCHEMA_LENIENT sunset monitor tests
    characterization: Phase 2B characterization tests (freeze-safe)

# Explicit excludes. Protected/generated/venv paths are not test directories.
# Prevents accidental collection if a stray test file appears.
norecursedirs =
    .venv
    .venv_hana
    .venv_macos
    .venv_wsl
    .venv.bak
    .venv.bak.py311
    packages_win
    mlruns
    models
    data
    drugbank
    graphify-out
    .understand-anything
    .agents
    .multiagent
    .worktrees
    cache
    debug_logs
    reviews
    hana
    hira
    lay_out
    python
    error
    out
```

- [ ] **Step 2: Verify no forbidden keys and collection unchanged**

Run:
```bash
grep -E "^\[project\]|^\[build-system\]|^addopts|^testpaths|^importmode|^dependencies" pytest.ini
python -m pytest --collect-only -q > /tmp/opencode/phase1_after_step1_collect.txt 2>&1
diff /tmp/opencode/phase1_baseline_collect.txt /tmp/opencode/phase1_after_step1_collect.txt
```
Expected: grep produces no output. `diff` produces no output (collection unchanged).

- [ ] **Step 3: Commit**

```bash
git add pytest.ini
git commit -m "chore(tooling): add pytest.ini with markers and norecursedirs

- Register contract/schema/drift/sunset/characterization markers
- Set norecursedirs for protected/generated/venv paths
- No addopts, no testpaths, no importmode change
- Existing pytest collection unchanged"
```

---

## Task 3: Create `ruff.toml`

Ruff is NOT installed. The config file is written now for when ruff becomes available. Config validation is BLOCKED. Do NOT install ruff, modify `constraints-py312.txt`, or touch `packages_win/py312/`.

- [ ] **Step 1: Write `ruff.toml`**

```toml
# Phase 1: ruff check-only config. Contract-related rules, no autofix.
# Ruff is NOT installed in .venv. Config validation is BLOCKED until ruff
# is available in an approved Python 3.12 environment.
# Run when available: ruff check . (no --fix flag)

target-version = "py312"

# Check-only: never autofix.
fix = false

# Contract-related rules:
#   F401  - unused imports
#   F811  - redefined-while-unused
#   I     - isort rules (import ordering)
#   TID   - tidy-imports (banned-imports, relative-imports)
# Note: TID is included for import tidy rules. No banned-import
# configuration is set in this file. TID does NOT enforce architecture
# boundaries unless explicit banned-imports are configured (not done here).
[lint]
select = ["F401", "F811", "I", "TID"]
ignore = []

# Explicit excludes. Protected/generated/venv/non-source paths.
extend-exclude = [
    ".venv",
    ".venv_hana",
    ".venv_macos",
    ".venv_wsl",
    ".venv.bak",
    ".venv.bak.py311",
    "__pycache__",
    ".pytest_cache",
    ".uv-cache",
    "packages_win",
    "mlruns",
    "models",
    "data",
    "drugbank",
    "graphify-out",
    ".understand-anything",
    ".agents",
    ".multiagent",
    ".worktrees",
    "cache",
    "debug_logs",
    "reviews",
    "out",
    "hana",
    "hira",
    "lay_out",
    "python",
    "error",
    "desktop_app.py",
]
```

- [ ] **Step 2: Verify no forbidden sections and no pyproject.toml**

Run:
```bash
grep -E "^\[project\]|^\[build-system\]|^dependencies|^fix = true" ruff.toml
ls pyproject.toml 2>&1 || echo "no pyproject.toml (correct)"
```
Expected: grep produces no output. `no pyproject.toml (correct)`.

- [ ] **Step 3: Ruff config validation (BLOCKED gate)**

Run:
```bash
python -c "import ruff" 2>&1 || echo "BLOCKED: ruff not installed"
```
Expected in the current environment: `BLOCKED: ruff not installed`. Continue independent checker work, but do not mark Phase 1 complete. Completion remains blocked until an approved Python 3.12 environment provides Ruff and `ruff check --show-settings .` parses this configuration successfully. Do NOT install Ruff or modify constraints to unblock this plan.

- [ ] **Step 4: Verify collection still unchanged**

Run:
```bash
python -m pytest --collect-only -q > /tmp/opencode/phase1_after_step2_collect.txt 2>&1
diff /tmp/opencode/phase1_baseline_collect.txt /tmp/opencode/phase1_after_step2_collect.txt
```
Expected: `diff` produces no output.

- [ ] **Step 5: Commit**

```bash
git add ruff.toml
git commit -m "chore(tooling): add ruff.toml check-only config

- fix=false (check-only, no autofix)
- target-version py312, F401/F811/I/TID rules
- extend-exclude for protected/generated/venv paths
- Ruff not installed: config validation BLOCKED gate
- No pyproject.toml, no [project]/[build-system]"
```

---

## Task 4: Create `scripts/ops/check_py312_drift.py` (TDD)

**Pattern:** Follows `scripts/ops/validate_dl_bundle.py`: `from __future__ import annotations`, frozen dataclass report, `build_parser()`, `main(argv) -> int`. Stdlib only.

- [ ] **Step 1: Write the failing test**

```python
from __future__ import annotations

import importlib.metadata
import sys
from unittest.mock import patch


def _write_constraints(path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _base_constraints() -> list[str]:
    return [
        "# constraints-py312.txt",
        "",
        "numpy==2.4.3",
        "pandas==2.3.3",
        "scikit-learn==1.8.0",
        "",
        "# comment line",
        "pydantic==2.12.5",
    ]


def test_check_drift_no_drift_when_all_match(tmp_path) -> None:
    from scripts.ops.check_py312_drift import check_drift

    constraints = tmp_path / "constraints-py312.txt"
    _write_constraints(constraints, _base_constraints())

    fake_versions = {
        "numpy": "2.4.3",
        "pandas": "2.3.3",
        "scikit-learn": "1.8.0",
        "pydantic": "2.12.5",
    }

    with patch.object(importlib.metadata, "version",
                      side_effect=lambda pkg: fake_versions[pkg]):
        report = check_drift(constraints)

    assert report.ok is True
    assert report.drifts == []


def test_check_drift_detects_version_mismatch(tmp_path) -> None:
    from scripts.ops.check_py312_drift import check_drift

    constraints = tmp_path / "constraints-py312.txt"
    _write_constraints(constraints, _base_constraints())

    fake_versions = {
        "numpy": "2.5.0",
        "pandas": "2.3.3",
        "scikit-learn": "1.8.0",
        "pydantic": "2.12.5",
    }

    with patch.object(importlib.metadata, "version",
                      side_effect=lambda pkg: fake_versions[pkg]):
        report = check_drift(constraints)

    assert report.ok is False
    assert any(d.package == "numpy" and d.pinned == "2.4.3" and d.installed == "2.5.0"
               for d in report.drifts)


def test_check_drift_detects_missing_package(tmp_path) -> None:
    from scripts.ops.check_py312_drift import check_drift

    constraints = tmp_path / "constraints-py312.txt"
    _write_constraints(constraints, _base_constraints())

    def fake_version(pkg):
        if pkg == "pydantic":
            raise importlib.metadata.PackageNotFoundError(pkg)
        return {"numpy": "2.4.3", "pandas": "2.3.3", "scikit-learn": "1.8.0"}[pkg]

    with patch.object(importlib.metadata, "version", side_effect=fake_version):
        report = check_drift(constraints)

    assert report.ok is False
    assert any(d.package == "pydantic" and "<not installed>" in d.installed
               for d in report.drifts)


def test_check_drift_skips_comment_and_blank_lines(tmp_path) -> None:
    from scripts.ops.check_py312_drift import _parse_constraints

    constraints = tmp_path / "constraints-py312.txt"
    _write_constraints(constraints, [
        "# header comment",
        "",
        "numpy==2.4.3",
        "  # indented comment",
        "pandas==2.3.3",
    ])

    pins = _parse_constraints(constraints)

    assert pins == {"numpy": "2.4.3", "pandas": "2.3.3"}


def test_check_drift_python_version_check(tmp_path) -> None:
    from scripts.ops.check_py312_drift import check_drift

    constraints = tmp_path / "constraints-py312.txt"
    _write_constraints(constraints, _base_constraints())

    fake_versions = {
        "numpy": "2.4.3",
        "pandas": "2.3.3",
        "scikit-learn": "1.8.0",
        "pydantic": "2.12.5",
    }

    with patch.object(importlib.metadata, "version",
                      side_effect=lambda pkg: fake_versions[pkg]), \
         patch.object(sys, "version_info", (3, 11, 0, "final", 0)):
        report = check_drift(constraints)

    assert report.ok is False
    assert report.python_ok is False
    assert "3.11" in report.python_version


def test_check_drift_cli_returns_zero_when_ok(tmp_path) -> None:
    from scripts.ops.check_py312_drift import main

    constraints = tmp_path / "constraints-py312.txt"
    _write_constraints(constraints, _base_constraints())

    fake_versions = {
        "numpy": "2.4.3",
        "pandas": "2.3.3",
        "scikit-learn": "1.8.0",
        "pydantic": "2.12.5",
    }

    with patch.object(importlib.metadata, "version",
                      side_effect=lambda pkg: fake_versions[pkg]):
        assert main([str(constraints)]) == 0


def test_check_drift_cli_returns_nonzero_on_drift(tmp_path) -> None:
    from scripts.ops.check_py312_drift import main

    constraints = tmp_path / "constraints-py312.txt"
    _write_constraints(constraints, _base_constraints())

    fake_versions = {
        "numpy": "99.0.0",
        "pandas": "2.3.3",
        "scikit-learn": "1.8.0",
        "pydantic": "2.12.5",
    }

    with patch.object(importlib.metadata, "version",
                      side_effect=lambda pkg: fake_versions[pkg]):
        assert main([str(constraints)]) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
python -m pytest tests/test_ops/test_check_py312_drift.py -v --tb=short 2>&1 | tail -20
```
Expected: All 7 tests FAIL with `ModuleNotFoundError: No module named 'scripts.ops.check_py312_drift'`.

- [ ] **Step 3: Write minimal implementation**

```python
"""Check installed package versions against constraints-py312.txt pins.

Check-only: reports drift, does not fix. Verifies Python 3.12 runtime parity.
Exit code 0 = no drift, 1 = drift or Python version mismatch.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class DriftItem:
    package: str
    pinned: str
    installed: str


@dataclass(frozen=True)
class DriftReport:
    python_version: str
    python_ok: bool
    drifts: list[DriftItem] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.python_ok and not self.drifts


def _parse_constraints(path: Path) -> dict[str, str]:
    """Parse == pins from constraints file. Skips comments and blank lines."""
    pins: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        m = re.match(r"^([a-zA-Z0-9_.-]+)==([0-9a-zA-Z.]+)", stripped)
        if m:
            pins[m.group(1).lower()] = m.group(2)
    return pins


def check_drift(constraints_path: Path | str) -> DriftReport:
    """Compare constraints pins against installed package versions."""
    constraints_path = Path(constraints_path)
    pins = _parse_constraints(constraints_path)
    vi = sys.version_info
    python_version = f"{vi[0]}.{vi[1]}.{vi[2]}"
    python_ok = vi[:2] == (3, 12)
    drifts: list[DriftItem] = []
    for pkg, pinned in sorted(pins.items()):
        try:
            installed = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            drifts.append(DriftItem(package=pkg, pinned=pinned, installed="<not installed>"))
            continue
        if installed != pinned:
            drifts.append(DriftItem(package=pkg, pinned=pinned, installed=installed))
    return DriftReport(
        python_version=python_version,
        python_ok=python_ok,
        drifts=drifts,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check installed package versions against constraints-py312.txt pins.",
    )
    parser.add_argument(
        "constraints_path",
        nargs="?",
        default="constraints-py312.txt",
        help="Path to constraints file (default: constraints-py312.txt)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = check_drift(args.constraints_path)
    if not report.python_ok:
        print(f"ERROR: Python {report.python_version} != 3.12 (dev/prod parity violation)")
    for d in report.drifts:
        print(f"DRIFT: {d.package} pinned={d.pinned} installed={d.installed}")
    print(f"python={report.python_version} python_ok={report.python_ok} drifts={len(report.drifts)}")
    print(f"status={'ok' if report.ok else 'failed'}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
python -m pytest tests/test_ops/test_check_py312_drift.py -v --tb=short 2>&1 | tail -20
```
Expected: All 7 tests PASS.

- [ ] **Step 5: LSP diagnostics and real constraints check**

Run `lsp_diagnostics` on both files. Expected: zero errors.
Run:
```bash
python -m scripts.ops.check_py312_drift constraints-py312.txt
```
Expected: Tool runs without crashing. `python=3.12.x python_ok=True`. Exit code 1 if pinned packages missing (informational).

- [ ] **Step 6: Commit**

```bash
git add scripts/ops/check_py312_drift.py tests/test_ops/test_check_py312_drift.py
git commit -m "feat(ops): add Python 3.12 dependency constraint drift checker

- scripts/ops/check_py312_drift.py: compares constraints-py312.txt pins
  against installed versions via importlib.metadata, verifies Python 3.12
- tests/test_ops/test_check_py312_drift.py: 7 focused tests
- Check-only, stdlib only, no production code change"
```

---

## Task 5: Create `scripts/ops/check_lenient_sunset.py` (TDD)

**Pattern:** Follows `scripts/ops/validate_dl_bundle.py`. Uses `ast` to read `_FEATURE_SCHEMA_LENIENT_SUNSET_DEFAULT` from `serving/predictor.py` without importing it (avoids pydantic). Stdlib only.

- [ ] **Step 1: Write the failing test**

```python
from __future__ import annotations

from datetime import date
from unittest.mock import patch


def test_read_sunset_default_from_source_returns_2026_08_01() -> None:
    """AST parser reads the actual constant from serving/predictor.py."""
    from scripts.ops.check_lenient_sunset import _read_sunset_default_from_source

    result = _read_sunset_default_from_source()

    assert result == date(2026, 8, 1)


def test_check_sunset_no_warning_when_lenient_unset(monkeypatch) -> None:
    from scripts.ops.check_lenient_sunset import check_sunset

    monkeypatch.delenv("FEATURE_SCHEMA_LENIENT", raising=False)
    monkeypatch.delenv("FEATURE_SCHEMA_LENIENT_SUNSET_DATE", raising=False)

    report = check_sunset(today=date(2026, 12, 31))

    assert report.ok is True
    assert report.warning is False
    assert report.lenient_active is False


def test_check_sunset_no_warning_when_before_sunset(monkeypatch) -> None:
    from scripts.ops.check_lenient_sunset import check_sunset

    monkeypatch.setenv("FEATURE_SCHEMA_LENIENT", "1")
    monkeypatch.delenv("FEATURE_SCHEMA_LENIENT_SUNSET_DATE", raising=False)

    report = check_sunset(today=date(2026, 7, 1))

    assert report.ok is True
    assert report.warning is False
    assert report.lenient_active is True


def test_check_sunset_warns_when_lenient_set_after_sunset(monkeypatch) -> None:
    from scripts.ops.check_lenient_sunset import check_sunset

    monkeypatch.setenv("FEATURE_SCHEMA_LENIENT", "1")
    monkeypatch.delenv("FEATURE_SCHEMA_LENIENT_SUNSET_DATE", raising=False)

    report = check_sunset(today=date(2026, 8, 1))

    assert report.ok is False
    assert report.warning is True
    assert report.lenient_active is False


def test_check_sunset_warns_when_lenient_set_on_sunset_date(monkeypatch) -> None:
    from scripts.ops.check_lenient_sunset import check_sunset

    monkeypatch.setenv("FEATURE_SCHEMA_LENIENT", "1")
    monkeypatch.delenv("FEATURE_SCHEMA_LENIENT_SUNSET_DATE", raising=False)

    report = check_sunset(today=date(2026, 8, 1))

    assert report.ok is False
    assert report.warning is True


def test_check_sunset_env_override_extends_sunset(monkeypatch) -> None:
    from scripts.ops.check_lenient_sunset import check_sunset

    monkeypatch.setenv("FEATURE_SCHEMA_LENIENT", "1")
    monkeypatch.setenv("FEATURE_SCHEMA_LENIENT_SUNSET_DATE", "2027-12-31")

    report = check_sunset(today=date(2027, 6, 1))

    assert report.ok is True
    assert report.warning is False
    assert report.lenient_active is True


def test_check_sunset_invalid_env_date_blocks_lenient(monkeypatch) -> None:
    from scripts.ops.check_lenient_sunset import check_sunset

    monkeypatch.setenv("FEATURE_SCHEMA_LENIENT", "1")
    monkeypatch.setenv("FEATURE_SCHEMA_LENIENT_SUNSET_DATE", "garbage-date")

    report = check_sunset(today=date(2026, 1, 1))

    assert report.ok is False
    assert report.warning is True
    assert report.lenient_active is False


def test_check_sunset_empty_env_uses_default(monkeypatch) -> None:
    from scripts.ops.check_lenient_sunset import check_sunset

    monkeypatch.setenv("FEATURE_SCHEMA_LENIENT", "1")
    monkeypatch.setenv("FEATURE_SCHEMA_LENIENT_SUNSET_DATE", "")

    report = check_sunset(today=date(2026, 7, 1))

    assert report.ok is True
    assert report.warning is False
    assert report.lenient_active is True


def test_check_sunset_cli_returns_zero_when_ok(monkeypatch) -> None:
    from scripts.ops.check_lenient_sunset import main

    monkeypatch.delenv("FEATURE_SCHEMA_LENIENT", raising=False)
    monkeypatch.delenv("FEATURE_SCHEMA_LENIENT_SUNSET_DATE", raising=False)

    with patch("scripts.ops.check_lenient_sunset.date") as mock_date:
        mock_date.today.return_value = date(2026, 7, 1)
        mock_date.side_effect = lambda *a, **k: date(*a, **k)
        assert main([]) == 0


def test_check_sunset_cli_returns_nonzero_when_warning(monkeypatch) -> None:
    from scripts.ops.check_lenient_sunset import main

    monkeypatch.setenv("FEATURE_SCHEMA_LENIENT", "1")
    monkeypatch.delenv("FEATURE_SCHEMA_LENIENT_SUNSET_DATE", raising=False)

    with patch("scripts.ops.check_lenient_sunset.date") as mock_date:
        mock_date.today.return_value = date(2026, 8, 1)
        mock_date.side_effect = lambda *a, **k: date(*a, **k)
        assert main([]) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
python -m pytest tests/test_ops/test_check_lenient_sunset.py -v --tb=short 2>&1 | tail -20
```
Expected: All 11 tests FAIL with `ModuleNotFoundError: No module named 'scripts.ops.check_lenient_sunset'`.

- [ ] **Step 3: Write minimal implementation**

```python
"""Check if FEATURE_SCHEMA_LENIENT escape hatch is active past its sunset date.

Check-only: warns if FEATURE_SCHEMA_LENIENT=1 is set after the sunset date.
Reads the sunset default from serving/predictor.py via AST parsing (does NOT
import serving, to avoid pydantic dependency).
Exit code 0 = no warning, 1 = sunset warning.
"""
from __future__ import annotations

import argparse
import ast
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Sequence


def _read_sunset_default_from_source() -> date:
    """Read _FEATURE_SCHEMA_LENIENT_SUNSET_DEFAULT from serving/predictor.py via AST.

    Avoids importing serving.predictor (which requires pydantic) by parsing
    the source file directly. Falls back to date(2026, 8, 1) if parsing fails.
    """
    predictor_path = Path(__file__).resolve().parents[2] / "serving" / "predictor.py"
    try:
        tree = ast.parse(predictor_path.read_text(encoding="utf-8"))
    except Exception:
        return date(2026, 8, 1)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (isinstance(target, ast.Name)
                        and target.id == "_FEATURE_SCHEMA_LENIENT_SUNSET_DEFAULT"):
                    if (isinstance(node.value, ast.Call)
                            and isinstance(node.value.func, ast.Name)
                            and node.value.func.id == "date"
                            and len(node.value.args) == 3):
                        try:
                            return date(
                                ast.literal_eval(node.value.args[0]),
                                ast.literal_eval(node.value.args[1]),
                                ast.literal_eval(node.value.args[2]),
                            )
                        except Exception:
                            return date(2026, 8, 1)
    return date(2026, 8, 1)


@dataclass(frozen=True)
class SunsetReport:
    lenient_env: bool
    lenient_active: bool
    sunset_date: date
    today: date
    warning: bool

    @property
    def ok(self) -> bool:
        return not self.warning


def check_sunset(today: date | None = None) -> SunsetReport:
    """Check if FEATURE_SCHEMA_LENIENT is active past its sunset date."""
    today = today or date.today()
    lenient_env = os.environ.get("FEATURE_SCHEMA_LENIENT", "").strip().lower() in (
        "1", "true", "yes",
    )
    raw_sunset = os.environ.get("FEATURE_SCHEMA_LENIENT_SUNSET_DATE", "").strip()
    if raw_sunset:
        from datetime import datetime
        try:
            sunset = datetime.strptime(raw_sunset, "%Y-%m-%d").date()
        except ValueError:
            # Invalid env date: safe side, block lenient (per Codex 2026-05-07 #6)
            sunset = _read_sunset_default_from_source()
            lenient_active = False
            warning = lenient_env and today >= sunset
            return SunsetReport(
                lenient_env=lenient_env,
                lenient_active=lenient_active,
                sunset_date=sunset,
                today=today,
                warning=warning,
            )
    else:
        sunset = _read_sunset_default_from_source()
    lenient_active = lenient_env and today < sunset
    warning = lenient_env and today >= sunset
    return SunsetReport(
        lenient_env=lenient_env,
        lenient_active=lenient_active,
        sunset_date=sunset,
        today=today,
        warning=warning,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check if FEATURE_SCHEMA_LENIENT is active past its sunset date.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    build_parser().parse_args(argv)
    report = check_sunset()
    if report.warning:
        print(
            f"WARNING: FEATURE_SCHEMA_LENIENT=1 is set but sunset date "
            f"{report.sunset_date} has passed (today={report.today}). "
            f"Lenient is blocked. Align train/serve schema and unset the env var."
        )
    else:
        print(
            f"lenient_env={report.lenient_env} lenient_active={report.lenient_active} "
            f"sunset={report.sunset_date} today={report.today} status=ok"
        )
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
python -m pytest tests/test_ops/test_check_lenient_sunset.py -v --tb=short 2>&1 | tail -20
```
Expected: All 11 tests PASS.

- [ ] **Step 5: LSP diagnostics and real source check**

Run `lsp_diagnostics` on both files. Expected: zero errors.
Run:
```bash
python -m scripts.ops.check_lenient_sunset
```
Expected: Tool runs without crashing. Sunset date reads `2026-08-01` (from AST). Exit code 0 if `FEATURE_SCHEMA_LENIENT` not set.

- [ ] **Step 6: Commit**

```bash
git add scripts/ops/check_lenient_sunset.py tests/test_ops/test_check_lenient_sunset.py
git commit -m "feat(ops): add FEATURE_SCHEMA_LENIENT sunset monitor

- scripts/ops/check_lenient_sunset.py: warns if FEATURE_SCHEMA_LENIENT=1
  is set after sunset date, reads sunset default from serving/predictor.py
  via AST parsing (no serving import, avoids pydantic dependency)
- tests/test_ops/test_check_lenient_sunset.py: 11 focused tests
- Check-only, stdlib only, no production code change"
```

---

## Task 6: Full pytest regression verification

**Goal:** Confirm that Phase 1 changes did not alter any pre-existing node ID or pass/fail result. Only tests from the two approved new Phase 1 test files may be added.

- [ ] **Step 1: Collection comparison**

Run:
```bash
python -m pytest --collect-only -q > /tmp/opencode/phase1_final_collect.txt 2>&1
diff /tmp/opencode/phase1_baseline_collect.txt /tmp/opencode/phase1_final_collect.txt
```
Expected: set comparison shows that every baseline node ID remains and that every added node ID belongs to `test_check_py312_drift.py` or `test_check_lenient_sunset.py`. No existing test ID or existing ERROR entry is removed or changed.

- [ ] **Step 2: Run new tests**

Run:
```bash
python -m pytest tests/test_ops/test_check_py312_drift.py tests/test_ops/test_check_lenient_sunset.py -v --tb=short 2>&1 | tail -30
```
Expected: All tests collected from the two approved new test files pass. Zero failures, zero errors.

- [ ] **Step 3: Run existing test_ops tests (regression check)**

Run:
```bash
python -m pytest tests/test_ops/ -q --tb=short 2>&1 | tail -10
```
Expected: No new failures introduced. Pre-existing collection errors (if any) are unchanged from baseline.

- [ ] **Step 4: Verify git status shows only Phase 1 files**

Run:
```bash
git status --short
```
Expected: Only the 6 Phase 1 files appear as new/modified. Pre-existing working-tree changes are NOT staged.

---

## Task 7: OpenCode verification

- [ ] **Step 1: Verify all checks pass**

Run:
```bash
# LSP diagnostics on all 4 new Python files (via lsp_diagnostics tool)
# Verify no production code modified:
git diff --stat HEAD~3 -- serving/ hana_app/ scripts/etl/ scripts/features/ scripts/train/ scripts/datasets/ dags/ rules/
# Verify no protected paths touched:
git diff --stat HEAD~3 -- packages_win/ mlruns/ data/ models/ out/ graphify-out/
# Verify no BAT files touched:
git diff --stat HEAD~3 -- '*.bat'
# Verify no pyproject.toml:
ls pyproject.toml 2>&1 || echo "no pyproject.toml (correct)"
```
Expected: LSP zero errors on all 4 files. All git diff commands produce no output. `no pyproject.toml (correct)`. If Ruff remains unavailable, report Phase 1 as BLOCKED rather than complete. In an approved environment with Ruff, run `ruff check --show-settings .` and require exit code 0 before completion.

---

## Task 8: AGY review (Python 3.12 / Windows)

AGY HQ reviews for Python 3.12 parity and Windows closed-network implications. No BAT edits expected.

- [ ] **Step 1: AGY review**

```bash
# agy -p "Review Phase 1 tooling files for Python 3.12 Windows closed-network
# compatibility. Check: no BAT edits needed, no external dependencies,
# packages_win excluded, importlib.metadata available in py312,
# AST parsing works on Windows paths."
```
Expected: AGY review completed or noted as unavailable. No BLOCK or HARD_STOP triggers.

---

## Acceptance Criteria

| Criterion | Verification method |
|---|---|
| Pre-existing node IDs and pass/fail results unchanged | Set comparison: baseline IDs remain; added IDs come only from the two approved test files |
| New tests pass | All tests in the two approved test files pass |
| Ruff check-only, no autofix | `ruff.toml` has `fix = false` |
| Ruff config validation gate | Approved Python 3.12 environment runs `ruff check --show-settings .` with exit code 0; unavailable Ruff keeps Phase 1 BLOCKED |
| Explicit excludes | `pytest.ini` norecursedirs + `ruff.toml` extend-exclude cover all protected paths |
| Drift checker works | `check_py312_drift.py` runs, reports drift, exit 0/1 |
| Python 3.12 parity check | `check_py312_drift.py` checks `sys.version_info[:2] == (3, 12)` |
| Sunset monitor works | `check_lenient_sunset.py` runs, warns post-sunset, exit 0/1 |
| Sunset reads source via AST | `_read_sunset_default_from_source()` returns `date(2026, 8, 1)` |
| No production code change | `git diff` shows no changes in production code |
| No `[project]`/deps/`[build-system]` | No such sections; no `pyproject.toml` |
| No protected paths touched | `packages_win/py312/`, `mlruns/`, `data/`, `models/` unmodified |
| No BAT edits | No `.bat` files changed |
| Freeze-safe | No Nov->Dec holdout, Gate 5A/5B, 2025-01, feature/label/version changes |

---

## Rollback

```bash
git log --oneline -- pytest.ini ruff.toml scripts/ops/check_py312_drift.py scripts/ops/check_lenient_sunset.py tests/test_ops/test_check_py312_drift.py tests/test_ops/test_check_lenient_sunset.py
# Review the listed Phase 1 commits, then run git revert once per listed hash,
# newest first. If changes are uncommitted, remove only the six files above.
```

Zero production impact. Test collection reverts to Plan A baseline. No dependency changes.

**Triggers:** pre-existing node ID/pass-fail changed, production code behavior change, AGY BLOCK/HARD_STOP, checker crash on Windows Python 3.12.

**Verify after rollback:** `git status` clean, `pytest --collect-only -q` matches Plan A baseline, no production code in `git diff`.
