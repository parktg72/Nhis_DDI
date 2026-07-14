# Global Codex LO Multiagent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Install a project-neutral Codex-LO multiagent base under `~/.codex` so new Codex sessions launched from any directory can dispatch bounded Claude Code/Fable 5 and AGY workers while MODE_11_hana retains its stronger local policy.

**Architecture:** Versioned templates and a Python installer live in this repository. The installer tests against an isolated `CODEX_HOME`, merges only owned TOML keys, installs absolute-path bridge profiles plus an in-memory-capture adapter atomically, and rolls back on failure. Project `.codex/config.toml` and `AGENTS.md` remain the specialization layer.

**Tech Stack:** Python 3.12 standard library, `tomllib`, pytest, Codex CLI strict diagnostics, Claude Code CLI, AGY CLI.

---

## Safety preflight

- Execute in an isolated Git worktree created with `using-git-worktrees`.
- Preserve existing `.understand-anything/**` changes and the unrelated untracked prepush plan.
- Do not touch `packages_win/py312/`, `mlruns/`, generated parquet files, `out/`, or BAT files.
- Keep MODE_11_hana, HANA, frozen-track, BAT, Python-version, and protected-artifact rules out of global templates.
- Never modify `~/.claude/settings.json`.
- Do not call a live external provider during unit tests.
- Real `~/.codex` writes require escalated approval after isolated validation.
- Do not push.

Run before Task 1:

```bash
git status --short --untracked-files=all
/mnt/c/model/mode_11_hana/.venv_wsl/bin/python --version
/mnt/c/model/mode_11_hana/.venv_wsl/bin/python .agents/adapters/protected_artifact_guard.py snapshot --root . --state /tmp/global-codex-preflight-protected.json
```

Expected: Python 3.12, only known user changes, and a protected metadata snapshot outside the repository.

## File map

Create:

- `tools/codex_global_multiagent/__init__.py` — installer version.
- `tools/codex_global_multiagent/install.py` — config merge, dry-run, install, manifest, rollback, and CLI.
- `tools/codex_global_multiagent/templates/claude-bridge.toml` — generic Claude/Fable bridge.
- `tools/codex_global_multiagent/templates/agy-bridge.toml` — generic AGY bridge.
- `tools/codex_global_multiagent/templates/call_external_agent.py` — in-memory provider adapter.
- `tools/codex_global_multiagent/templates/claude-advisor-settings.json` — isolated Fable setting.
- `tests/test_agents/test_global_codex_multiagent.py` — global installation tests.

Modify:

- `README.md` — installation, start, verification, and recovery.

Install outside the repository:

- `~/.codex/config.toml`
- `~/.codex/agents/claude-bridge.toml`
- `~/.codex/agents/agy-bridge.toml`
- `~/.codex/multiagent/call_external_agent.py`
- `~/.codex/multiagent/claude-advisor-settings.json`
- `~/.codex/multiagent/manifest.json`

## Task 1: Lock the generic bridge contract

**Files:**

- Create: `tools/codex_global_multiagent/__init__.py`
- Create: `tools/codex_global_multiagent/templates/claude-bridge.toml`
- Create: `tools/codex_global_multiagent/templates/agy-bridge.toml`
- Create: `tools/codex_global_multiagent/templates/claude-advisor-settings.json`
- Create: `tests/test_agents/test_global_codex_multiagent.py`

- [ ] **Step 1: Write failing template tests**

Create the test module with:

```python
from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tomllib

import pytest


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "tools" / "codex_global_multiagent"
TEMPLATES = PACKAGE / "templates"
FORBIDDEN = (
    "MODE_11_hana",
    "/mnt/c/model/mode_11_hana",
    "HANA",
    "RESEARCH_TRACK_FROZEN",
    "Gate 5A",
    "Gate 5B",
    "chcp 65001",
    "packages_win/py312",
    "OpenCode",
)


def read_toml(name: str) -> dict:
    return tomllib.loads((TEMPLATES / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", ["claude-bridge.toml", "agy-bridge.toml"])
def test_global_bridge_is_generic_read_only_and_codex_owned(name: str) -> None:
    text = (TEMPLATES / name).read_text(encoding="utf-8")
    config = read_toml(name)
    instructions = config["developer_instructions"]
    assert config["sandbox_mode"] == "read-only"
    assert "Codex LO owns" in instructions
    assert "do not contact the user" in instructions
    assert "do not call another subagent" in instructions
    assert not any(term.casefold() in text.casefold() for term in FORBIDDEN)


def test_claude_bridge_has_worker_and_one_shot_advisor() -> None:
    instructions = read_toml("claude-bridge.toml")["developer_instructions"]
    assert "call_external_agent.py claude" in instructions
    assert "call_external_agent.py claude-advisor" in instructions
    assert "exactly once" in instructions
    assert "--model fable" not in instructions


def test_agy_bridge_has_only_agy() -> None:
    instructions = read_toml("agy-bridge.toml")["developer_instructions"]
    assert "call_external_agent.py agy" in instructions
    assert "claude-advisor" not in instructions


def test_advisor_settings_are_minimal() -> None:
    settings = json.loads(
        (TEMPLATES / "claude-advisor-settings.json").read_text(encoding="utf-8")
    )
    assert settings == {"advisorModel": "fable"}
```

- [ ] **Step 2: Run tests and verify RED**

```bash
/mnt/c/model/mode_11_hana/.venv_wsl/bin/python -m pytest tests/test_agents/test_global_codex_multiagent.py -v
```

Expected: failures because templates do not exist.

- [ ] **Step 3: Create the package marker and templates**

`__init__.py`:

```python
"""Project-neutral Codex LO multiagent installer."""

INSTALLER_VERSION = 1
```

`claude-bridge.toml`:

```toml
name = "claude-bridge"
description = "Read-only bridge from Codex LO to Claude Code and its one-shot built-in Fable advisor."
sandbox_mode = "read-only"
developer_instructions = """
You are a bridge agent, not Claude Code and not the LO. Codex LO owns user communication, decomposition, sequencing, approval, conflict resolution, verification, and final reporting.

For an existing self-contained brief in the active workspace, invoke the installed call_external_agent.py claude <brief-path> <workspace-path>. For explicit advisor review, invoke call_external_agent.py claude-advisor <brief-path> <workspace-path>; call the built-in advisor exactly once in a fresh session.

If runtime permission is unavailable, return BLOCK with the exact command and evidence. Do not weaken the sandbox, do not contact the user, do not call another subagent, do not edit workspace files, and do not trust provider output without validation. Return exact files changed, command, validation, risks, and one next step to Codex LO.
"""
```

`agy-bridge.toml`:

```toml
name = "agy-bridge"
description = "Read-only bridge from Codex LO to AGY for bounded environment and operational-risk checks."
sandbox_mode = "read-only"
developer_instructions = """
You are a bridge agent, not AGY and not the LO. Codex LO owns user communication, decomposition, sequencing, approval, conflict resolution, verification, and final reporting.

For an existing self-contained brief in the active workspace, invoke the installed call_external_agent.py agy <brief-path> <workspace-path>.

If runtime permission is unavailable, return BLOCK with the exact command and evidence. Do not weaken the sandbox, do not contact the user, do not call another subagent, do not edit workspace files, and do not trust provider output without validation. Return exact files changed, command, validation, risks, and one next step to Codex LO.
"""
```

`claude-advisor-settings.json`:

```json
{
  "advisorModel": "fable"
}
```

- [ ] **Step 4: Run tests and verify GREEN**

Expected: all four template tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add tools/codex_global_multiagent tests/test_agents/test_global_codex_multiagent.py
git commit -m "test: lock global agent templates"
```

## Task 2: Build the no-temp provider adapter

**Files:**

- Create: `tools/codex_global_multiagent/templates/call_external_agent.py`
- Modify: `tests/test_agents/test_global_codex_multiagent.py`

- [ ] **Step 1: Add failing adapter tests**

Append helpers and tests:

```python
ADAPTER = TEMPLATES / "call_external_agent.py"


def write_executable(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def adapter_env(tmp_path: Path) -> dict[str, str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    mock = """#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
Path(os.environ['ARGV_RECORDER']).write_text(json.dumps(sys.argv[1:]), encoding='utf-8')
print(json.dumps({'status': 'ok', 'files_changed': 'none'}))
"""
    write_executable(bin_dir / "claude", mock)
    write_executable(bin_dir / "agy", mock)
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    env["ARGV_RECORDER"] = str(tmp_path / "argv.json")
    env["GLOBAL_AGENT_ADAPTER_TIMEOUT"] = "5"
    return env


def run_adapter(tmp_path: Path, provider: str, brief: Path, workspace: Path):
    return subprocess.run(
        [sys.executable, str(ADAPTER), provider, str(brief), str(workspace)],
        text=True,
        capture_output=True,
        check=False,
        env=adapter_env(tmp_path),
    )


def test_adapter_rejects_unknown_provider(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    brief = workspace / "brief.md"
    brief.write_text("review", encoding="utf-8")
    result = run_adapter(tmp_path, "opencode", brief, workspace)
    assert result.returncode == 2
    assert json.loads(result.stdout)["error"] == "provider_not_allowed"


def test_adapter_rejects_brief_escape(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    brief = tmp_path / "outside.md"
    brief.write_text("review", encoding="utf-8")
    result = run_adapter(tmp_path, "claude", brief, workspace)
    assert result.returncode == 2
    assert json.loads(result.stdout)["error"] == "brief_outside_workspace"


@pytest.mark.parametrize(
    ("provider", "required", "forbidden"),
    [
        ("claude", ["--permission-mode", "plan", "--tools", "Read,Grep,Glob"], ["--model"]),
        ("claude-advisor", ["--tools", "advisor", "--settings"], ["--model"]),
        ("agy", ["--sandbox", "--mode", "plan", "--add-dir"], ["--dangerously-skip-permissions"]),
    ],
)
def test_provider_argv_is_bounded(
    tmp_path: Path, provider: str, required: list[str], forbidden: list[str]
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    brief = workspace / "brief.md"
    brief.write_text("bounded review", encoding="utf-8")
    result = run_adapter(tmp_path, provider, brief, workspace)
    argv = json.loads((tmp_path / "argv.json").read_text(encoding="utf-8"))
    assert result.returncode == 0
    assert all(value in argv for value in required)
    assert all(value not in argv for value in forbidden)
    if provider == "claude-advisor":
        assert argv[1].count("exactly once") == 1


def test_adapter_uses_git_root(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    subprocess.run(["git", "init", "-q", str(workspace)], check=True)
    nested = workspace / "src"
    nested.mkdir()
    brief = workspace / "brief.md"
    brief.write_text("review", encoding="utf-8")
    result = run_adapter(tmp_path, "claude", brief, nested)
    assert json.loads(result.stdout)["workspace"] == str(workspace.resolve())
```

Add a timeout/redaction test that replaces the Claude mock with a process emitting `API_KEY=secret-value` and sleeping two seconds. With timeout `1`, assert exit 124, `status: timeout`, `[REDACTED]` present, the secret absent, and no capture file below the workspace.

- [ ] **Step 2: Run tests and verify RED**

Expected: adapter tests fail because the script is absent.

- [ ] **Step 3: Implement the adapter**

Implement these exact units:

```python
#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time


ALLOWED_PROVIDERS = {"claude", "claude-advisor", "agy"}
SECRET_PATTERNS = (
    re.compile(r"(?i)(Authorization\s*:\s*Bearer\s+)[^\s\",}]+"),
    re.compile(
        r'(?i)("?(?:AWS_SECRET_ACCESS_KEY|API_KEY|TOKEN|SECRET|PASSWORD)"?'
        r'\s*[=:]\s*"?)[^"\s,}]+'
    ),
    re.compile(r"(?i)sk-[A-Za-z0-9_-]+"),
)


def emit(payload: dict[str, object]) -> None:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def fail(error: str, code: int, **details: object) -> int:
    emit({"status": "error", "error": error, **details})
    return code


def bounded_timeout(env: dict[str, str]) -> int:
    raw = env.get("GLOBAL_AGENT_ADAPTER_TIMEOUT", "300")
    if not raw.isdecimal() or not 1 <= int(raw) <= 3600:
        raise ValueError("invalid_timeout")
    return int(raw)


def canonical_workspace(candidate: str) -> Path:
    workspace = Path(candidate).expanduser().resolve(strict=True)
    if not workspace.is_dir():
        raise ValueError("workspace_not_directory")
    try:
        probe = subprocess.run(
            ["git", "-C", str(workspace), "rev-parse", "--show-toplevel"],
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        return workspace
    if probe.returncode != 0:
        return workspace
    return Path(probe.stdout.strip()).resolve(strict=True)


def contained_brief(candidate: str, workspace: Path) -> Path:
    brief = Path(candidate).expanduser().resolve(strict=True)
    if not brief.is_file():
        raise ValueError("brief_not_file")
    try:
        brief.relative_to(workspace)
    except ValueError as exc:
        raise ValueError("brief_outside_workspace") from exc
    return brief


def redact(value: str) -> str:
    redacted = value
    for pattern in SECRET_PATTERNS:
        replacement = r"\1[REDACTED]" if pattern.groups else "[REDACTED]"
        redacted = pattern.sub(replacement, redacted)
    return redacted


def provider_command(provider: str, brief: str, workspace: Path) -> list[str]:
    if provider == "claude":
        return [
            "claude", "-p", brief, "--permission-mode", "plan",
            "--tools", "Read,Grep,Glob", "--output-format", "json",
            "--no-session-persistence",
        ]
    if provider == "claude-advisor":
        prompt = (
            "Call the built-in advisor exactly once before any other tool. "
            "Do not call it again. Use its feedback to answer this brief.\n\n"
            + brief
        )
        return [
            "claude", "-p", prompt, "--permission-mode", "plan",
            "--tools", "advisor", "--settings",
            str(Path(__file__).resolve().parent / "claude-advisor-settings.json"),
            "--output-format", "stream-json", "--verbose",
            "--no-session-persistence",
        ]
    timeout = bounded_timeout(os.environ)
    return [
        "agy", "--sandbox", "--mode", "plan", "--add-dir", str(workspace),
        "--print", brief, "--print-timeout", f"{timeout}s",
    ]


def text_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        return fail("invalid_arguments", 2)
    provider, brief_arg, workspace_arg = argv[1:]
    if provider not in ALLOWED_PROVIDERS:
        return fail("provider_not_allowed", 2)
    try:
        timeout = bounded_timeout(os.environ)
        workspace = canonical_workspace(workspace_arg)
        brief_path = contained_brief(brief_arg, workspace)
    except FileNotFoundError:
        return fail("path_not_found", 2)
    except ValueError as exc:
        return fail(str(exc), 2)
    executable = "claude" if provider.startswith("claude") else "agy"
    if shutil.which(executable) is None:
        return fail("cli_not_found", 3, cli=executable)
    brief = brief_path.read_text(encoding="utf-8")
    if not brief.strip():
        return fail("brief_empty", 2)
    command = provider_command(provider, brief, workspace)
    started = time.monotonic()
    try:
        result = subprocess.run(
            command,
            cwd=workspace,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        code = result.returncode
        status = "ok" if code == 0 else "error"
        stdout = result.stdout
        stderr = result.stderr
    except subprocess.TimeoutExpired as exc:
        code = 124
        status = "timeout"
        stdout = text_output(exc.stdout)
        stderr = text_output(exc.stderr)
    emit({
        "status": status,
        "provider": provider,
        "brief": str(brief_path),
        "workspace": str(workspace),
        "exit_code": code,
        "duration_seconds": int(time.monotonic() - started),
        "stdout": stdout,
        "stderr_sanitized": redact(stderr),
    })
    return code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
```

Concrete requirements:

- Require provider, brief path, and workspace path only.
- Accept timeout integers 1 through 3600.
- Resolve Git root with `git -C <candidate> rev-parse --show-toplevel`; otherwise use the canonical directory.
- Resolve the brief with `strict=True`, require a regular file, and require `brief.relative_to(workspace)`.
- Use the shown `subprocess.run` call and argument arrays; never `shell=True`.
- Claude worker argv is `claude -p <brief> --permission-mode plan --tools Read,Grep,Glob --output-format json --no-session-persistence`.
- Advisor mode allows only `advisor`, passes the sibling settings JSON, uses stream JSON and no persistence, requests exactly one advisor call, and never passes `--model`.
- AGY argv is `agy --sandbox --mode plan --add-dir <workspace> --print <brief> --print-timeout <seconds>s`.
- A timeout emits exit 124 and `status: timeout`.
- Redact bearer values, `AWS_SECRET_ACCESS_KEY`, `API_KEY`, `TOKEN`, `SECRET`, `PASSWORD`, and `sk-*` from stderr.
- Emit one JSON object with `status`, `provider`, `brief`, `workspace`, `exit_code`, `duration_seconds`, `stdout`, and `stderr_sanitized`.

- [ ] **Step 4: Run tests and verify GREEN**

Expected: all template and adapter tests pass.

- [ ] **Step 5: Parse and commit**

```bash
/mnt/c/model/mode_11_hana/.venv_wsl/bin/python -m py_compile tools/codex_global_multiagent/templates/call_external_agent.py
git add tools/codex_global_multiagent/templates/call_external_agent.py tests/test_agents/test_global_codex_multiagent.py
git commit -m "feat: add global agent adapter"
```

## Task 3: Build config rendering and the install transaction

**Files:**

- Create: `tools/codex_global_multiagent/install.py`
- Modify: `tests/test_agents/test_global_codex_multiagent.py`

- [ ] **Step 1: Add failing renderer tests**

Append:

```python
from tools.codex_global_multiagent.install import InstallConflict, install, render_global_config


def existing_config() -> str:
    return '''model = "gpt-test"
model_reasoning_effort = "high"

[projects."/work"]
trust_level = "trusted"
'''


def test_render_preserves_existing_settings(tmp_path: Path) -> None:
    home = tmp_path / ".codex"
    rendered = render_global_config(existing_config(), home)
    parsed = tomllib.loads(rendered)
    assert parsed["model"] == "gpt-test"
    assert parsed["projects"]["/work"]["trust_level"] == "trusted"
    assert parsed["features"]["multi_agent"] is True
    assert parsed["agents"]["max_depth"] == 1
    assert parsed["agents"]["claude-bridge"]["config_file"] == str(
        home / "agents/claude-bridge.toml"
    )


def test_render_is_idempotent(tmp_path: Path) -> None:
    home = tmp_path / ".codex"
    once = render_global_config(existing_config(), home)
    assert render_global_config(once, home) == once


def test_render_rejects_foreign_agent_key(tmp_path: Path) -> None:
    source = existing_config() + '''
[agents.claude-bridge]
description = "foreign"
config_file = "/other/claude.toml"
'''
    with pytest.raises(InstallConflict, match="agents.claude-bridge"):
        render_global_config(source, tmp_path / ".codex")
```

- [ ] **Step 2: Run renderer tests and verify RED**

Expected: import failure because `install.py` is absent.

- [ ] **Step 3: Implement deterministic rendering**

Create `install.py` with these units:

```python
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib

from tools.codex_global_multiagent import INSTALLER_VERSION


BEGIN = "# BEGIN CODEX GLOBAL MULTIAGENT (managed)"
END = "# END CODEX GLOBAL MULTIAGENT (managed)"
PACKAGE_ROOT = Path(__file__).resolve().parent
TEMPLATES = PACKAGE_ROOT / "templates"


class InstallConflict(RuntimeError):
    pass


def managed_block(codex_home: Path) -> str:
    agents = codex_home / "agents"
    claude_description = (
        "Delegate bounded requirements, architecture, logical QA, or one-shot "
        "advisor work to Claude Code."
    )
    agy_description = (
        "Delegate bounded environment, deployment, dependency, and "
        "operational-risk checks to AGY."
    )
    return "\n".join([
        BEGIN,
        "[features]",
        "multi_agent = true",
        "",
        "[agents]",
        "max_depth = 1",
        "interrupt_message = true",
        "",
        "[agents.claude-bridge]",
        f"description = {json.dumps(claude_description)}",
        f"config_file = {json.dumps(str(agents / 'claude-bridge.toml'))}",
        'nickname_candidates = ["ClaudeBridge"]',
        "",
        "[agents.agy-bridge]",
        f"description = {json.dumps(agy_description)}",
        f"config_file = {json.dumps(str(agents / 'agy-bridge.toml'))}",
        'nickname_candidates = ["AgyBridge"]',
        END,
    ])


def remove_managed_block(source: str) -> str:
    pattern = re.compile(
        rf"(?:^|\n){re.escape(BEGIN)}\n.*?\n{re.escape(END)}(?:\n|$)",
        re.DOTALL,
    )
    return pattern.sub("\n", source).rstrip()


def reject_conflicts(parsed: dict[str, object]) -> None:
    features = parsed.get("features", {})
    agents = parsed.get("agents", {})
    if isinstance(features, dict) and "multi_agent" in features:
        raise InstallConflict("features.multi_agent already exists outside managed block")
    if isinstance(agents, dict):
        for key in ("max_depth", "interrupt_message", "claude-bridge", "agy-bridge"):
            if key in agents:
                raise InstallConflict(f"agents.{key} already exists outside managed block")


def render_global_config(source: str, codex_home: Path) -> str:
    unmanaged = remove_managed_block(source)
    parsed = tomllib.loads(unmanaged) if unmanaged.strip() else {}
    reject_conflicts(parsed)
    prefix = f"{unmanaged}\n\n" if unmanaged else ""
    rendered = f"{prefix}{managed_block(codex_home)}\n"
    tomllib.loads(rendered)
    return rendered
```

`managed_block()` renders exactly `[features]`, `[agents]`, `[agents.claude-bridge]`, and `[agents.agy-bridge]` with absolute config paths. `remove_managed_block()` removes only marker-bounded text. `reject_conflicts()` rejects the five managed keys outside the block. `render_global_config()` parses before and after, preserves unmanaged text, and returns one final newline.

- [ ] **Step 4: Add failing transaction tests**

Append:

```python
def seed_home(tmp_path: Path) -> Path:
    home = tmp_path / ".codex"
    home.mkdir()
    (home / "config.toml").write_text(existing_config(), encoding="utf-8")
    return home


def test_install_writes_manifest_and_preserves_config(tmp_path: Path) -> None:
    home = seed_home(tmp_path)
    result = install(home, tmp_path / "backups")
    assert result.status == "installed"
    assert tomllib.loads((home / "config.toml").read_text())["model"] == "gpt-test"
    manifest = json.loads((home / "multiagent/manifest.json").read_text())
    assert manifest["installer_version"] == 1
    assert manifest["backup_dir"] == result.backup_dir
    assert len(manifest["hashes"]) == 5


def test_install_is_idempotent(tmp_path: Path) -> None:
    home = seed_home(tmp_path)
    assert install(home, tmp_path / "backups").status == "installed"
    assert install(home, tmp_path / "backups").status == "unchanged"


def test_install_refuses_changed_managed_file(tmp_path: Path) -> None:
    home = seed_home(tmp_path)
    install(home, tmp_path / "backups")
    bridge = home / "agents/claude-bridge.toml"
    bridge.write_text(bridge.read_text() + "\n# user edit\n")
    with pytest.raises(InstallConflict, match="managed file changed"):
        install(home, tmp_path / "backups")


@pytest.mark.parametrize("fail_after", [1, 2, 3, 4, 5, 6])
def test_install_rolls_back_every_replacement(tmp_path: Path, fail_after: int) -> None:
    home = seed_home(tmp_path)
    before = {p.relative_to(home): p.read_bytes() for p in home.rglob("*") if p.is_file()}
    with pytest.raises(RuntimeError, match="injected install failure"):
        install(home, tmp_path / "backups", fail_after=fail_after)
    after = {p.relative_to(home): p.read_bytes() for p in home.rglob("*") if p.is_file()}
    assert after == before
```

- [ ] **Step 5: Run transaction tests and verify RED**

Expected: `install()` missing or signature failures.

- [ ] **Step 6: Implement the atomic transaction**

Implement these interfaces:

```python
MANAGED_FILES = (
    ("claude-bridge.toml", Path("agents/claude-bridge.toml")),
    ("agy-bridge.toml", Path("agents/agy-bridge.toml")),
    ("call_external_agent.py", Path("multiagent/call_external_agent.py")),
    ("claude-advisor-settings.json", Path("multiagent/claude-advisor-settings.json")),
)


@dataclass(frozen=True)
class InstallResult:
    status: str
    codex_home: str
    backup_dir: str | None
    installed: tuple[str, ...]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def proposed_files(codex_home: Path, current_config: str) -> dict[Path, bytes]:
    adapter_path = str(codex_home / "multiagent" / "call_external_agent.py")
    files = {
        codex_home / "config.toml": render_global_config(
            current_config, codex_home
        ).encode("utf-8")
    }
    for template_name, relative in MANAGED_FILES:
        content = (TEMPLATES / template_name).read_bytes()
        if template_name.endswith("bridge.toml"):
            content = content.replace(b"call_external_agent.py", adapter_path.encode())
        files[codex_home / relative] = content
    return files


def validate_proposed(files: dict[Path, bytes]) -> None:
    for path, content in files.items():
        if path.suffix == ".toml":
            tomllib.loads(content.decode("utf-8"))
        elif path.suffix == ".json":
            json.loads(content.decode("utf-8"))
        elif path.suffix == ".py":
            compile(content.decode("utf-8"), str(path), "exec")


def atomic_write(path: Path, content: bytes, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if executable:
            temporary_path.chmod(0o700)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def read_manifest(codex_home: Path) -> dict[str, object] | None:
    path = codex_home / "multiagent" / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def assert_managed_files_unchanged(
    codex_home: Path, manifest: dict[str, object]
) -> None:
    hashes = manifest.get("hashes")
    if not isinstance(hashes, dict):
        raise InstallConflict("malformed manifest hashes")
    for relative, expected in hashes.items():
        if relative == "config.toml":
            continue
        path = codex_home / str(relative)
        if not path.is_file() or sha256(path) != expected:
            raise InstallConflict(f"managed file changed: {relative}")


def install(
    codex_home: Path,
    backup_root: Path,
    *,
    fail_after: int | None = None,
) -> InstallResult:
    codex_home = codex_home.expanduser().resolve()
    codex_home.mkdir(parents=True, exist_ok=True)
    backup_root.mkdir(parents=True, exist_ok=True)
    config_path = codex_home / "config.toml"
    current = config_path.read_text(encoding="utf-8") if config_path.exists() else ""
    old_manifest = read_manifest(codex_home)
    if old_manifest is not None:
        assert_managed_files_unchanged(codex_home, old_manifest)
    files = proposed_files(codex_home, current)
    validate_proposed(files)
    if old_manifest is not None and all(
        path.is_file() and path.read_bytes() == content
        for path, content in files.items()
    ):
        return InstallResult("unchanged", str(codex_home), None, tuple())

    backup_dir = Path(
        tempfile.mkdtemp(prefix="codex-global-multiagent.", dir=backup_root)
    )
    backup_dir.chmod(0o700)
    manifest_path = codex_home / "multiagent" / "manifest.json"
    destinations = [*files, manifest_path]
    originals = {
        path: path.read_bytes() if path.exists() else None for path in destinations
    }
    for path, original in originals.items():
        if original is None:
            continue
        backup_path = backup_dir / path.relative_to(codex_home)
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        backup_path.write_bytes(original)

    replaced: list[Path] = []
    try:
        for index, (path, content) in enumerate(files.items(), start=1):
            atomic_write(
                path,
                content,
                executable=path.name == "call_external_agent.py",
            )
            replaced.append(path)
            if fail_after == index:
                raise RuntimeError("injected install failure")
        hashes = {
            str(path.relative_to(codex_home)): sha256(path) for path in files
        }
        manifest = json.dumps(
            {
                "installer_version": INSTALLER_VERSION,
                "backup_dir": str(backup_dir),
                "hashes": hashes,
            },
            indent=2,
            sort_keys=True,
        ).encode("utf-8") + b"\n"
        atomic_write(manifest_path, manifest)
        replaced.append(manifest_path)
        if fail_after == len(files) + 1:
            raise RuntimeError("injected install failure")
    except Exception:
        for path in reversed(replaced):
            original = originals[path]
            if original is None:
                path.unlink(missing_ok=True)
            else:
                atomic_write(
                    path,
                    original,
                    executable=path.name == "call_external_agent.py",
                )
        raise
    return InstallResult(
        "installed",
        str(codex_home),
        str(backup_dir),
        tuple(str(path) for path in destinations),
    )
```

The transaction must:

1. Parse all proposed TOML and JSON before touching destinations.
2. Check manifest hashes before replacing any prior managed file.
3. Create a mode-restricted backup under `backup_root`.
4. Back up every existing destination.
5. Write sibling temporary files, flush and `fsync`, then use `os.replace`.
6. Write `manifest.json` last with installer version, backup path, and five managed hashes.
7. On any exception, restore originals in reverse order and remove only newly created files.
8. Return `unchanged` without a new backup when every managed byte already matches.

- [ ] **Step 7: Run tests and verify GREEN**

Expected: renderer, install, idempotence, conflict, and all injected rollback tests pass.

- [ ] **Step 8: Commit Task 3**

```bash
git add tools/codex_global_multiagent/install.py tests/test_agents/test_global_codex_multiagent.py
git commit -m "feat: install global agents atomically"
```

## Task 4: Add isolated Codex CLI validation

**Files:**

- Modify: `tools/codex_global_multiagent/install.py`
- Modify: `tests/test_agents/test_global_codex_multiagent.py`

- [ ] **Step 1: Add failing check/probe tests**

Append tests that:

- invoke `install.py --check --codex-home <fixture> --backup-root <tmp>`;
- assert `check_ok` while the requested home remains byte-identical;
- call `run_codex_probe(fixture_home, neutral_dir)`;
- assert strict config loaded, `multi_agent` is true, and agent names are exactly `agy-bridge` and `claude-bridge`;
- create a neutral Git project with a local `.codex/config.toml` redefining the same agents and assert the effective paths are local without duplicate names.

Use the real local Codex binary only for model-free `features`, `doctor`, or `debug prompt-input` calls. No model request is allowed.

- [ ] **Step 2: Run check/probe tests and verify RED**

Expected: missing CLI and probe functions.

- [ ] **Step 3: Implement the model-free probe and CLI**

Add:

```python
def run_codex_probe(codex_home: Path, cwd: Path) -> dict[str, object]:
    env = os.environ.copy()
    env["CODEX_HOME"] = str(codex_home)
    features = subprocess.run(
        ["codex", "features", "list"],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    doctor = subprocess.run(
        ["codex", "--strict-config", "doctor", "--summary", "--ascii"],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    config = tomllib.loads(
        (codex_home / "config.toml").read_text(encoding="utf-8")
    )
    agents = config.get("agents", {})
    agent_names = sorted(
        key for key, value in agents.items() if isinstance(value, dict)
    ) if isinstance(agents, dict) else []
    doctor_output = doctor.stdout + doctor.stderr
    multi_agent = any(
        len(parts := line.split()) >= 3
        and parts[0] == "multi_agent"
        and parts[2] == "true"
        for line in features.stdout.splitlines()
    )
    return {
        "config_loaded": (
            "config loaded" in doctor_output.casefold()
            and "unknown key" not in doctor_output.casefold()
            and "parse error" not in doctor_output.casefold()
        ),
        "multi_agent": multi_agent,
        "agent_names": agent_names,
        "features_exit": features.returncode,
        "doctor_exit": doctor.returncode,
        "doctor_output": doctor_output,
    }


def isolated_check(codex_home: Path, backup_root: Path) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="codex-global-check.") as temporary:
        root = Path(temporary)
        fixture = root / ".codex"
        fixture.mkdir()
        source = codex_home / "config.toml"
        if source.exists():
            (fixture / "config.toml").write_bytes(source.read_bytes())
        result = install(fixture, root / "backups")
        neutral = root / "neutral"
        neutral.mkdir()
        probe = run_codex_probe(fixture, neutral)
        if not probe["config_loaded"] or not probe["multi_agent"]:
            raise RuntimeError(json.dumps(probe, ensure_ascii=False))
        if probe["agent_names"] != ["agy-bridge", "claude-bridge"]:
            raise RuntimeError(json.dumps(probe, ensure_ascii=False))
        return {"status": "check_ok", "install": result.status, "probe": probe}


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--codex-home", type=Path, default=Path.home() / ".codex"
    )
    parser.add_argument(
        "--backup-root",
        type=Path,
        default=Path(os.environ.get("TMPDIR", "/tmp")),
    )
    parser.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    try:
        if args.check:
            payload = isolated_check(args.codex_home, args.backup_root)
        else:
            payload = asdict(install(args.codex_home, args.backup_root))
    except (InstallConflict, RuntimeError, ValueError, tomllib.TOMLDecodeError) as exc:
        print(json.dumps({
            "status": "error",
            "error": type(exc).__name__,
            "message": str(exc),
        }))
        return 2
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

`run_codex_probe()` sets `CODEX_HOME` and runs:

```text
codex features list
codex --strict-config doctor --summary --ascii
```

It accepts unrelated provider/state diagnostics only when the output explicitly confirms config loaded and has no parse or unknown-key failure. `isolated_check()` copies only `config.toml` to a temporary home, installs there, probes a neutral directory, and deletes the fixture. CLI supports only `--codex-home`, `--backup-root`, and `--check`; it has no force option.

- [ ] **Step 4: Verify actual Codex output and local override**

```bash
/mnt/c/model/mode_11_hana/.venv_wsl/bin/python tools/codex_global_multiagent/install.py --codex-home /tmp/codex-global-fixture-home --backup-root /tmp --check
```

Expected: `check_ok`, global multi-agent true, two global agent names, and successful project override. If the installed Codex cannot demonstrate same-key local override, stop and revise the design rather than registering duplicate agents.

- [ ] **Step 5: Run all global tests and commit**

```bash
/mnt/c/model/mode_11_hana/.venv_wsl/bin/python -m pytest tests/test_agents/test_global_codex_multiagent.py -v
git add tools/codex_global_multiagent/install.py tests/test_agents/test_global_codex_multiagent.py
git commit -m "test: verify global Codex loading"
```

## Task 5: Document operation and recovery

**Files:**

- Modify: `README.md`
- Modify: `tests/test_agents/test_global_codex_multiagent.py`

- [ ] **Step 1: Add a failing README contract test**

```python
def test_readme_documents_global_start_verify_and_backup() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for required in (
        "tools/codex_global_multiagent/install.py --check",
        "cd /path/to/any/project",
        "codex",
        "new Codex session",
        "manifest.json",
        "backup_dir",
        "MODE_11_hana",
    ):
        assert required in readme
```

- [ ] **Step 2: Run the documentation test and verify RED**

Expected: README contract failure.

- [ ] **Step 3: Add the README section**

Document these exact commands:

```bash
/mnt/c/model/mode_11_hana/.venv_wsl/bin/python tools/codex_global_multiagent/install.py --check
/mnt/c/model/mode_11_hana/.venv_wsl/bin/python tools/codex_global_multiagent/install.py
cd /path/to/any/project
codex
```

Explain that new session means exit and relaunch, dispatch is demand-driven, MODE_11_hana local policy remains stronger, and `manifest.json` records `backup_dir`. Do not document an automated uninstall because this scope does not implement one.

- [ ] **Step 4: Run test and commit**

```bash
/mnt/c/model/mode_11_hana/.venv_wsl/bin/python -m pytest tests/test_agents/test_global_codex_multiagent.py -v
git add README.md tests/test_agents/test_global_codex_multiagent.py
git commit -m "docs: explain global Codex agents"
```

## Task 6: Install to the real user Codex home

**Files outside repository:** the six managed `~/.codex` destinations listed in the file map.

- [ ] **Step 1: Run isolated check against the real config**

```bash
/mnt/c/model/mode_11_hana/.venv_wsl/bin/python tools/codex_global_multiagent/install.py --codex-home /home/ptg/.codex --backup-root /tmp --check
```

Expected: `check_ok` and no real-home byte changes.

- [ ] **Step 2: Record hashes or absence for the six destinations**

Never read or print `auth.json`, history, databases, tokens, or credentials.

- [ ] **Step 3: Request escalated approval and install once**

```bash
/mnt/c/model/mode_11_hana/.venv_wsl/bin/python tools/codex_global_multiagent/install.py --codex-home /home/ptg/.codex --backup-root /tmp
```

Expected: `installed`, a `/tmp/codex-global-multiagent.*` backup, and the six managed destinations. Never use `sudo`.

- [ ] **Step 4: Run installer again**

Expected: `unchanged`, no additional backup, identical hashes.

- [ ] **Step 5: Validate neutral Git and non-Git directories**

From disposable directories under `/tmp`, run model-free features and strict doctor checks. Expected: multi-agent true, config loaded, and global agent paths below `/home/ptg/.codex/agents/`.

- [ ] **Step 6: Validate MODE_11_hana specialization**

From this repository, assert exactly one Claude and one AGY bridge, both using repository-local profiles, with all local hard gates intact.

- [ ] **Step 7: Verify protected and user state**

```bash
/mnt/c/model/mode_11_hana/.venv_wsl/bin/python .agents/adapters/protected_artifact_guard.py verify --root . --state /tmp/global-codex-preflight-protected.json
git status --short --untracked-files=all
git diff --name-only -- packages_win/py312 mlruns out '*.parquet' '*.bat'
```

Expected: protected verification passes, protected/BAT diff is empty, and known user changes remain.

## Task 7: Final regression and cross-family QA

- [ ] **Step 1: Run focused plus existing orchestration tests**

```bash
/mnt/c/model/mode_11_hana/.venv_wsl/bin/python -m pytest tests/test_agents/test_global_codex_multiagent.py tests/test_agents/test_codex_lo_orchestration.py -v
```

Expected: all tests pass under Python 3.12.

- [ ] **Step 2: Run parsers and diff checks**

```bash
/mnt/c/model/mode_11_hana/.venv_wsl/bin/python -m py_compile tools/codex_global_multiagent/install.py tools/codex_global_multiagent/templates/call_external_agent.py
git diff --check
```

Expected: exit 0.

- [ ] **Step 3: Request Claude Code read-only final QA**

Limit the brief to the installer, templates, tests, README diff, design, and plan. Require exact files inspected, findings with file:line, validation, risk, and one next step. Claude must not edit or contact another worker.

- [ ] **Step 4: Use Fable advisor once for an evidence-only finish challenge**

Use a fresh no-persistence Claude session and the installed dedicated Fable settings. Invoke the built-in advisor exactly once with the verified evidence packet; do not ask advisor mode to read files.

- [ ] **Step 5: Apply only verified findings and rerun Steps 1–2**

Document technically invalid or out-of-scope suggestions instead of applying them.

- [ ] **Step 6: Commit only authorized repository files**

Confirm staged paths exclude `.understand-anything/**`, the unrelated prepush plan, protected paths, BAT, and artifacts.

```bash
git commit -m "feat: install global Codex multiagent"
```

- [ ] **Step 7: Finish the isolated branch**

Do not push. Use the finishing-branch workflow and preserve all pre-existing user changes.

## Final handoff

Report exact repository files, global destinations and hashes, manifest backup directory, test commands/counts, neutral Git/non-Git results, MODE_11_hana override result, live-provider status, protected/user state, commit status, and remaining provider-auth/network/raw-stdout risks. The start command is:

```bash
cd /path/to/project
codex
```
