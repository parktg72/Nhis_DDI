# Codex LO Multiagent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Codex the only MODE_11_hana LO, expose Claude Code and AGY as bounded external subagents, configure Claude Code's built-in advisor to Fable 5, and remove OpenCode from every active orchestration surface.

**Architecture:** Project-scoped Codex custom agents act as thin bridge agents and call one allowlisted shell adapter. The adapter accepts only repository brief files, runs Claude Code/Claude advisor/AGY in read-only modes, and returns a JSON envelope; Codex LO validates all evidence and remains the only user-facing decision maker. Existing HANA, freeze, protected-path, Python 3.12, and BAT gates remain unchanged.

**Tech Stack:** Codex CLI project configuration (TOML), Claude Code CLI 2.1.208 built-in advisor, AGY CLI 1.1.1, Bash, `jq`, GNU `timeout`, Python 3.12 `tomllib`, pytest.

> [!IMPORTANT]
> **Execution status (2026-07-14):** Tasks 1–5 are implemented and validated. Direct Claude, Fable advisor, and AGY adapter smokes passed; Claude file-level final QA returned PASS with no blocking findings; and the separate evidence-only Fable finish advisory returned PASS. The bridge-mediated live path was not directly exercised and remains approval-gated. The exact initial snippets below are historical TDD scaffolding and must not overwrite the hardened current implementation.
>
> Hardened final deviations from the initial scaffolding are: timeout provenance, stderr redaction, and temporary-directory safety in the adapter; a metadata-only protected-artifact snapshot/verify guard; a narrow `.gitignore` allowlist for the approved orchestration files; a tracked minimal `.claude/settings.json` with ignored migration-only machine permissions; a 60-test orchestration suite; and separate strict Codex configuration inspection plus feature-state validation.

---

## Scope and execution constraints

- Approved design: `docs/superpowers/specs/2026-07-14-codex-lo-multiagent-design.md`.
- Preserve unrelated dirty files, especially `.understand-anything/**` and `docs/superpowers/plans/2026-07-13-prepush-contract-repair.md`.
- Do not edit `.bat`, `packages_win/py312/`, `mlruns/`, generated parquet, or `out/`.
- Do not run ETL, feature build, training, or any frozen-holdout command. This plan does not require the HANA temp-disk preflight.
- Use `/mnt/c/model/mode_11_hana/.venv_wsl/bin/python`; stop with BLOCK if it is not Python 3.12.
- Do not commit unless the user separately requests a commit. Commit commands below are handoff instructions only and must otherwise be recorded as skipped.
- External Claude/AGY live calls consume quota and may require home/socket access outside the sandbox. Run each only through an explicit approval prompt.
- Both custom bridges remain strict read-only. External adapter execution requires Codex LO/user-approved runtime permission; if unavailable, a bridge must not self-escalate, contact the user, or weaken its sandbox, and must return `BLOCK` to Codex LO with the exact attempted command and runtime evidence.
- A local Codex strict `:read-only` probe blocked `mktemp /tmp` as expected fail-closed behavior. Parent approval/runtime permission is therefore a prerequisite for bridge-mediated live adapter dispatch; direct adapter smoke success does not prove that bridge path.
- Adapter stdout is intentionally raw; only stderr receives limited secret-like masking. The protected-artifact guard compares metadata only and does not hash or prove file contents, so its result must be paired with the configured Git-surface validator.

## File map

### Create

- `.codex/config.toml` — project-scoped Codex multi-agent registration and depth limit.
- `.codex/agents/claude-bridge.toml` — Codex custom bridge for Claude worker and Fable advisor modes.
- `.codex/agents/agy-bridge.toml` — Codex custom bridge for AGY environment/risk work.
- `.agents/adapters/call_external_agent.sh` — single allowlisted external CLI entrypoint and JSON envelope producer.
- `.agents/adapters/protected_artifact_guard.py` — metadata-only baseline/verification guard for ignored protected artifacts and BAT files.
- `.claude/agents/.gitkeep` — retain the now-empty approved Claude-agent directory.
- `.claude/settings.json` — versioned minimal shared configuration containing only `advisorModel: "fable"`.
- `tests/test_agents/__init__.py` — test package marker.
- `tests/test_agents/fixtures/smoke_brief.md` — fixed read-only smoke brief used by fake and live tests.
- `tests/test_agents/test_codex_lo_orchestration.py` — hardened 60-test config, adapter, policy, and protected-surface regression suite.
- `docs/superpowers/specs/2026-07-14-codex-lo-multiagent-design.md` — approved current orchestration design.
- `docs/superpowers/plans/2026-07-14-codex-lo-multiagent.md` — implementation and validation record.

### Modify

- `.gitignore` — narrowly allowlist the approved orchestration adapters and shared Claude settings while explicitly retaining local Claude settings as ignored.
- `AGENTS.md` — make Codex the only LO and limit external subagents to Claude Code and AGY.
- `CLAUDE.md` — describe Claude as a Codex worker, remove OpenCode policy, and point to Codex project settings.
- `.agents/agents_config.json` — replace legacy HQ/MCP/OpenCode mapping with the actual Codex LO + CLI bridge mapping.
- `.agents/codex_hq.md` — convert from worker/Hermes-return language to Codex LO authority and verification rules.
- `.agents/claude_hq.md` — return evidence to Codex LO and document Fable one-shot advisor behavior.
- `.agents/agy_hq.md` — return environment/risk evidence to Codex LO.
- `.agents/message_deferral_guide.md` — remove Hermes/full-HQ wording and make sequential dispatch a Codex LO rule.
- `docs/superpowers/specs/2026-07-12-opencode-lo-contract-design.md` — add a superseded notice only; retain historical provenance.
- `docs/superpowers/specs/contracts/profile_contracts.md` — clarify current frozen-contract authority while retaining the 2026-07-12 design as historical provenance.

### Local migration only — ignored, never a deliverable

- `.claude/settings.local.json` — preserves the pre-existing machine-local `permissions` object exactly in local scope; it must never be staged, committed, or reviewed as a shared deliverable, and a fresh checkout must not require it.

### Delete

- `.agents/opencode_hq.md` — active OpenCode HQ definition.
- `.claude/agents/opencode-worker.md` — active OpenCode worker definition.
- `.claude/agents/agy-bridge.md` — legacy Claude-Agent/MCP bridge superseded by the Codex custom bridge.
- `.claude/agents/antigravity-worker.md` — legacy local Claude-model worker; AGY CLI is now the only AGY provider.
- `.claude/agents/claude-bridge.md` — legacy same-family MCP bridge superseded by the Codex custom bridge.
- `.claude/agents/codex-bridge.md` — Codex is LO and no longer needs a Claude-side Codex bridge.
- `.claude/agents/codex-worker.md` — Codex performs implementation directly as LO.
- `.claude/agents/hermes-worker.md` — Hermes is not part of the approved agent pool.

## Task 1: Lock the Codex project configuration and Fable advisor setting

**Files:**

- Create: `.codex/config.toml`
- Create: `.codex/agents/claude-bridge.toml`
- Create: `.codex/agents/agy-bridge.toml`
- Create: `tests/test_agents/__init__.py`
- Create: `tests/test_agents/test_codex_lo_orchestration.py`
- Create: `.claude/settings.json`
- Local migration only: `.claude/settings.local.json` (ignored; never stage or commit)

- [x] **Step 1: Verify the required runtime before running pytest**

Run:

```bash
/mnt/c/model/mode_11_hana/.venv_wsl/bin/python --version
```

Expected: `Python 3.12.x`. If the command is missing or reports any other minor version, stop this task with the repository's Python 3.12 BLOCK.

- [x] **Step 2: Write the failing project-configuration tests**

Create `tests/test_agents/__init__.py` as an empty file. Create `tests/test_agents/test_codex_lo_orchestration.py` with this initial content:

```python
from __future__ import annotations

import json
import os
import subprocess
import textwrap
import tomllib
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
CODEX_CONFIG = ROOT / ".codex" / "config.toml"
CLAUDE_AGENT = ROOT / ".codex" / "agents" / "claude-bridge.toml"
AGY_AGENT = ROOT / ".codex" / "agents" / "agy-bridge.toml"
CLAUDE_SETTINGS = ROOT / ".claude" / "settings.json"
ADAPTER = ROOT / ".agents" / "adapters" / "call_external_agent.sh"
SMOKE_BRIEF = ROOT / "tests" / "test_agents" / "fixtures" / "smoke_brief.md"


def load_toml(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def test_codex_project_registers_only_approved_external_bridges() -> None:
    config = load_toml(CODEX_CONFIG)
    assert config["features"]["multi_agent"] is True
    assert config["agents"]["max_depth"] == 1
    roles = {
        key
        for key, value in config["agents"].items()
        if isinstance(value, dict)
    }
    assert roles == {"claude-bridge", "agy-bridge"}
    assert config["agents"]["claude-bridge"]["config_file"] == (
        "./agents/claude-bridge.toml"
    )
    assert config["agents"]["agy-bridge"]["config_file"] == (
        "./agents/agy-bridge.toml"
    )


@pytest.mark.parametrize(
    ("path", "expected_name"),
    [(CLAUDE_AGENT, "claude-bridge"), (AGY_AGENT, "agy-bridge")],
)
def test_custom_bridge_agent_schema(path: Path, expected_name: str) -> None:
    agent = load_toml(path)
    assert agent["name"] == expected_name
    assert agent["description"].strip()
    assert agent["developer_instructions"].strip()
    assert agent["sandbox_mode"] == "read-only"
    assert "call_external_agent.sh" in agent["developer_instructions"]
    assert "Codex LO" in agent["developer_instructions"]
    assert "user" in agent["developer_instructions"].lower()


def test_claude_code_uses_fable_as_builtin_advisor() -> None:
    serialized = CLAUDE_SETTINGS.read_text(encoding="utf-8")
    settings = json.loads(serialized)
    assert settings == {"advisorModel": "fable"}
    assert "permissions" not in serialized
    assert "bypassPermissions" not in serialized
```

- [x] **Step 3: Run the focused tests and confirm the expected failure**

Run:

```bash
/mnt/c/model/mode_11_hana/.venv_wsl/bin/python -m pytest tests/test_agents/test_codex_lo_orchestration.py -k 'project or custom_bridge or fable' -v
```

Expected: FAIL because `.codex/config.toml` and the custom-agent TOML files do not exist and `.claude/settings.json` has no `advisorModel`.

- [x] **Step 4: Add the minimal Codex project configuration**

Create `.codex/config.toml`:

```toml
[features]
multi_agent = true

[agents]
max_depth = 1
interrupt_message = true

[agents.claude-bridge]
description = "Delegate bounded requirements, architecture, logical QA, or Fable advisor work to Claude Code."
config_file = "./agents/claude-bridge.toml"
nickname_candidates = ["ClaudeBridge"]

[agents.agy-bridge]
description = "Delegate bounded environment, deployment, Python 3.12, BAT, disk, and risk checks to AGY."
config_file = "./agents/agy-bridge.toml"
nickname_candidates = ["AgyBridge"]
```

Create `.codex/agents/claude-bridge.toml`:

```toml
name = "claude-bridge"
description = "Read-only bridge from Codex LO to Claude Code for requirements, architecture, logical QA, and one-shot built-in Fable advisor work."
sandbox_mode = "read-only"
developer_instructions = """
You are a bridge agent, not Claude itself and not the LO. Codex LO owns user communication, sequencing, decisions, verification, and final reporting.

For a self-contained brief file inside /mnt/c/model/mode_11_hana, invoke only:
  .agents/adapters/call_external_agent.sh claude <brief-path>
or, when Codex LO explicitly requests built-in advisor review:
  .agents/adapters/call_external_agent.sh claude-advisor <brief-path>

Do not call another subagent, contact the user, edit repository files, or claim success from model output alone. Validate the adapter JSON envelope and return: exact files changed (normally none), exact command, validation status, risks, and one recommended next step to Codex LO. Preserve all MODE_11_hana freeze, protected-path, HANA-schema, Python 3.12, BAT, and train-serving parity gates.
"""
```

Create `.codex/agents/agy-bridge.toml`:

```toml
name = "agy-bridge"
description = "Read-only bridge from Codex LO to AGY for environment, deployment, Python 3.12, BAT, disk, and operational risk checks."
sandbox_mode = "read-only"
developer_instructions = """
You are a bridge agent, not AGY itself and not the LO. Codex LO owns user communication, sequencing, decisions, verification, and final reporting.

For a self-contained brief file inside /mnt/c/model/mode_11_hana, invoke only:
  .agents/adapters/call_external_agent.sh agy <brief-path>

Do not call another subagent, contact the user, edit repository files, or claim success from model output alone. Validate the adapter JSON envelope and return: exact files changed (normally none), exact command, validation status, risks, and one recommended next step to Codex LO. Stop with BLOCK or HARD_STOP when Python 3.12 parity, BAT CRLF/chcp 65001, feature-build temp disk, protected paths, unconfirmed HANA schema, or RESEARCH_TRACK_FROZEN gates are violated.
"""
```

- [x] **Step 5: Publish minimal shared Fable settings and migrate machine permissions locally**

Create the versioned `.claude/settings.json` as exactly:

```json
{
  "advisorModel": "fable"
}
```

Move the pre-existing `permissions` object, without semantic changes, into ignored `.claude/settings.local.json` under its existing top-level `permissions` key. The local file must contain no `advisorModel`, and the shared file must contain no `permissions`, `bypassPermissions`, host path, or allowlist. Update `.gitignore` so `.claude/settings.json` is versionable and `.claude/settings.local.json` remains explicitly ignored. Never stage the local file.

- [x] **Step 6: Re-run the focused tests**

Run:

```bash
/mnt/c/model/mode_11_hana/.venv_wsl/bin/python -m pytest tests/test_agents/test_codex_lo_orchestration.py -k 'project or custom_bridge or fable' -v
```

Expected: 4 tests PASS.

- [x] **Step 7: Validate the files independently of pytest**

Run:

```bash
/mnt/c/model/mode_11_hana/.venv_wsl/bin/python -c "import json,pathlib; root=pathlib.Path('.'); shared=json.loads((root/'.claude/settings.json').read_text(encoding='utf-8')); local=json.loads((root/'.claude/settings.local.json').read_text(encoding='utf-8')); original=json.loads(pathlib.Path('/mnt/c/model/mode_11_hana/.claude/settings.json').read_text(encoding='utf-8')); assert shared == {'advisorModel':'fable'}; assert local == {'permissions':original['permissions']}; print('SETTINGS_SPLIT_OK')"
/mnt/c/model/mode_11_hana/.venv_wsl/bin/python -c "import tomllib,pathlib; root=pathlib.Path('.'); [tomllib.load(open(p,'rb')) for p in [root/'.codex/config.toml', root/'.codex/agents/claude-bridge.toml', root/'.codex/agents/agy-bridge.toml']]; print('CONFIG_PARSE_OK')"
git check-ignore --quiet -- .claude/settings.local.json
test $? -eq 0
git check-ignore --quiet -- .claude/settings.json
test $? -eq 1
```

Expected: `SETTINGS_SPLIT_OK`, `CONFIG_PARSE_OK`, shared settings versionable, and local settings ignored.

- [ ] **Step 8: Commit — skipped because no explicit user authorization was provided**

Status: `commit skipped — no explicit user authorization`. The commands below remain handoff-only:

```bash
git add .gitignore .codex/config.toml .codex/agents/claude-bridge.toml .codex/agents/agy-bridge.toml .claude/settings.json tests/test_agents/__init__.py tests/test_agents/test_codex_lo_orchestration.py
git commit -m "feat(agents): configure Codex LO bridges"
```

Otherwise do not stage anything and record: `commit skipped — no explicit user authorization`.

## Task 2: Build the bounded external CLI adapter with TDD

**Files:**

- Create: `.agents/adapters/call_external_agent.sh`
- Create: `tests/test_agents/fixtures/smoke_brief.md`
- Modify: `tests/test_agents/test_codex_lo_orchestration.py`

- [x] **Step 1: Add the fixed smoke brief**

Create `tests/test_agents/fixtures/smoke_brief.md`:

```markdown
# Read-only bridge smoke

Inspect no protected artifacts and change no files.
Return exactly these fields:
- status: ok
- files_changed: none
- validation: read-only smoke
- risk: none
- next_step: return to Codex LO
```

- [x] **Step 2: Add failing adapter tests**

Append the following helpers and tests to `tests/test_agents/test_codex_lo_orchestration.py`:

```python
def write_fake_cli(directory: Path, name: str, body: str) -> None:
    path = directory / name
    path.write_text(
        "#!/usr/bin/env bash\nset -eu\n" + textwrap.dedent(body),
        encoding="utf-8",
    )
    path.chmod(0o755)


def run_adapter(
    provider: str,
    brief: Path,
    fake_bin: Path,
    *,
    timeout_seconds: int = 2,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["AGENT_ADAPTER_TIMEOUT"] = str(timeout_seconds)
    return subprocess.run(
        [str(ADAPTER), provider, str(brief)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def parse_envelope(completed: subprocess.CompletedProcess[str]) -> dict:
    assert completed.stdout, completed.stderr
    return json.loads(completed.stdout)


def test_adapter_rejects_unknown_provider(tmp_path: Path) -> None:
    completed = run_adapter("unknown", SMOKE_BRIEF, tmp_path)
    envelope = parse_envelope(completed)
    assert completed.returncode == 2
    assert envelope["status"] == "error"
    assert envelope["error"] == "provider_not_allowed"


def test_adapter_rejects_brief_outside_repository(tmp_path: Path) -> None:
    brief = tmp_path / "outside.md"
    brief.write_text("outside", encoding="utf-8")
    completed = run_adapter("claude", brief, tmp_path)
    envelope = parse_envelope(completed)
    assert completed.returncode == 2
    assert envelope["error"] == "brief_outside_repo"


def test_claude_worker_is_read_only(tmp_path: Path) -> None:
    write_fake_cli(tmp_path, "claude", 'printf "FAKE_CLAUDE:%s\\n" "$*"\n')
    completed = run_adapter("claude", SMOKE_BRIEF, tmp_path)
    envelope = parse_envelope(completed)
    assert completed.returncode == 0
    assert envelope["status"] == "ok"
    assert envelope["provider"] == "claude"
    assert "--permission-mode plan" in envelope["stdout"]
    assert "--output-format json" in envelope["stdout"]


def test_claude_advisor_is_fresh_one_shot_and_builtin(tmp_path: Path) -> None:
    write_fake_cli(tmp_path, "claude", 'printf "FAKE_ADVISOR:%s\\n" "$*"\n')
    completed = run_adapter("claude-advisor", SMOKE_BRIEF, tmp_path)
    envelope = parse_envelope(completed)
    assert completed.returncode == 0
    assert envelope["status"] == "ok"
    assert envelope["provider"] == "claude-advisor"
    assert "--no-session-persistence" in envelope["stdout"]
    assert "--tools advisor" in envelope["stdout"]
    assert "exactly once" in envelope["stdout"]
    assert "--model fable" not in envelope["stdout"]


def test_agy_worker_uses_sandbox_and_plan_mode(tmp_path: Path) -> None:
    write_fake_cli(tmp_path, "agy", 'printf "FAKE_AGY:%s\\n" "$*"\n')
    completed = run_adapter("agy", SMOKE_BRIEF, tmp_path)
    envelope = parse_envelope(completed)
    assert completed.returncode == 0
    assert envelope["status"] == "ok"
    assert envelope["provider"] == "agy"
    assert "--sandbox" in envelope["stdout"]
    assert "--mode plan" in envelope["stdout"]


def test_adapter_reports_timeout(tmp_path: Path) -> None:
    write_fake_cli(tmp_path, "claude", "sleep 3\n")
    completed = run_adapter(
        "claude",
        SMOKE_BRIEF,
        tmp_path,
        timeout_seconds=1,
    )
    envelope = parse_envelope(completed)
    assert completed.returncode == 124
    assert envelope["status"] == "timeout"
    assert envelope["exit_code"] == 124


def test_adapter_redacts_secret_like_stderr(tmp_path: Path) -> None:
    write_fake_cli(
        tmp_path,
        "claude",
        'printf "API_KEY=topsecret\\n" >&2\nexit 7\n',
    )
    completed = run_adapter("claude", SMOKE_BRIEF, tmp_path)
    envelope = parse_envelope(completed)
    assert completed.returncode == 7
    assert envelope["status"] == "error"
    assert "topsecret" not in envelope["stderr_sanitized"]
    assert "[REDACTED]" in envelope["stderr_sanitized"]
```

- [x] **Step 3: Run the adapter tests and verify they fail**

Run:

```bash
/mnt/c/model/mode_11_hana/.venv_wsl/bin/python -m pytest tests/test_agents/test_codex_lo_orchestration.py -k 'adapter or claude_worker or claude_advisor or agy_worker' -v
```

Expected: FAIL because `.agents/adapters/call_external_agent.sh` does not exist.

- [x] **Step 4: Implement the minimal adapter**

Create `.agents/adapters/call_external_agent.sh` with LF line endings:

> [!WARNING]
> This minimal adapter block is historical TDD scaffolding only. Do not overwrite the current hardened `.agents/adapters/call_external_agent.sh`; the current adapter and `tests/test_agents/test_codex_lo_orchestration.py` are authoritative for timeout provenance, redaction, and temporary-directory safety.

```bash
#!/usr/bin/env bash
set -u
set -o pipefail

provider="${1:-}"
brief_arg="${2:-}"
timeout_seconds="${AGENT_ADAPTER_TIMEOUT:-300}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "$script_dir/../.." && pwd -P)"
start_epoch="$(date +%s)"

emit_simple_error() {
  local error="$1"
  local code="$2"
  jq -n \
    --arg status "error" \
    --arg error "$error" \
    --arg provider "$provider" \
    --argjson exit_code "$code" \
    '{status:$status,error:$error,provider:$provider,exit_code:$exit_code}'
  exit "$code"
}

case "$provider" in
  claude|claude-advisor|agy) ;;
  *) emit_simple_error "provider_not_allowed" 2 ;;
esac

if [[ -z "$brief_arg" ]]; then
  emit_simple_error "brief_missing" 2
fi

brief_path="$(realpath -e -- "$brief_arg" 2>/dev/null)" || \
  emit_simple_error "brief_not_found" 2

case "$brief_path" in
  "$repo_root"/*) ;;
  *) emit_simple_error "brief_outside_repo" 2 ;;
esac

for dependency in jq timeout realpath; do
  command -v "$dependency" >/dev/null 2>&1 || \
    emit_simple_error "missing_dependency:$dependency" 3
done

case "$provider" in
  claude|claude-advisor) command_name="claude" ;;
  agy) command_name="agy" ;;
esac
command -v "$command_name" >/dev/null 2>&1 || \
  emit_simple_error "missing_cli:$command_name" 3

work_dir="$(mktemp -d "${TMPDIR:-/tmp}/mode11-agent.XXXXXX")" || \
  emit_simple_error "tempdir_failed" 3
stdout_file="$work_dir/stdout"
stderr_file="$work_dir/stderr"
sanitized_file="$work_dir/stderr.sanitized"
cleanup() {
  rm -f -- "$stdout_file" "$stderr_file" "$sanitized_file"
  rmdir -- "$work_dir" 2>/dev/null || true
}
trap cleanup EXIT

brief_content="$(<"$brief_path")"

case "$provider" in
  claude)
    command=(
      claude -p "$brief_content"
      --permission-mode plan
      --tools Read,Grep,Glob
      --output-format json
      --no-session-persistence
    )
    ;;
  claude-advisor)
    advisor_prompt="Call the built-in advisor exactly once before any other tool. Do not call it again. Use the advisor feedback to answer this read-only brief, then return the required evidence fields.\n\n$brief_content"
    command=(
      claude -p "$advisor_prompt"
      --permission-mode plan
      --tools advisor
      --settings "$repo_root/.claude/settings.json"
      --output-format stream-json
      --verbose
      --no-session-persistence
    )
    ;;
  agy)
    command=(
      agy --sandbox --mode plan
      --add-dir "$repo_root"
      --print "$brief_content"
      --print-timeout "${timeout_seconds}s"
    )
    ;;
esac

(
  cd -- "$repo_root" || exit 3
  timeout --signal=TERM --kill-after=5s "$timeout_seconds" "${command[@]}"
) >"$stdout_file" 2>"$stderr_file"
exit_code=$?

sed -E \
  -e 's/(sk-[A-Za-z0-9_-]{8})[A-Za-z0-9_-]+/\1[REDACTED]/g' \
  -e 's/((API_KEY|TOKEN|SECRET|PASSWORD)[=:][[:space:]]*)[^[:space:]]+/\1[REDACTED]/Ig' \
  "$stderr_file" >"$sanitized_file"

end_epoch="$(date +%s)"
duration_seconds="$((end_epoch - start_epoch))"
if [[ "$exit_code" -eq 0 ]]; then
  status="ok"
elif [[ "$exit_code" -eq 124 ]]; then
  status="timeout"
else
  status="error"
fi

jq -n \
  --arg status "$status" \
  --arg provider "$provider" \
  --arg brief "$brief_path" \
  --argjson exit_code "$exit_code" \
  --argjson duration_seconds "$duration_seconds" \
  --rawfile stdout "$stdout_file" \
  --rawfile stderr_sanitized "$sanitized_file" \
  '{
    status:$status,
    provider:$provider,
    brief:$brief,
    exit_code:$exit_code,
    duration_seconds:$duration_seconds,
    stdout:$stdout,
    stderr_sanitized:$stderr_sanitized
  }'
exit "$exit_code"
```

Make it executable:

```bash
chmod 755 .agents/adapters/call_external_agent.sh
```

- [x] **Step 5: Run syntax and focused tests**

Run:

```bash
bash -n .agents/adapters/call_external_agent.sh
/mnt/c/model/mode_11_hana/.venv_wsl/bin/python -m pytest tests/test_agents/test_codex_lo_orchestration.py -k 'adapter or claude_worker or claude_advisor or agy_worker' -v
```

Expected: shell syntax passes and 7 adapter tests PASS.

- [x] **Step 6: Run ShellCheck when available**

Run:

```bash
if command -v shellcheck >/dev/null 2>&1; then shellcheck .agents/adapters/call_external_agent.sh; else echo 'SHELLCHECK_NOT_INSTALLED'; fi
```

Expected: no ShellCheck findings, or the explicit non-installed notice. Do not install packages for this task.

- [ ] **Step 7: Commit — skipped because no explicit user authorization was provided**

Status: `commit skipped — no explicit user authorization`. The commands below remain handoff-only:

```bash
git add .agents/adapters/call_external_agent.sh tests/test_agents/fixtures/smoke_brief.md tests/test_agents/test_codex_lo_orchestration.py
git commit -m "feat(agents): add bounded external bridges"
```

Otherwise do not stage anything and record the skipped commit.

## Task 3: Make governance consistent and remove OpenCode from active surfaces

**Files:**

- Modify: `AGENTS.md`
- Modify: `CLAUDE.md`
- Modify: `.agents/agents_config.json`
- Modify: `.agents/codex_hq.md`
- Modify: `.agents/claude_hq.md`
- Modify: `.agents/agy_hq.md`
- Modify: `.agents/message_deferral_guide.md`
- Delete: `.agents/opencode_hq.md`
- Delete: `.claude/agents/opencode-worker.md`
- Delete: `.claude/agents/agy-bridge.md`
- Delete: `.claude/agents/antigravity-worker.md`
- Delete: `.claude/agents/claude-bridge.md`
- Delete: `.claude/agents/codex-bridge.md`
- Delete: `.claude/agents/codex-worker.md`
- Delete: `.claude/agents/hermes-worker.md`
- Modify: `tests/test_agents/test_codex_lo_orchestration.py`

- [x] **Step 1: Add failing active-policy tests**

Append to `tests/test_agents/test_codex_lo_orchestration.py`:

```python
ACTIVE_POLICY_FILES = [
    ROOT / "AGENTS.md",
    ROOT / "CLAUDE.md",
    ROOT / ".agents" / "agents_config.json",
    ROOT / ".agents" / "codex_hq.md",
    ROOT / ".agents" / "claude_hq.md",
    ROOT / ".agents" / "agy_hq.md",
    ROOT / ".agents" / "message_deferral_guide.md",
    CODEX_CONFIG,
    CLAUDE_AGENT,
    AGY_AGENT,
]


def test_codex_is_the_only_active_lo() -> None:
    contents = "\n".join(
        path.read_text(encoding="utf-8") for path in ACTIVE_POLICY_FILES
    )
    assert "Codex LO" in contents
    assert "OpenCode" not in contents
    assert "Hermes LO" not in contents
    assert "Claude LO" not in contents


def test_only_approved_active_agent_definitions_remain() -> None:
    assert not list((ROOT / ".agents").glob("*opencode*"))
    assert not list((ROOT / ".claude" / "agents").glob("*.md"))


def test_agents_config_matches_runtime_topology() -> None:
    config = json.loads(
        (ROOT / ".agents" / "agents_config.json").read_text(encoding="utf-8")
    )
    assert config["version"] == "2.0.0"
    assert config["lo_configuration"]["lo_agent"] == "Codex"
    assert set(config["external_workers"]) == {"Claude Code", "AGY"}
    assert config["advisor"]["model_alias"] == "fable"
    assert config["advisor"]["mechanism"] == "Claude Code built-in advisor"
    assert config["advisor"]["max_calls_per_session"] == 1
    handlers = config["actual_handlers"]
    assert set(handlers) == {"claude", "claude-advisor", "agy"}
    serialized = json.dumps(config, ensure_ascii=False)
    assert "opencode" not in serialized.lower()


def test_hard_gates_remain_in_active_policy() -> None:
    contents = "\n".join(
        path.read_text(encoding="utf-8") for path in ACTIVE_POLICY_FILES
    )
    for required in (
        "RESEARCH_TRACK_FROZEN",
        "Python 3.12",
        "chcp 65001",
        "packages_win/py312/",
        "mlruns/",
        "RequestFeatureBuilder",
        "HANA_FEAT_TMP",
    ):
        assert required in contents
```

- [x] **Step 2: Run the policy tests and confirm they fail**

Run:

```bash
/mnt/c/model/mode_11_hana/.venv_wsl/bin/python -m pytest tests/test_agents/test_codex_lo_orchestration.py -k 'only_active_lo or approved_active or runtime_topology or hard_gates' -v
```

Expected: FAIL on current OpenCode/Hermes references, legacy `.claude/agents/*.md`, and the old JSON topology.

- [x] **Step 3: Update `AGENTS.md` orchestration ownership while preserving hard gates**

Replace the opening paragraph with:

```markdown
This repository is a HANA prescription-data ML serving system for inappropriate-prescription risk prediction. Codex is the sole L0/LO orchestrator: it owns user communication, decomposition, sequencing, implementation, verification, conflict resolution, and final reporting. Claude Code and AGY are bounded external subagents that return evidence; Codex LO verifies that evidence before reporting success.
```

Apply these exact ownership replacements throughout `AGENTS.md`:

```text
AGY/OpenCode LO          -> AGY/Codex LO
OpenCode LO              -> Codex LO
AGY / OpenCode           -> AGY / Codex LO
```

Replace `## Subagent roles` with:

```markdown
## Subagent roles

- Claude Code / `claude-bridge`: requirements, architecture, operational definitions, label semantics, leakage/schema/freeze logical review, and final QA. Its `claude-advisor` mode uses the built-in Fable 5 advisor exactly once in a fresh session.
- AGY HQ / `agy-bridge`: environment, DevOps, Windows offline deployment, Python 3.12 parity, BAT/CRLF checks, disk-space and risk gates, and explicitly requested external research. It is not the orchestrator or implementation owner.
- Codex LO implements and validates directly. There is no separate Codex worker in the default topology.
```

Replace the first communication bullet with:

```markdown
- Codex LO may have only one outbound external-worker request in flight. Queue additional Claude/AGY work until the current worker reports completion or idle.
```

Leave the dataset, freeze, schema, parity, BAT, protected-path, and shared-constant rules substantively unchanged.

- [x] **Step 4: Replace the active multi-AI section in `CLAUDE.md`**

Keep the project identity, environment, HANA, schema, freeze, and “묻지 않고 하면 안 되는 것” sections. Replace `## 다중 AI 협업 (전역 + 본 레포 우선순위)` through the message-deferral subsection with:

```markdown
## 다중 AI 협업

- **Codex LO**: 유일한 L0/LO. 사용자 소통, 작업 분해, 구현, worker 라우팅, 승인, 통합, 검증, 충돌 해결, 최종 보고를 담당한다.
- **Claude Code subagent**: 요구사항·아키텍처·운영 정의·label/schema/freeze 논리 검토·최종 QA를 read-only로 반환한다.
- **Fable 5 advisor**: Claude Code 내장 advisor다. `advisorModel: "fable"`을 사용하며 plan/finish 검토는 각각 새 Claude 세션에서 정확히 한 번 호출한다.
- **AGY subagent**: Python 3.12, Windows 폐쇄망 배포, BAT CRLF/`chcp 65001`, offline dependency, feature-build temp disk, 보호 경로와 리스크 게이트를 read-only로 점검한다.
- 실행 정본은 `AGENTS.md`, `.codex/config.toml`, `.codex/agents/*.toml`, `.agents/agents_config.json`, `.agents/adapters/call_external_agent.sh`다.
- `.multiagent/`는 gitignored 레거시 생성물이며 현재 실행 정본이 아니다.
- Critical 변경(라벨 정의·학습/서빙 스키마·HANA 쿼리·freeze/gate 정책)은 Codex와 Claude의 cross-family review가 필요하다.

### 메시지 전송 보류 원칙

- Codex LO는 외부 worker 요청을 한 번에 하나만 보낸다.
- Claude 또는 AGY 응답이 완료/idle 상태가 되기 전에는 다음 outbound 요청을 보류한다.
- worker는 다른 worker를 호출하거나 사용자에게 직접 메시지하지 않는다.
```

- [x] **Step 5: Replace `.agents/agents_config.json` with the executable topology**

Use this complete JSON:

> [!WARNING]
> This JSON block is historical topology scaffolding only. Do not overwrite the current hardened `.agents/agents_config.json`; that file and the current orchestration tests are authoritative, including the fail-closed active-policy command and the two-layer protected Git/metadata validation commands.

```json
{
  "project": "MODE_11_hana",
  "version": "2.0.0",
  "operating_model": {
    "default": "codex_lo",
    "summary": "Codex is the sole LO. Claude Code and AGY are sequential, read-only external workers."
  },
  "lo_configuration": {
    "lo_agent": "Codex",
    "responsibilities": [
      "user communication",
      "decomposition and sequencing",
      "implementation",
      "worker dispatch",
      "evidence verification",
      "conflict resolution",
      "final reporting"
    ],
    "max_outbound_workers_in_flight": 1,
    "max_spawn_depth": 1
  },
  "external_workers": [
    "Claude Code",
    "AGY"
  ],
  "advisor": {
    "provider": "Claude Code",
    "mechanism": "Claude Code built-in advisor",
    "model_alias": "fable",
    "resolved_model": "claude-fable-5",
    "default_gates": [
      "plan",
      "finish"
    ],
    "fresh_session_per_call": true,
    "max_calls_per_session": 1
  },
  "actual_handlers": {
    "claude": {
      "bridge": ".codex/agents/claude-bridge.toml",
      "adapter": ".agents/adapters/call_external_agent.sh claude",
      "default_access": "read-only",
      "role": "requirements, architecture, operational definitions, logical QA, final QA"
    },
    "claude-advisor": {
      "bridge": ".codex/agents/claude-bridge.toml",
      "adapter": ".agents/adapters/call_external_agent.sh claude-advisor",
      "default_access": "read-only",
      "role": "one-shot built-in Fable advisor review in a fresh Claude session"
    },
    "agy": {
      "bridge": ".codex/agents/agy-bridge.toml",
      "adapter": ".agents/adapters/call_external_agent.sh agy",
      "default_access": "read-only sandbox/plan",
      "role": "environment, deployment, Python 3.12, BAT, disk, and operational risk"
    }
  },
  "critical_review": {
    "topics": [
      "label definitions",
      "train-serving schema",
      "HANA query logic",
      "freeze or gate policy"
    ],
    "required_families": [
      "OpenAI via Codex LO",
      "Anthropic via Claude Code"
    ]
  },
  "hard_gates": {
    "python_runtime": "Python 3.12 only",
    "bat": "CRLF and chcp 65001; AGY sign-off required for edits",
    "feature_temp": "HANA_FEAT_TMP -> HANA_TMP_DIR -> hana_config.json -> system temp; 10GB+ free",
    "feature_parity": "RequestFeatureBuilder names and order must match training; serving changes require schema diff, tests/test_serving, tests/test_features, /reload, and sample payload sanity",
    "protected_paths": [
      "packages_win/py312/",
      "mlruns/",
      "**/*.parquet",
      "out/"
    ],
    "research": "RESEARCH_TRACK_FROZEN; no Nov->Dec tuning, ablation, feature, or hyperparameter work; Gate 5A/Gate 5B/2025-01 are retired"
  },
  "validation_commands": {
    "python": "/mnt/c/model/mode_11_hana/.venv_wsl/bin/python --version",
    "toml_json": "/mnt/c/model/mode_11_hana/.venv_wsl/bin/python -c \"import json,tomllib,pathlib; root=pathlib.Path('.'); json.load(open(root/'.agents/agents_config.json',encoding='utf-8')); [tomllib.load(open(p,'rb')) for p in [root/'.codex/config.toml',root/'.codex/agents/claude-bridge.toml',root/'.codex/agents/agy-bridge.toml']]; print('CONFIG_PARSE_OK')\"",
    "adapter_syntax": "bash -n .agents/adapters/call_external_agent.sh",
    "tests": "/mnt/c/model/mode_11_hana/.venv_wsl/bin/python -m pytest tests/test_agents/test_codex_lo_orchestration.py -v",
    "active_provider_check": "rg -n -i 'opencode|Hermes LO|Claude LO' AGENTS.md CLAUDE.md .agents .codex .claude/settings.json",
    "protected_diff": "git diff --name-only -- packages_win/py312 mlruns out '*.parquet' '*.bat'"
  }
}
```

- [x] **Step 6: Rewrite the three role documents and deferral guide**

Replace `.agents/codex_hq.md` completely with:

```markdown
# Codex LO

Codex is MODE_11_hana's sole L0/LO. It owns user communication, decomposition, sequencing, implementation, worker dispatch, evidence verification, conflict resolution, and final reporting. Claude Code and AGY return evidence; Codex verifies it before adoption.

## Responsibilities

- Implement scoped repository changes directly with TDD and focused validation.
- Send self-contained read-only briefs to Claude Code or AGY only when their specialty adds value.
- Keep at most one external-worker request in flight and prevent recursive delegation.
- Verify worker claims against files, commands, tests, and repository policy before adoption.
- Require OpenAI-via-Codex and Anthropic-via-Claude review for critical label, train-serving schema, HANA query, or freeze/gate policy changes.

## Routing

- Claude Code worker: requirements, architecture, operational definitions, leakage/schema/freeze logic, and final QA.
- Claude advisor mode: built-in Fable 5 advisor, exactly once in a fresh Claude session for plan or finish review.
- AGY worker: Python 3.12, Windows closed-network deployment, BAT, offline dependency, feature temp disk, protected-path, and operational risk gates.

## Hard gates

- Final data is 2024-07..12 Raw: 184 daily records parquet files plus 500k eligibility. No 2025-01 acquisition.
- `RESEARCH_TRACK_FROZEN` is indefinite. No Nov-to-Dec tuning, ablation, feature, or hyperparameter work. Gate 5A and Gate 5B are retired.
- Never guess HANA schema, table, or column names.
- `RequestFeatureBuilder` names and order must match training. Serving changes require a training schema diff, `tests/test_serving`, `tests/test_features`, `/reload`, and sample payload sanity.
- Active Python and pytest must be Python 3.12.
- Feature builds require the `HANA_FEAT_TMP`/fallback destination and 10GB+ free-space preflight.
- `.bat` edits require CRLF, `chcp 65001`, and AGY sign-off.
- Do not modify or delete `packages_win/py312/`, `mlruns/`, generated parquet, or `out/` without explicit user approval.

## Worker-result acceptance

Accept only results that state exact files changed, exact commands/tests, validation status, risks, and one recommended next step. Worker success is evidence, not the final decision.
```

Replace `.agents/claude_hq.md` completely with:

```markdown
# Claude Code Worker

Claude Code is Codex LO's read-only requirements, architecture, operational-definition, logical-QA, and final-QA worker. It never communicates with the user or another worker. Advisor work uses Claude Code's built-in advisor with `advisorModel: "fable"`, exactly once in a fresh session, and returns the synthesized evidence to Codex LO.

## Modes

- `claude`: inspect requirements, architecture, label semantics, leakage, schema parity, freeze logic, or final QA in read-only mode.
- `claude-advisor`: call the built-in Fable 5 advisor exactly once before any other tool, never retry in the same session, and synthesize its advice for Codex LO.

## Hard gates

- Final data is 2024-07..12 Raw: 184 daily records parquet files plus 500k eligibility. No 2025-01 acquisition.
- `RESEARCH_TRACK_FROZEN` is indefinite. No Nov-to-Dec tuning, ablation, feature, or hyperparameter work. Gate 5A and Gate 5B are retired.
- Never guess HANA schema, table, or column names.
- `RequestFeatureBuilder` names and order must match training. Serving changes require a training schema diff, `tests/test_serving`, `tests/test_features`, `/reload`, and sample payload sanity.
- Active Python and pytest must be Python 3.12.
- Feature builds require the `HANA_FEAT_TMP`/fallback destination and 10GB+ free-space preflight.
- `.bat` edits require CRLF and `chcp 65001`; Claude does not finalize BAT changes.
- Do not modify or delete `packages_win/py312/`, `mlruns/`, generated parquet, or `out/` without explicit user approval.

## Return format

Return result type, summary, evidence or deliverable, exact files changed, exact commands/tests, validation status, blockers, risks, and one recommended next step to Codex LO.
```

Replace `.agents/agy_hq.md` completely with:

```markdown
# AGY Worker

AGY is Codex LO's read-only environment, DevOps, Python 3.12, Windows offline deployment, BAT, temp-disk, and operational-risk worker. It is not the orchestrator or implementation owner and returns all blockers and evidence to Codex LO.

## Responsibilities

- Check active Python and pytest are Python 3.12 and report any other active runtime as BLOCK.
- Check Windows closed-network packaging and offline dependency risks without changing the wheelhouse.
- Verify `.bat` edits preserve CRLF and contain `chcp 65001`.
- Before any feature build, resolve `HANA_FEAT_TMP` -> `HANA_TMP_DIR` -> `hana_config.json` -> system temp and require 10GB+ free space.
- Detect protected-path, schema-guessing, train-serving parity, and research-freeze violations.

## Hard gates

- Final data is 2024-07..12 Raw: 184 daily records parquet files plus 500k eligibility. No 2025-01 acquisition.
- `RESEARCH_TRACK_FROZEN` is indefinite. No Nov-to-Dec tuning, ablation, feature, or hyperparameter work. Gate 5A and Gate 5B are retired.
- Never guess HANA schema, table, or column names.
- `RequestFeatureBuilder` names and order must match training. Serving changes require a training schema diff, `tests/test_serving`, `tests/test_features`, `/reload`, and sample payload sanity.
- Do not modify or delete `packages_win/py312/`, `mlruns/`, generated parquet, or `out/` without explicit user approval.

## Return format

Return scope, exact files changed, gate diagnostics, exact commands/tests, findings, validation status, risks, and one recommended next step to Codex LO.
```

Replace `.agents/message_deferral_guide.md` with:

```markdown
# Sequential external-worker dispatch

Codex LO may have only one outbound Claude Code or AGY request in flight.

## Rules

1. Send a self-contained brief to one worker.
2. Until that worker reports completion or idle, queue all follow-up and other-worker messages locally.
3. After completion, Codex LO validates the result before sending the next request.
4. Workers never communicate with each other or the user and never spawn another worker.
5. Do not poll consuming bridge queues. Use the adapter process completion and result envelope.

Every worker result must include exact files changed, commands/tests run, validation status, risks, and one recommended next step.
```

- [x] **Step 7: Remove all obsolete active agent definitions**

Delete exactly these tracked files with `apply_patch`:

```text
.agents/opencode_hq.md
.claude/agents/opencode-worker.md
.claude/agents/agy-bridge.md
.claude/agents/antigravity-worker.md
.claude/agents/claude-bridge.md
.claude/agents/codex-bridge.md
.claude/agents/codex-worker.md
.claude/agents/hermes-worker.md
```

Do not remove the `.claude/agents/` directory itself and do not use `rm -rf`.

- [x] **Step 8: Run the active-policy tests**

Run:

```bash
/mnt/c/model/mode_11_hana/.venv_wsl/bin/python -m pytest tests/test_agents/test_codex_lo_orchestration.py -k 'only_active_lo or approved_active or runtime_topology or hard_gates' -v
```

Expected: 4 tests PASS.

- [x] **Step 9: Run the active-provider residual search**

Run:

```bash
rg -n -i 'opencode|Hermes LO|Claude LO' AGENTS.md CLAUDE.md .agents .codex .claude/settings.json
```

Expected: no output. Historical docs and Git history are intentionally outside this active-surface command.

- [ ] **Step 10: Commit — skipped because no explicit user authorization was provided**

Status: `commit skipped — no explicit user authorization`. The commands below remain handoff-only:

```bash
git add AGENTS.md CLAUDE.md .agents .claude/agents tests/test_agents/test_codex_lo_orchestration.py
git commit -m "refactor(agents): make Codex the sole LO"
```

Otherwise do not stage anything and record the skipped commit.

## Task 4: Mark the prior OpenCode design as historical and verify the complete system

**Files:**

- Create: `docs/superpowers/specs/2026-07-14-codex-lo-multiagent-design.md`
- Create: `docs/superpowers/plans/2026-07-14-codex-lo-multiagent.md`
- Modify: `docs/superpowers/specs/2026-07-12-opencode-lo-contract-design.md`
- Modify: `docs/superpowers/specs/contracts/profile_contracts.md`
- Verify: every approved tracked and untracked file in the final file map

- [x] **Step 1: Snapshot protected artifact metadata before Task 4 scoped changes**

Run these commands in one shell and retain the state variable through Step 6:

```bash
task4_protected_state="/tmp/mode11-task4-protected-$$.json"
test ! -e "$task4_protected_state"
/mnt/c/model/mode_11_hana/.venv_wsl/bin/python .agents/adapters/protected_artifact_guard.py snapshot --root . --state "$task4_protected_state"
```

Expected: Python 3.12 runs the configured metadata-only guard, the unique state is outside the repository, and the snapshot reports `status: "ok"`.

- [x] **Step 2: Add a superseded notice without rewriting history**

Insert immediately below the title in `docs/superpowers/specs/2026-07-12-opencode-lo-contract-design.md`:

```markdown
> [!IMPORTANT]
> **Superseded on 2026-07-14.** OpenCode is no longer an active agent, worker, fallback, or LO in MODE_11_hana. The current orchestration design is `docs/superpowers/specs/2026-07-14-codex-lo-multiagent-design.md`, with Codex as the sole LO and Claude Code/AGY as the only external subagents. The remainder of this file is retained only as historical decision provenance and is not an executable policy.
```

- [x] **Step 3: Run all orchestration tests**

Run:

```bash
/mnt/c/model/mode_11_hana/.venv_wsl/bin/python -m pytest tests/test_agents/test_codex_lo_orchestration.py -v
```

Expected: the complete hardened suite passes (currently 60 tests).

- [x] **Step 4: Run parser, syntax, and bytecode checks**

Run:

```bash
/mnt/c/model/mode_11_hana/.venv_wsl/bin/python -m json.tool .agents/agents_config.json >/dev/null
/mnt/c/model/mode_11_hana/.venv_wsl/bin/python -m json.tool .claude/settings.json >/dev/null
/mnt/c/model/mode_11_hana/.venv_wsl/bin/python -m json.tool .claude/settings.local.json >/dev/null
/mnt/c/model/mode_11_hana/.venv_wsl/bin/python -c "import tomllib,pathlib; root=pathlib.Path('.'); [tomllib.load(open(p,'rb')) for p in [root/'.codex/config.toml',root/'.codex/agents/claude-bridge.toml',root/'.codex/agents/agy-bridge.toml']]; print('TOML_OK')"
bash -n .agents/adapters/call_external_agent.sh
env PYTHONPYCACHEPREFIX=/tmp/mode11-task4-pycache /mnt/c/model/mode_11_hana/.venv_wsl/bin/python -m py_compile .agents/adapters/protected_artifact_guard.py
```

Expected: `TOML_OK` and exit code 0 from every command, with guard bytecode written only under `/tmp`.

- [x] **Step 5: Validate strict configuration and feature state separately**

Run:

```bash
codex --strict-config doctor --summary --ascii
```

Expected evidence: the configuration section reports `[ok] config loaded` (spacing may be padded) and no unknown-key or configuration-parse error. The overall doctor exit may be nonzero because of unrelated terminal, state, or provider-reachability diagnostics; it is not the feature-state gate.

Then run the separate feature-state gate:

```bash
codex features list
```

Expected: exit code 0 and `multi_agent` reports `stable true`.

- [x] **Step 6: Run the configured protected Git-surface validator and metadata verification**

Run the current `validation_commands.protected_diff` from `.agents/agents_config.json` verbatim, then verify the pre-change metadata state:

```bash
/mnt/c/model/mode_11_hana/.venv_wsl/bin/python -c "import json,subprocess; config=json.load(open('.agents/agents_config.json',encoding='utf-8')); command=config['validation_commands']['protected_diff']; raise SystemExit(subprocess.run(['bash','-c',command]).returncode)"
/mnt/c/model/mode_11_hana/.venv_wsl/bin/python .agents/adapters/protected_artifact_guard.py verify --root . --state "$task4_protected_state"
```

Both commands must exit 0. The configured Git validator certifies the tracked, staged, untracked, and branch commit surface; the metadata guard certifies ignored protected artifacts and BAT files against the pre-change snapshot.

The following simple query is supplemental visibility only and is never the certification:

```bash
git diff --name-only -- packages_win/py312 mlruns out '*.parquet' '*.bat'
```

Expected: no output.

- [x] **Step 7: Review tracked and untracked scope completely**

Run:

```bash
git status --short --untracked-files=all
git diff --check
git diff -- .gitignore AGENTS.md CLAUDE.md .agents .claude/agents docs/superpowers/specs/2026-07-12-opencode-lo-contract-design.md docs/superpowers/specs/contracts/profile_contracts.md
```

Normal `git diff` is insufficient because it omits untracked files. Inspect every approved non-empty untracked file directly and require `git diff --no-index` return code 1 (content differs from `/dev/null`):

```bash
untracked_nonempty=(
  .agents/adapters/call_external_agent.sh
  .agents/adapters/protected_artifact_guard.py
  .claude/settings.json
  .codex/config.toml
  .codex/agents/claude-bridge.toml
  .codex/agents/agy-bridge.toml
  docs/superpowers/specs/2026-07-14-codex-lo-multiagent-design.md
  docs/superpowers/plans/2026-07-14-codex-lo-multiagent.md
  tests/test_agents/fixtures/smoke_brief.md
  tests/test_agents/test_codex_lo_orchestration.py
)
review_failed=0
for path in "${untracked_nonempty[@]}"; do
  git diff --no-index -- /dev/null "$path"
  rc=$?
  if [ "$rc" -ne 1 ]; then
    review_failed=1
  fi
done
test "$review_failed" -eq 0
test -f .claude/agents/.gitkeep
test ! -s .claude/agents/.gitkeep
test -f tests/test_agents/__init__.py
test ! -s tests/test_agents/__init__.py
```

The status, tracked diff, direct untracked inspection, and empty-marker checks together must cover `.gitignore`, the versioned minimal `.claude/settings.json`, both new docs, all tests, all TOMLs, both adapters, `.agents/adapters/protected_artifact_guard.py`, and `.claude/agents/.gitkeep`. `.claude/settings.local.json` is migration-only ignored machine state: parse it and verify permission preservation, but never stage, commit, or review it as a deliverable. Preserve unrelated worktree changes.

- [ ] **Step 8: Commit — skipped because no explicit user authorization was provided**

Do not stage or commit. The historical commit commands are handoff-only and remain unauthorized.

## Task 5: Run explicitly approved live Claude, Fable advisor, and AGY smokes

**Files:**

- Read: `tests/test_agents/fixtures/smoke_brief.md`
- Execute: `.agents/adapters/call_external_agent.sh`
- Verify: configured protected Git surface plus metadata state
- Do not modify repository files.

- [x] **Step 1: Obtain approval and snapshot protected metadata before the first live call**

After explicit live-call approval, run these commands in one shell and retain the state variable through Step 5:

```bash
task5_protected_state="/tmp/mode11-task5-live-protected-$$.json"
test ! -e "$task5_protected_state"
/mnt/c/model/mode_11_hana/.venv_wsl/bin/python .agents/adapters/protected_artifact_guard.py snapshot --root . --state "$task5_protected_state"
```

Do not start any external adapter call unless the snapshot reports `status: "ok"`.

- [x] **Step 2: Run the Claude worker live call**

Run only after the approval surface grants external/home access:

```bash
.agents/adapters/call_external_agent.sh claude tests/test_agents/fixtures/smoke_brief.md
```

Expected: JSON envelope with `status: "ok"`, `provider: "claude"`, `exit_code: 0`, and a Claude JSON response in `stdout`. Confirm `git status --short --untracked-files=all` shows no new model-written repository file.

- [x] **Step 3: Run the separate one-shot built-in Fable advisor call**

Run:

```bash
.agents/adapters/call_external_agent.sh claude-advisor tests/test_agents/fixtures/smoke_brief.md
```

Expected: JSON envelope with `status: "ok"`, `provider: "claude-advisor"`, and `exit_code: 0`. Inspect the Claude output/debug metadata for `advisorModel` or `claude-fable-5`; do not claim Fable validation from prompt wording alone. The adapter starts a fresh no-persistence session and makes no second advisor call.

- [x] **Step 4: Run the AGY live call**

Run:

```bash
.agents/adapters/call_external_agent.sh agy tests/test_agents/fixtures/smoke_brief.md
```

Expected: JSON envelope with `status: "ok"`, `provider: "agy"`, `exit_code: 0`, and the requested evidence fields. If sandbox/home/socket restrictions fail, record the exact BLOCK; do not add `--dangerously-skip-permissions`.

- [x] **Step 5: Re-run tests and complete both protected certifications after all calls**

Run:

```bash
/mnt/c/model/mode_11_hana/.venv_wsl/bin/python -m pytest tests/test_agents/test_codex_lo_orchestration.py -v
/mnt/c/model/mode_11_hana/.venv_wsl/bin/python -c "import json,subprocess; config=json.load(open('.agents/agents_config.json',encoding='utf-8')); command=config['validation_commands']['protected_diff']; raise SystemExit(subprocess.run(['bash','-c',command]).returncode)"
/mnt/c/model/mode_11_hana/.venv_wsl/bin/python .agents/adapters/protected_artifact_guard.py verify --root . --state "$task5_protected_state"
git status --short --untracked-files=all
```

Expected: all tests pass, the configured Git-surface validator exits 0, metadata verification reports `status: "ok"`, and no new live-call file appears.

The simple query below is supplemental only:

```bash
git diff --name-only -- packages_win/py312 mlruns out '*.parquet' '*.bat'
```

Expected: no output.

- [x] **Step 6: Produce the final handoff**

The exact deliverable file lists must explicitly include `.gitignore`, `.claude/settings.json`, `.agents/adapters/protected_artifact_guard.py`, `.claude/agents/.gitkeep`, both new 2026-07-14 documents, every test/TOML, and both adapters; do not rely on a normal `git diff` to discover untracked files. Never include `.claude/settings.local.json` in the deliverable list.

Report exactly:

```text
Files changed: <exact paths>
Files deleted: <exact paths>
Commands/tests: <exact commands and exit status>
Live validation: <Claude / Fable advisor / AGY, each passed|failed|blocked>
OpenCode active residuals: <none or exact locations>
Protected Git commit surface: <configured protected_diff result>
Protected metadata/BAT state: <snapshot and verify result>
Risks: <remaining limitations, especially sandbox/quota/model metadata>
Commit status: not committed unless explicitly authorized
Recommended next step: restart Codex in the trusted repository so project .codex/config.toml and custom agents load
```

### Recorded Task 5 results (2026-07-14)

- Pre-call protected metadata snapshot: `/tmp/mode11-task5-protected-baseline.json`; 16 entries; metadata only.
- Claude worker adapter: `status: ok`, exit code 0, duration 7 seconds. It returned exactly `status: ok`, `files_changed: none`, `validation: read-only smoke`, `risk: none`, and `next_step: Codex LO`; base-model metadata was `claude-opus-4-8[1m]`.
- Claude advisor adapter: fresh no-persistence session; `status: ok`, exit code 0, duration 82 seconds. Output contained exactly one `server_tool_use` named `advisor` and exactly one `advisor_message` whose model was `claude-fable-5`; `modelUsage` also confirmed `claude-fable-5`, and no second advisor call occurred. It returned the exact smoke fields above and changed no repository file.
- AGY adapter: `status: ok`, exit code 0, duration 11 seconds. It returned the exact smoke fields above and changed no repository file.
- Claude final QA: **PASS**, with no blocking findings.
- Live-path coverage: the three approved smokes exercised the adapter entrypoint directly, not through the strict read-only custom bridges. A local Codex strict `:read-only` probe blocked `mktemp /tmp` as expected fail-closed behavior; bridge-mediated live dispatch remains not directly exercised and requires parent-approved runtime permission.
- Post-live Python 3.12 validation: the orchestration suite passed with `60 passed in 13.34s`; the configured active/protected validators passed; shared settings were exactly `advisorModel: fable`; all TOMLs and Bash parsed; protected metadata verification passed with all 16 entries unchanged; the staged diff was empty; and worktree status remained unchanged from the approved orchestration surface.
- Repository integrity: no provider wrote a repository file, and no protected or BAT artifact changed. No commit was made; every commit checkbox remains skipped and unchecked.
- Finish-review role split: Claude Code performed the independent file-level cross-family review with `Read,Grep,Glob`. The built-in Fable advisor performed an evidence-only architecture/risk challenge because advisor mode intentionally exposes only the `advisor` tool.
- The first substantive Fable finish attempt returned a truthful environment BLOCK after it tried to perform file reads that advisor mode does not expose; this was not a repository defect. The brief was corrected to require evidence-packet review only, without weakening permissions or adding tools.
- Corrected Fable finish advisory: **PASS**, `status: ok`, exit code 0, duration 169 seconds, fresh no-persistence session, exactly one `server_tool_use` named `advisor`, and `modelUsage` confirmed `claude-fable-5`. It changed no file and retained the documented risks: approval-gated bridge dispatch is not end-to-end exercised, raw provider stdout is not redacted, and the protected-artifact guard is metadata-only.
- **Finish review: COMPLETE.** Claude supplied the file-level PASS; Fable supplied the one-shot evidence/architecture PASS; Codex LO retains final acceptance authority.
