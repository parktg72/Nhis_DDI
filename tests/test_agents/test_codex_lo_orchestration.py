import json
import os
import re
import shutil
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
CLAUDE_LOCAL_SETTINGS = ROOT / ".claude" / "settings.local.json"
ADAPTER = ROOT / ".agents" / "adapters" / "call_external_agent.sh"
PROTECTED_GUARD = (
    ROOT / ".agents" / "adapters" / "protected_artifact_guard.py"
)
PYTHON_312 = Path("/mnt/c/model/mode_11_hana/.venv_wsl/bin/python")
SMOKE_BRIEF = ROOT / "tests" / "test_agents" / "fixtures" / "smoke_brief.md"
EXPECTED_AGENTS_FILES = {
    "adapters/call_external_agent.sh",
    "adapters/protected_artifact_guard.py",
    "agents_config.json",
    "agy_hq.md",
    "claude_hq.md",
    "codex_hq.md",
    "message_deferral_guide.md",
}
EXPECTED_CODEX_FILES = {
    "agents/agy-bridge.toml",
    "agents/claude-bridge.toml",
    "config.toml",
}
BAT_PATHSPECS = (
    ":(glob)*.bat",
    ":(glob)**/*.bat",
)
PROTECTED_PATHSPECS = (
    "packages_win/py312/",
    "mlruns/",
    "out/",
    ":(glob)*.parquet",
    ":(glob)**/*.parquet",
)
ADVISOR_INSTRUCTION = (
    "Call the built-in advisor exactly once before any other tool. "
    "Do not call it again. Use the advisor feedback to answer this read-only "
    "brief, then return the required evidence fields."
)
ARGV_DUMP_BODY = "jq -cn --args '$ARGS.positional' -- \"$@\"\n"


def files_under(directory):
    return {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file()
    }


def active_policy_files():
    files = {
        ROOT / "AGENTS.md",
        ROOT / "CLAUDE.md",
        CLAUDE_SETTINGS,
    }
    for directory in (
        ROOT / ".agents",
        ROOT / ".codex",
        ROOT / ".claude" / "agents",
    ):
        files.update(path for path in directory.rglob("*") if path.is_file())
    return tuple(sorted(files))


def changed_paths(pathspecs):
    commands = (
        ["git", "diff", "--name-only", "HEAD", "--", *pathspecs],
        [
            "git",
            "ls-files",
            "--others",
            "--exclude-standard",
            "--",
            *pathspecs,
        ],
    )
    paths = set()
    for command in commands:
        result = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        paths.update(line for line in result.stdout.splitlines() if line)

    main_ref = subprocess.run(
        [
            "git",
            "show-ref",
            "--verify",
            "--quiet",
            "refs/heads/main",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if main_ref.returncode == 0:
        branch_diff = subprocess.run(
            [
                "git",
                "diff",
                "--name-only",
                "main...HEAD",
                "--",
                *pathspecs,
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert branch_diff.returncode == 0, branch_diff.stderr
        paths.update(line for line in branch_diff.stdout.splitlines() if line)
    return paths


def run_git(repo, *args):
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result


def init_guard_repo(repo):
    repo.mkdir()
    run_git(repo, "init", "-b", "main")
    run_git(repo, "config", "user.name", "Validator Test")
    run_git(repo, "config", "user.email", "validator@example.invalid")
    (repo / ".gitignore").write_text("ignored.bat\n", encoding="utf-8")
    (repo / "baseline.txt").write_text("baseline\n", encoding="utf-8")
    protected_file = repo / "packages_win" / "py312" / "baseline.whl"
    protected_file.parent.mkdir(parents=True)
    protected_file.write_text("baseline\n", encoding="utf-8")
    run_git(repo, "add", ".gitignore", "baseline.txt", protected_file.relative_to(repo))
    run_git(repo, "commit", "-m", "baseline")
    return protected_file


def run_configured_protected_validator(repo):
    config = json.loads(
        (ROOT / ".agents" / "agents_config.json").read_text(encoding="utf-8")
    )
    command = config["validation_commands"]["protected_diff"]
    return subprocess.run(
        ["bash", "-c", command],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )


def run_protected_guard(action, repo, state):
    result = subprocess.run(
        [
            PYTHON_312,
            PROTECTED_GUARD,
            action,
            "--root",
            repo,
            "--state",
            state,
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.stdout, result.stderr
    return result, json.loads(result.stdout)


def create_guard_candidates(repo):
    repo.mkdir(exist_ok=True)
    wheel = repo / "packages_win" / "py312" / "wheel.whl"
    wheel.parent.mkdir(parents=True)
    wheel.write_text("wheel-v1\n", encoding="utf-8")
    parquet = repo / "data" / "sample.parquet"
    parquet.parent.mkdir()
    parquet.write_bytes(b"parquet-v1")
    bat = repo / "deploy.bat"
    bat.write_bytes(b"@echo off\r\nchcp 65001\r\n")
    out_dir = repo / "out"
    out_dir.mkdir()
    symlink = out_dir / "wheel-link"
    symlink.symlink_to("../packages_win/py312/wheel.whl")
    return {
        "wheel": wheel,
        "parquet": parquet,
        "bat": bat,
        "symlink": symlink,
    }


def init_ignored_guard_repo(repo):
    repo.mkdir()
    run_git(repo, "init", "-b", "main")
    run_git(repo, "config", "user.name", "Guard Test")
    run_git(repo, "config", "user.email", "guard@example.invalid")
    (repo / ".gitignore").write_text(
        "packages_win/py312/\n*.parquet\n*.bat\nout/\n",
        encoding="utf-8",
    )
    (repo / "baseline.txt").write_text("baseline\n", encoding="utf-8")
    run_git(repo, "add", ".gitignore", "baseline.txt")
    run_git(repo, "commit", "-m", "baseline")
    return create_guard_candidates(repo)


def write_fake_cli(directory, name, body):
    fake_cli = directory / name
    fake_cli.write_text(
        "#!/usr/bin/env bash\nset -eu\n" + textwrap.dedent(body),
        encoding="utf-8",
    )
    fake_cli.chmod(0o755)
    return fake_cli


def run_adapter(
    provider,
    brief,
    fake_bin,
    timeout_seconds=2,
    kill_after_seconds=5,
    extra_env=None,
):
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"
    env["AGENT_ADAPTER_TIMEOUT"] = str(timeout_seconds)
    env["AGENT_ADAPTER_KILL_AFTER"] = str(kill_after_seconds)
    if extra_env:
        env.update({key: str(value) for key, value in extra_env.items()})

    def bounded_timeout(value, fallback):
        try:
            parsed = int(str(value))
        except (TypeError, ValueError):
            return fallback
        return parsed if 1 <= parsed <= 3600 else fallback

    provider_deadline = bounded_timeout(timeout_seconds, 1)
    kill_grace = bounded_timeout(kill_after_seconds, 5)
    return subprocess.run(
        [ADAPTER, provider, brief],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=provider_deadline + max(kill_grace, 5) + 3,
    )


def parse_envelope(result):
    assert result.stdout
    return json.loads(result.stdout)


def parse_provider_argv(envelope):
    return json.loads(envelope["stdout"])


def test_project_config_enables_only_the_custom_bridge_roles():
    with CODEX_CONFIG.open("rb") as config_file:
        config = tomllib.load(config_file)

    assert config["features"]["multi_agent"] is True
    assert config["agents"]["max_depth"] == 1
    assert config["agents"]["interrupt_message"] is True

    roles = {
        name: role
        for name, role in config["agents"].items()
        if isinstance(role, dict)
    }
    assert set(roles) == {"claude-bridge", "agy-bridge"}
    assert roles["claude-bridge"]["config_file"] == "./agents/claude-bridge.toml"
    assert roles["agy-bridge"]["config_file"] == "./agents/agy-bridge.toml"


@pytest.mark.parametrize(
    (
        "agent_path",
        "expected_name",
        "identity_boundary",
        "bridge_invocations",
        "required_gates",
    ),
    [
        (
            CLAUDE_AGENT,
            "claude-bridge",
            "You are a bridge agent, not Claude itself and not the LO.",
            (
                ".agents/adapters/call_external_agent.sh claude <brief-path>",
                ".agents/adapters/call_external_agent.sh claude-advisor <brief-path>",
            ),
            (
                "MODE_11_hana freeze",
                "protected-path",
                "HANA-schema",
                "Python 3.12",
                "BAT",
                "train-serving parity",
            ),
        ),
        (
            AGY_AGENT,
            "agy-bridge",
            "You are a bridge agent, not AGY itself and not the LO.",
            (".agents/adapters/call_external_agent.sh agy <brief-path>",),
            (
                "Stop with BLOCK or HARD_STOP",
                "Python 3.12 parity",
                "BAT CRLF/chcp 65001",
                "feature-build temp disk",
                "protected paths",
                "unconfirmed HANA schema",
                "RESEARCH_TRACK_FROZEN",
            ),
        ),
    ],
)
def test_custom_bridge_configs_are_read_only_and_report_to_codex_lo(
    agent_path,
    expected_name,
    identity_boundary,
    bridge_invocations,
    required_gates,
):
    with agent_path.open("rb") as agent_file:
        agent = tomllib.load(agent_file)

    instructions = agent["developer_instructions"]
    assert agent["name"] == expected_name
    assert agent["description"].strip()
    assert instructions.strip()
    assert agent["sandbox_mode"] == "read-only"
    assert identity_boundary in instructions
    assert (
        "Codex LO owns user communication, sequencing, decisions, verification, "
        "and final reporting."
    ) in instructions

    for invocation in bridge_invocations:
        assert invocation in instructions

    for prohibition in (
        "Do not call another subagent",
        "contact the user",
        "edit repository files",
        "claim success from model output alone",
    ):
        assert prohibition in instructions

    assert "Validate the adapter JSON envelope" in instructions
    assert (
        "return: exact files changed (normally none), exact command, validation "
        "status, risks, and one recommended next step to Codex LO."
    ) in instructions
    assert (
        "External adapter execution requires Codex LO/user-approved runtime "
        "permission."
    ) in instructions
    assert (
        "If the current runtime cannot execute the adapter, do not "
        "self-escalate, prompt or contact the user, or weaken the sandbox."
    ) in instructions
    assert (
        "Return BLOCK to Codex LO with the exact attempted command and runtime "
        "evidence."
    ) in instructions

    for gate in required_gates:
        assert gate in instructions


def test_claude_settings_uses_fable_advisor_model():
    serialized = CLAUDE_SETTINGS.read_text(encoding="utf-8")
    settings = json.loads(serialized)

    assert settings == {"advisorModel": "fable"}
    assert "permissions" not in serialized
    assert "bypassPermissions" not in serialized


def test_adapter_rejects_unknown_provider(tmp_path):
    result = run_adapter("unknown", SMOKE_BRIEF, tmp_path)

    assert result.returncode == 2
    envelope = parse_envelope(result)
    assert envelope["status"] == "error"
    assert envelope["error"] == "provider_not_allowed"


def test_adapter_rejects_brief_outside_repo(tmp_path):
    outside_brief = tmp_path / "outside.md"
    outside_brief.write_text("read-only", encoding="utf-8")

    result = run_adapter("claude", outside_brief, tmp_path)

    assert result.returncode == 2
    envelope = parse_envelope(result)
    assert envelope["status"] == "error"
    assert envelope["error"] == "brief_outside_repo"


def test_adapter_reports_tempdir_failure_as_one_safe_json(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    write_fake_cli(fake_bin, "claude", "exit 0\n")
    missing_tmpdir = tmp_path / "missing" / "tmp"
    unsafe_root_files = [
        Path("/stdout"),
        Path("/stderr"),
        Path("/stderr.sanitized"),
    ]
    assert not any(path.exists() for path in unsafe_root_files)

    result = run_adapter(
        "claude",
        SMOKE_BRIEF,
        fake_bin,
        extra_env={"TMPDIR": missing_tmpdir},
    )

    assert result.returncode == 3
    envelope = parse_envelope(result)
    assert envelope["status"] == "error"
    assert envelope["error"] == "tempdir_failed"
    assert not any(path.exists() for path in unsafe_root_files)


def test_adapter_reports_capture_setup_failure_as_one_json(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_work_dir = tmp_path / "agent-adapter.fake"
    fake_work_dir.mkdir()
    (fake_work_dir / "stdout").mkdir()
    write_fake_cli(fake_bin, "mktemp", 'printf \'%s\\n\' "$FAKE_WORK_DIR"\n')
    write_fake_cli(fake_bin, "claude", "exit 0\n")

    result = run_adapter(
        "claude",
        SMOKE_BRIEF,
        fake_bin,
        extra_env={"FAKE_WORK_DIR": fake_work_dir},
    )

    assert result.returncode == 3
    envelope = parse_envelope(result)
    assert envelope["status"] == "error"
    assert envelope["error"] == "capture_setup_failed"


def test_adapter_never_attempts_cleanup_of_canonical_root_tempdir(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    rmdir_marker = tmp_path / "rmdir-invoked"
    write_fake_cli(fake_bin, "mktemp", "printf '////\\n'\n")
    write_fake_cli(
        fake_bin,
        "rmdir",
        "printf '%s\\n' \"$@\" > \"$RMDIR_MARKER\"\n",
    )
    write_fake_cli(fake_bin, "claude", "exit 0\n")

    result = run_adapter(
        "claude",
        SMOKE_BRIEF,
        fake_bin,
        extra_env={"RMDIR_MARKER": rmdir_marker},
    )

    assert result.returncode == 3
    envelope = parse_envelope(result)
    assert envelope["error"] == "tempdir_failed"
    assert not rmdir_marker.exists()


@pytest.mark.parametrize("timeout_seconds", [0, "not-a-number", -1, 3601])
def test_adapter_rejects_invalid_timeout_without_invoking_provider(
    tmp_path,
    timeout_seconds,
):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    invoked_marker = tmp_path / "provider-invoked"
    write_fake_cli(
        fake_bin,
        "claude",
        "printf 'invoked\\n' > \"$FAKE_INVOKED_PATH\"\n",
    )

    result = run_adapter(
        "claude",
        SMOKE_BRIEF,
        fake_bin,
        timeout_seconds=timeout_seconds,
        extra_env={"FAKE_INVOKED_PATH": invoked_marker},
    )

    assert result.returncode == 2
    envelope = parse_envelope(result)
    assert envelope["status"] == "error"
    assert envelope["error"] == "invalid_timeout"
    assert not invoked_marker.exists()


def test_claude_worker_adapter_reports_kill_after_timeout(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    write_fake_cli(fake_bin, "claude", "trap '' TERM\nsleep 10\n")

    result = run_adapter(
        "claude",
        SMOKE_BRIEF,
        fake_bin,
        timeout_seconds=1,
        kill_after_seconds=1,
    )

    assert result.returncode == 137
    envelope = parse_envelope(result)
    assert envelope["status"] == "timeout"
    assert envelope["exit_code"] == 137
    assert envelope["duration_seconds"] >= 1
    assert (
        "timeout: sending signal KILL to command"
        in envelope["stderr_sanitized"]
    )


def test_claude_worker_adapter_keeps_immediate_137_as_error(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    write_fake_cli(fake_bin, "claude", "exit 137\n")

    result = run_adapter("claude", SMOKE_BRIEF, fake_bin)

    assert result.returncode == 137
    envelope = parse_envelope(result)
    assert envelope["status"] == "error"
    assert envelope["exit_code"] == 137
    assert envelope["duration_seconds"] < 2


def test_claude_worker_adapter_keeps_provider_124_as_error(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    write_fake_cli(fake_bin, "claude", "exit 124\n")

    result = run_adapter("claude", SMOKE_BRIEF, fake_bin)

    assert result.returncode == 124
    envelope = parse_envelope(result)
    assert envelope["status"] == "error"
    assert envelope["exit_code"] == 124


def test_provider_stderr_cannot_spoof_kill_timeout_provenance(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    diagnostic = "timeout: sending signal KILL to command 'bash'"
    write_fake_cli(
        fake_bin,
        "claude",
        f"printf '%s\\n' \"{diagnostic}\" >&2\nexit 137\n",
    )

    result = run_adapter("claude", SMOKE_BRIEF, fake_bin)

    assert result.returncode == 137
    envelope = parse_envelope(result)
    assert envelope["status"] == "error"
    assert envelope["exit_code"] == 137
    assert diagnostic in envelope["stderr_sanitized"]


@pytest.mark.parametrize("attempt", range(5))
def test_claude_worker_adapter_keeps_late_provider_137_as_error(
    tmp_path,
    attempt,
):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    write_fake_cli(fake_bin, "claude", "sleep 0.75\nexit 137\n")

    result = run_adapter(
        "claude",
        SMOKE_BRIEF,
        fake_bin,
        timeout_seconds=1,
    )

    assert result.returncode == 137, attempt
    envelope = parse_envelope(result)
    assert envelope["status"] == "error", attempt
    assert envelope["exit_code"] == 137


def test_claude_worker_adapter_invokes_plan_mode_json(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    write_fake_cli(fake_bin, "claude", ARGV_DUMP_BODY)

    result = run_adapter("claude", SMOKE_BRIEF, fake_bin)

    assert result.returncode == 0
    envelope = parse_envelope(result)
    assert envelope["status"] == "ok"
    assert envelope["provider"] == "claude"
    brief_content = SMOKE_BRIEF.read_text(encoding="utf-8").rstrip("\n")
    assert parse_provider_argv(envelope) == [
        "-p",
        brief_content,
        "--permission-mode",
        "plan",
        "--tools",
        "Read,Grep,Glob",
        "--output-format",
        "json",
        "--no-session-persistence",
    ]


def test_claude_advisor_adapter_invokes_advisor_once_without_fable_override(
    tmp_path,
):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    write_fake_cli(fake_bin, "claude", ARGV_DUMP_BODY)

    result = run_adapter("claude-advisor", SMOKE_BRIEF, fake_bin)

    assert result.returncode == 0
    envelope = parse_envelope(result)
    assert envelope["status"] == "ok"
    assert envelope["provider"] == "claude-advisor"
    brief_content = SMOKE_BRIEF.read_text(encoding="utf-8").rstrip("\n")
    advisor_prompt = f"{ADVISOR_INSTRUCTION}\n\n{brief_content}"
    argv = parse_provider_argv(envelope)
    assert argv == [
        "-p",
        advisor_prompt,
        "--permission-mode",
        "plan",
        "--tools",
        "advisor",
        "--settings",
        str(CLAUDE_SETTINGS.resolve()),
        "--output-format",
        "stream-json",
        "--verbose",
        "--no-session-persistence",
    ]
    assert argv[1].startswith(ADVISOR_INSTRUCTION)
    assert brief_content in argv[1]
    assert "--model" not in argv


def test_agy_worker_adapter_invokes_sandbox_plan_mode(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    write_fake_cli(fake_bin, "agy", ARGV_DUMP_BODY)

    result = run_adapter("agy", SMOKE_BRIEF, fake_bin)

    assert result.returncode == 0
    envelope = parse_envelope(result)
    assert envelope["status"] == "ok"
    assert envelope["provider"] == "agy"
    brief_content = SMOKE_BRIEF.read_text(encoding="utf-8").rstrip("\n")
    assert parse_provider_argv(envelope) == [
        "--sandbox",
        "--mode",
        "plan",
        "--add-dir",
        str(ROOT.resolve()),
        "--print",
        brief_content,
        "--print-timeout",
        "2s",
    ]


def test_claude_worker_adapter_reports_timeout(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    write_fake_cli(fake_bin, "claude", "sleep 3\n")

    result = run_adapter("claude", SMOKE_BRIEF, fake_bin, timeout_seconds=1)

    assert result.returncode == 124
    envelope = parse_envelope(result)
    assert envelope["status"] == "timeout"
    assert envelope["exit_code"] == 124


@pytest.mark.parametrize(
    ("stderr_line", "secret_value"),
    [
        ("API_KEY=topsecret", "topsecret"),
        ("api_key=lower-secret", "lower-secret"),
        ("Authorization: Bearer bearer-secret", "bearer-secret"),
        ("AWS_SECRET_ACCESS_KEY=aws-secret", "aws-secret"),
        ('{"TOKEN":"json-secret"}', "json-secret"),
        ("PASSWORD=upper-secret", "upper-secret"),
        ("request failed for sk-secret-token", "sk-secret-token"),
    ],
)
def test_claude_worker_adapter_redacts_secret_stderr(
    tmp_path,
    stderr_line,
    secret_value,
):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    write_fake_cli(
        fake_bin,
        "claude",
        "printf '%s\\n' \"$FAKE_STDERR\" >&2\nexit 7\n",
    )

    result = run_adapter(
        "claude",
        SMOKE_BRIEF,
        fake_bin,
        extra_env={"FAKE_STDERR": stderr_line},
    )

    assert result.returncode == 7
    envelope = parse_envelope(result)
    assert envelope["status"] == "error"
    assert secret_value not in envelope["stderr_sanitized"]
    assert "[REDACTED]" in envelope["stderr_sanitized"]
    assert result.stderr == ""


def test_claude_worker_adapter_fails_closed_when_redaction_fails(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    secret_value = "redaction-must-not-leak"
    write_fake_cli(fake_bin, "sed", "exit 9\n")
    write_fake_cli(
        fake_bin,
        "claude",
        "printf 'API_KEY=%s\\n' \"$FAKE_SECRET\" >&2\nexit 7\n",
    )

    result = run_adapter(
        "claude",
        SMOKE_BRIEF,
        fake_bin,
        extra_env={"FAKE_SECRET": secret_value},
    )

    assert result.returncode == 3
    envelope = parse_envelope(result)
    assert envelope["status"] == "error"
    assert envelope["error"] == "redaction_failed"
    assert secret_value not in result.stdout
    assert result.stderr == ""


def test_adapter_reports_missing_sed_dependency_as_one_json(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    for command in ("bash", "jq", "timeout", "realpath"):
        executable = shutil.which(command)
        assert executable
        (fake_bin / command).symlink_to(executable)
    write_fake_cli(fake_bin, "claude", "exit 0\n")

    result = run_adapter(
        "claude",
        SMOKE_BRIEF,
        fake_bin,
        extra_env={"PATH": fake_bin},
    )

    assert result.returncode == 3
    envelope = parse_envelope(result)
    assert envelope["status"] == "error"
    assert envelope["error"] == "dependency_missing"
    assert envelope["dependency"] == "sed"


def test_codex_is_the_only_active_lo():
    policy_files = active_policy_files()
    active_content = "\n".join(
        path.read_text(encoding="utf-8") for path in policy_files
    )
    forbidden_authorities = (
        "Open" + "Code",
        "Hermes" + " LO",
        "Claude" + " LO",
    )

    assert "Codex LO" in active_content
    for authority in forbidden_authorities:
        assert authority.casefold() not in active_content.casefold()


def test_only_approved_active_agent_definitions_remain():
    assert files_under(ROOT / ".agents") == EXPECTED_AGENTS_FILES
    assert files_under(ROOT / ".codex") == EXPECTED_CODEX_FILES
    claude_agents_dir = ROOT / ".claude" / "agents"
    assert files_under(claude_agents_dir) == {".gitkeep"}
    assert not any(
        path.suffix.casefold() == ".md"
        for path in claude_agents_dir.rglob("*")
        if path.is_file()
    )

    expected_active_files = {
        ROOT / "AGENTS.md",
        ROOT / "CLAUDE.md",
        CLAUDE_SETTINGS,
        claude_agents_dir / ".gitkeep",
        *(ROOT / ".agents" / path for path in EXPECTED_AGENTS_FILES),
        *(ROOT / ".codex" / path for path in EXPECTED_CODEX_FILES),
    }
    assert set(active_policy_files()) == expected_active_files


def test_repo_local_orchestration_allowlist_is_narrow():
    approved_adapters = {
        "adapters/call_external_agent.sh",
        "adapters/protected_artifact_guard.py",
    }
    assert approved_adapters <= EXPECTED_AGENTS_FILES
    assert files_under(ROOT / ".claude" / "agents") == {".gitkeep"}

    for path in (
        *(f".agents/{adapter}" for adapter in approved_adapters),
        ".claude/agents/.gitkeep",
        ".claude/settings.json",
    ):
        result = subprocess.run(
            ["git", "check-ignore", "--quiet", "--", path],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 1, (
            f"expected {path} not to be ignored; "
            f"stdout={result.stdout!r}, stderr={result.stderr!r}"
        )

    result = subprocess.run(
        [
            "git",
            "check-ignore",
            "--quiet",
            "--",
            ".agents/adapters/unapproved.tmp",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, (
        "expected arbitrary adapter contents to remain ignored; "
        f"stdout={result.stdout!r}, stderr={result.stderr!r}"
    )

    rogue_agent = subprocess.run(
        [
            "git",
            "check-ignore",
            "--quiet",
            "--",
            ".claude/agents/rogue.md",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert rogue_agent.returncode == 0, (
        "expected arbitrary Claude agent definitions to remain ignored; "
        f"stdout={rogue_agent.stdout!r}, stderr={rogue_agent.stderr!r}"
    )

    local_settings = subprocess.run(
        [
            "git",
            "check-ignore",
            "--quiet",
            "--",
            ".claude/settings.local.json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert local_settings.returncode == 0, (
        "expected machine-local Claude settings to remain ignored; "
        f"stdout={local_settings.stdout!r}, stderr={local_settings.stderr!r}"
    )
    assert CLAUDE_LOCAL_SETTINGS not in active_policy_files()


def test_agents_config_matches_runtime_topology():
    config_path = ROOT / ".agents" / "agents_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    with CODEX_CONFIG.open("rb") as config_file:
        codex_config = tomllib.load(config_file)

    assert config["version"] == "2.0.0"
    lo_config = config["lo_configuration"]
    assert lo_config["lo_agent"] == "Codex"
    assert lo_config["max_outbound_external_worker_requests"] == 1
    assert lo_config["max_subagent_depth"] == codex_config["agents"][
        "max_depth"
    ]
    assert lo_config.get("outbound_request_enforcement") == (
        "Enforced by Codex LO dispatch and sequential policy; this is not a "
        "global adapter lock."
    )
    assert config["runtime_approval_boundary"] == {
        "bridge_sandbox": "strict read-only",
        "external_adapter_execution": (
            "Requires Codex LO/user-approved runtime permission; execution is "
            "not guaranteed inside the strict read-only bridge sandbox."
        ),
        "fail_closed": (
            "If the runtime cannot execute the adapter, the bridge must not "
            "self-escalate, prompt or contact the user, or weaken its sandbox; "
            "it returns BLOCK to Codex LO with the exact attempted command and "
            "runtime evidence."
        ),
        "observed_probe": (
            "A local Codex strict :read-only probe blocked mktemp /tmp as "
            "expected; parent approval/runtime permission is therefore a "
            "prerequisite for live adapter dispatch."
        ),
    }
    assert set(config["external_workers"]) == {"Claude Code", "AGY"}
    advisor = config["advisor"]
    assert advisor["provider"] == "Claude Code"
    assert advisor["mechanism"] == "Claude Code built-in advisor"
    assert advisor["model_alias"] == "fable"
    assert advisor["resolved_model"] == "claude-fable-5"
    assert advisor["default_gates"] == ["plan", "finish"]
    assert advisor["fresh_session_per_call"] is True
    assert advisor["max_calls_per_session"] == 1

    expected_handlers = {
        "claude": {
            "bridge": ".codex/agents/claude-bridge.toml",
            "adapter": ".agents/adapters/call_external_agent.sh claude",
            "default_access": "read-only",
        },
        "claude-advisor": {
            "bridge": ".codex/agents/claude-bridge.toml",
            "adapter": (
                ".agents/adapters/call_external_agent.sh claude-advisor"
            ),
            "default_access": "read-only",
        },
        "agy": {
            "bridge": ".codex/agents/agy-bridge.toml",
            "adapter": ".agents/adapters/call_external_agent.sh agy",
            "default_access": "read-only sandbox/plan",
        },
    }
    assert set(config["actual_handlers"]) == set(expected_handlers)
    for name, expected in expected_handlers.items():
        handler = config["actual_handlers"][name]
        assert {
            key: handler[key] for key in expected
        } == expected
        assert (ROOT / handler["bridge"]).is_file()
        adapter_path = handler["adapter"].split(maxsplit=1)[0]
        assert (ROOT / adapter_path).is_file()

    assert ".claude/settings.json" in (ROOT / "CLAUDE.md").read_text(
        encoding="utf-8"
    )
    forbidden_provider = "Open" + "Code"
    assert forbidden_provider.casefold() not in json.dumps(config).casefold()


def test_hard_gates_remain_in_active_policy():
    config_path = ROOT / ".agents" / "agents_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected_hard_gates = {
        "final_dataset": (
            "Final dataset is fixed: 2024-07..12 Raw, 184 daily "
            "records_YYYYMMDD.parquet files plus 500k eligibility; no "
            "2025-01 acquisition."
        ),
        "hana_schema": (
            "HANA schema/table/column names must never be guessed; use "
            "confirmed sources only."
        ),
        "environment": (
            "Windows production is closed-network; Python 3.12 dev/prod "
            "parity is required."
        ),
        "shared_constants": (
            "_PID_BATCH_T30 and strata_utils._DEFAULT_AGE_BINS must not be "
            "redefined."
        ),
        "python_runtime": "Python 3.12 only",
        "bat_files": (
            "BAT files must preserve CRLF and include chcp 65001; AGY "
            "sign-off is required."
        ),
        "feature_temp_preflight": (
            "Temp destination priority: HANA_FEAT_TMP -> HANA_TMP_DIR -> "
            "hana_config.json -> system temp; require 10GB+ free space."
        ),
        "train_serving_parity": (
            "RequestFeatureBuilder feature names and order must match "
            "training exactly; serving changes require a training schema "
            "diff, tests/test_serving, tests/test_features, /reload, and "
            "sample payload sanity checks."
        ),
        "protected_paths": [
            "packages_win/py312/",
            "mlruns/",
            "**/*.parquet",
            "out/",
        ],
        "protected_path_policy": (
            "Protected paths may not be modified, deleted, or committed "
            "without explicit user approval."
        ),
        "research_track": (
            "RESEARCH_TRACK_FROZEN; no Nov->Dec tuning, ablation, feature, "
            "or hyperparameter work; Gate 5A, Gate 5B, and 2025-01 are "
            "retired."
        ),
    }
    assert config["hard_gates"] == expected_hard_gates

    common_primary_tokens = (
        "2024-07..12",
        "184 daily",
        "500k eligibility",
        "2025-01",
        "RESEARCH_TRACK_FROZEN",
        "Gate 5A",
        "Gate 5B",
        "HANA",
        "RequestFeatureBuilder",
        "tests/test_serving",
        "tests/test_features",
        "/reload",
        "sample payload sanity",
        "Python 3.12",
        "HANA_FEAT_TMP",
        "10GB+",
        "CRLF",
        "chcp 65001",
        "packages_win/py312/",
        "mlruns/",
        "parquet",
        "out/",
    )
    tokens_by_file = {
        ROOT / "AGENTS.md": common_primary_tokens
        + (
            "must not be guessed",
            "canceled/retired",
            "_PID_BATCH_T30",
            "strata_utils._DEFAULT_AGE_BINS",
        ),
        ROOT / ".agents" / "codex_hq.md": common_primary_tokens
        + (
            "Never guess HANA",
            "are retired",
            "_PID_BATCH_T30",
            "strata_utils._DEFAULT_AGE_BINS",
            "protected_snapshot",
            "protected_verify",
            "Workers may not refresh",
        ),
        ROOT / ".agents" / "claude_hq.md": (
            "2024-07..12",
            "184 daily",
            "500k eligibility",
            "2025-01",
            "RESEARCH_TRACK_FROZEN",
            "Gate 5A",
            "Gate 5B",
            "Never guess HANA",
            "RequestFeatureBuilder",
            "tests/test_serving",
            "tests/test_features",
            "/reload",
            "sample payload sanity",
            "Python 3.12",
            "HANA_FEAT_TMP",
            "10GB+",
            "CRLF",
            "chcp 65001",
            "packages_win/py312/",
            "mlruns/",
            "parquet",
            "out/",
            "Structured return",
        ),
        ROOT / ".agents" / "agy_hq.md": (
            "2024-07..12",
            "184 daily",
            "500k eligibility",
            "2025-01",
            "RESEARCH_TRACK_FROZEN",
            "Gate 5A",
            "Gate 5B",
            "Never guess HANA",
            "confirmed sources",
            "RequestFeatureBuilder",
            "tests/test_serving",
            "tests/test_features",
            "/reload",
            "sample payload sanity",
            "Python 3.12",
            "Windows offline",
            "HANA_FEAT_TMP",
            "10GB+",
            "CRLF",
            "chcp 65001",
            "packages_win/py312/",
            "mlruns/",
            "parquet",
            "out/",
            "Structured return",
            "protected_snapshot",
            "protected_verify",
            "cannot refresh",
        ),
    }
    for policy_file, required_tokens in tokens_by_file.items():
        content = policy_file.read_text(encoding="utf-8")
        for token in required_tokens:
            assert token in content, f"{policy_file}: missing {token!r}"


def test_configured_policy_validator_is_fail_closed(tmp_path):
    config = json.loads(
        (ROOT / ".agents" / "agents_config.json").read_text(encoding="utf-8")
    )
    command = config["validation_commands"]["active_policy_search"]
    tracked_marker = ROOT / ".claude" / "agents" / ".gitkeep"
    assert ".claude/agents" in command
    assert tracked_marker.is_file()
    assert tracked_marker.read_bytes() == b""

    def create_policy_tree(directory, include_settings=True):
        for relative in (".agents", ".codex", ".claude/agents"):
            (directory / relative).mkdir(parents=True, exist_ok=True)
        (directory / ".claude" / "agents" / ".gitkeep").write_bytes(b"")
        for relative in ("AGENTS.md", "CLAUDE.md"):
            (directory / relative).write_text("Codex LO\n", encoding="utf-8")
        if include_settings:
            (directory / ".claude" / "settings.json").write_text(
                "{}\n", encoding="utf-8"
            )

    clean_root = tmp_path / "clean"
    create_policy_tree(clean_root)
    clean = subprocess.run(
        ["bash", "-c", command],
        cwd=clean_root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert clean.returncode == 0, clean.stderr

    forbidden_root = tmp_path / "forbidden"
    create_policy_tree(forbidden_root)
    forbidden_authority = "Open" + "Code"
    (forbidden_root / ".claude" / "agents" / "rogue.md").write_text(
        forbidden_authority, encoding="utf-8"
    )
    forbidden = subprocess.run(
        ["bash", "-c", command],
        cwd=forbidden_root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert forbidden.returncode == 1

    error_root = tmp_path / "error"
    create_policy_tree(error_root, include_settings=False)
    errored = subprocess.run(
        ["bash", "-c", command],
        cwd=error_root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert errored.returncode > 1


def test_configured_current_change_guard_covers_protected_and_bat_paths(
):
    config = json.loads(
        (ROOT / ".agents" / "agents_config.json").read_text(encoding="utf-8")
    )
    command = config["validation_commands"]["protected_diff"]
    for required in (
        "git diff --name-only HEAD",
        "git ls-files --others --exclude-standard",
        "git show-ref --verify --quiet refs/heads/main",
        "main...HEAD",
        "packages_win/py312/",
        "mlruns/",
        "out/",
        "*.parquet",
        "*.bat",
    ):
        assert required in command
    assert "git ls-files --others --ignored --exclude-standard" not in command
    assert "|| true" not in command

    clean = subprocess.run(
        ["bash", "-c", command],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert clean.returncode == 0, clean.stdout + clean.stderr



@pytest.mark.parametrize(
    ("scenario", "expected_path"),
    [
        ("clean", None),
        ("tracked_unstaged", "packages_win/py312/baseline.whl"),
        ("staged", "packages_win/py312/baseline.whl"),
        ("untracked_nonignored", "new.bat"),
        ("preexisting_ignored", None),
        ("committed_feature", "out/committed.txt"),
    ],
)
def test_protected_validator_detects_isolated_repo_states(
    tmp_path,
    scenario,
    expected_path,
):
    repo = tmp_path / scenario
    protected_file = init_guard_repo(repo)

    if scenario in {"tracked_unstaged", "staged"}:
        protected_file.write_text("changed\n", encoding="utf-8")
        if scenario == "staged":
            run_git(repo, "add", protected_file.relative_to(repo))
    elif scenario == "untracked_nonignored":
        (repo / "new.bat").write_bytes(b"@echo off\r\nchcp 65001\r\n")
    elif scenario == "preexisting_ignored":
        (repo / "ignored.bat").write_bytes(b"@echo off\r\nchcp 65001\r\n")
        run_git(repo, "check-ignore", "ignored.bat")
    elif scenario == "committed_feature":
        run_git(repo, "switch", "-c", "feature")
        committed_file = repo / "out" / "committed.txt"
        committed_file.parent.mkdir()
        committed_file.write_text("committed\n", encoding="utf-8")
        run_git(repo, "add", committed_file.relative_to(repo))
        run_git(repo, "commit", "-m", "protected feature change")

    result = run_configured_protected_validator(repo)
    if expected_path is None:
        assert result.returncode == 0, result.stdout + result.stderr
    else:
        assert result.returncode == 1, result.stdout + result.stderr
        assert expected_path in result.stdout


def test_protected_validator_fails_closed_without_main_merge_base(tmp_path):
    repo = tmp_path / "unrelated"
    init_guard_repo(repo)
    run_git(repo, "switch", "--orphan", "unrelated")
    run_git(repo, "commit", "--allow-empty", "-m", "unrelated root")

    result = run_configured_protected_validator(repo)

    assert result.returncode > 1
    assert "no merge base" in result.stderr.casefold()


def test_changed_bat_files_preserve_windows_contract():
    for relative in changed_paths(BAT_PATHSPECS):
        bat_file = ROOT / relative
        assert bat_file.is_file(), f"changed BAT was deleted: {relative}"
        content = bat_file.read_bytes()
        without_crlf = content.replace(b"\r\n", b"")
        assert b"\r" not in without_crlf and b"\n" not in without_crlf, (
            f"changed BAT must use CRLF-only line endings: {relative}"
        )
        assert re.search(br"(?i)\bchcp[ \t]+65001\b", content), (
            f"changed BAT must include chcp 65001: {relative}"
        )


def test_current_change_set_excludes_protected_artifacts():
    assert changed_paths(PROTECTED_PATHSPECS) == set()


def test_configured_artifact_guard_commands_are_exact():
    config = json.loads(
        (ROOT / ".agents" / "agents_config.json").read_text(encoding="utf-8")
    )
    commands = config["validation_commands"]
    command_prefix = (
        f"{PYTHON_312} .agents/adapters/protected_artifact_guard.py"
    )
    state_argument = '"${TMPDIR:-/tmp}/mode11-protected-baseline.json"'

    assert commands.get("protected_snapshot") == (
        f"{command_prefix} snapshot --root . --state {state_argument}"
    )
    assert commands.get("protected_verify") == (
        f"{command_prefix} verify --root . --state {state_argument}"
    )
    assert PROTECTED_GUARD.is_file()
    assert b"\r\n" not in PROTECTED_GUARD.read_bytes()


def test_protected_artifact_guard_snapshot_then_verify(tmp_path):
    repo = tmp_path / "repo"
    candidates = init_ignored_guard_repo(repo)
    state = tmp_path / "guard-state.json"

    snapshot, snapshot_payload = run_protected_guard(
        "snapshot", repo, state
    )
    verify, verify_payload = run_protected_guard("verify", repo, state)

    assert snapshot.returncode == 0
    assert snapshot_payload["status"] == "ok"
    assert snapshot_payload["action"] == "snapshot"
    assert state.stat().st_mode & 0o777 == 0o600
    manifest = json.loads(state.read_text(encoding="utf-8"))
    assert manifest["version"] == 1
    wheel_entry = manifest["entries"][
        candidates["wheel"].relative_to(repo).as_posix()
    ]
    assert set(wheel_entry) == {"mode", "size", "mtime_ns"}
    symlink_entry = manifest["entries"][
        candidates["symlink"].relative_to(repo).as_posix()
    ]
    assert symlink_entry["symlink_target"] == (
        "../packages_win/py312/wheel.whl"
    )
    assert verify.returncode == 0
    assert verify_payload["status"] == "ok"
    assert verify_payload["action"] == "verify"


@pytest.mark.parametrize(
    ("change", "change_class", "expected_path"),
    [
        ("modify", "modified", "packages_win/py312/wheel.whl"),
        ("add", "added", "out/new.bin"),
        ("delete", "removed", "deploy.bat"),
    ],
)
def test_protected_artifact_guard_reports_changes(
    tmp_path,
    change,
    change_class,
    expected_path,
):
    repo = tmp_path / "repo"
    candidates = init_ignored_guard_repo(repo)
    state = tmp_path / "guard-state.json"
    snapshot, _ = run_protected_guard("snapshot", repo, state)
    assert snapshot.returncode == 0

    if change == "modify":
        candidates["wheel"].write_text(
            "wheel-version-two-longer\n", encoding="utf-8"
        )
    elif change == "add":
        (repo / expected_path).write_text("new\n", encoding="utf-8")
    else:
        candidates["bat"].unlink()

    verify, payload = run_protected_guard("verify", repo, state)

    assert verify.returncode == 1
    assert payload["status"] == "changed"
    assert expected_path in payload[change_class]


def test_protected_artifact_guard_prunes_nested_worktrees_and_venvs(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    excluded_files = (
        repo / ".git" / "out" / "ignored.bin",
        repo / ".worktrees" / "nested" / "ignored.bat",
        repo / ".venv" / "data" / "ignored.parquet",
        repo / ".venv-test" / "out" / "ignored.bin",
    )
    for excluded in excluded_files:
        excluded.parent.mkdir(parents=True, exist_ok=True)
        excluded.write_text("before\n", encoding="utf-8")
    state = tmp_path / "guard-state.json"
    snapshot, _ = run_protected_guard("snapshot", repo, state)
    assert snapshot.returncode == 0

    for excluded in excluded_files:
        excluded.write_text("after and longer\n", encoding="utf-8")
    verify, payload = run_protected_guard("verify", repo, state)

    assert verify.returncode == 0
    assert payload["status"] == "ok"


def test_protected_artifact_guard_rejects_state_inside_root(tmp_path):
    repo = tmp_path / "repo"
    create_guard_candidates(repo)

    result, payload = run_protected_guard(
        "snapshot", repo, repo / "guard-state.json"
    )

    assert result.returncode > 1
    assert payload["status"] == "error"
    assert payload["error"] == "state_inside_root"


def test_protected_artifact_guard_fails_closed_for_bad_state(tmp_path):
    repo = tmp_path / "repo"
    create_guard_candidates(repo)
    missing_state = tmp_path / "missing.json"
    missing, missing_payload = run_protected_guard(
        "verify", repo, missing_state
    )
    malformed_state = tmp_path / "malformed.json"
    malformed_state.write_text("not-json\n", encoding="utf-8")
    malformed, malformed_payload = run_protected_guard(
        "verify", repo, malformed_state
    )

    assert missing.returncode > 1
    assert missing_payload["status"] == "error"
    assert missing_payload["error"] == "missing_state"
    assert malformed.returncode > 1
    assert malformed_payload["status"] == "error"
    assert malformed_payload["error"] == "malformed_state"


def test_ignored_artifact_baseline_complements_git_validator(tmp_path):
    repo = tmp_path / "repo"
    init_guard_repo(repo)
    ignored_bat = repo / "ignored.bat"
    ignored_bat.write_bytes(b"@echo off\r\nchcp 65001\r\n")
    run_git(repo, "check-ignore", "ignored.bat")
    state = tmp_path / "guard-state.json"

    git_before = run_configured_protected_validator(repo)
    snapshot, _ = run_protected_guard("snapshot", repo, state)
    verify_before, _ = run_protected_guard("verify", repo, state)

    assert git_before.returncode == 0
    assert snapshot.returncode == 0
    assert verify_before.returncode == 0

    ignored_bat.write_bytes(
        b"@echo off\r\nchcp 65001\r\necho changed\r\n"
    )
    git_after = run_configured_protected_validator(repo)
    verify_after, payload = run_protected_guard("verify", repo, state)

    assert git_after.returncode == 0
    assert verify_after.returncode == 1
    assert "ignored.bat" in payload["modified"]
