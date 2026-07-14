# Global Codex LO Multiagent Design

**Date:** 2026-07-14  
**Status:** Approved for implementation planning  
**Scope:** User-level Codex configuration on this machine

## Goal

Make Codex the sole LO in every newly started local Codex session, regardless of the working directory, with Claude Code and AGY available as bounded external subagents and Claude Code's built-in Fable 5 advisor available as a one-shot advisory mode.

The global layer must remain project-neutral. MODE_11_hana-specific HANA schema, frozen-research, Python 3.12, BAT, protected-artifact, and train-serving rules remain local to this repository and strengthen the global defaults when Codex runs here.

## Non-goals

- Do not copy MODE_11_hana policy into unrelated projects.
- Do not make Claude Code, Fable, or AGY an LO.
- Do not dispatch a subagent for every prompt automatically.
- Do not install OpenCode or add it to the agent pool.
- Do not bypass Codex approvals, sandbox controls, provider authentication, or network policy.
- Do not modify a target project's files merely because Codex starts there.
- Do not push or publish the global configuration.

## Selected approach

Use a global base configuration under `~/.codex` and retain project-local configuration as the specialization layer.

This is preferred over an opt-in Codex profile because the user wants ordinary `codex` launches to enable the topology automatically. It is preferred over copying files into every project because per-project installation is not universal and creates drift.

## Configuration hierarchy

### Global base

The existing `~/.codex/config.toml` remains the user-level base. Installation adds only:

- `[features].multi_agent = true`
- `[agents].max_depth = 1`
- `[agents].interrupt_message = true`
- `agents.claude-bridge`
- `agents.agy-bridge`

Existing model, reasoning, project trust, and unrelated settings are preserved byte-for-byte where possible and semantically unchanged otherwise.

The global agent entries use absolute paths under `~/.codex/agents/` so they do not depend on the directory from which Codex is launched.

### Project specialization

Project `.codex/config.toml` and `AGENTS.md` files remain authoritative for project-specific behavior. MODE_11_hana keeps its current agent names and local bridge profiles. The implementation must verify the installed Codex version's effective merge behavior from both this repository and a neutral temporary project before accepting the global install.

If the effective configuration does not allow the local entries to replace the same global agent keys cleanly, the installer must stop without changing `~/.codex`; it must not create duplicate global/local Claude or AGY agents as a workaround.

## Installed components

The user-level installation consists of:

```text
~/.codex/config.toml
~/.codex/agents/claude-bridge.toml
~/.codex/agents/agy-bridge.toml
~/.codex/multiagent/call_external_agent.py
~/.codex/multiagent/claude-advisor-settings.json
~/.codex/multiagent/manifest.json
```

The repository contains a reproducible installer and tests:

```text
tools/codex_global_multiagent/install.py
tools/codex_global_multiagent/templates/claude-bridge.toml
tools/codex_global_multiagent/templates/agy-bridge.toml
tools/codex_global_multiagent/templates/call_external_agent.py
tools/codex_global_multiagent/templates/claude-advisor-settings.json
tests/test_agents/test_global_codex_multiagent.py
```

`manifest.json` records installer version, installed file hashes, and the pre-install configuration backup path. It contains no credentials.

## Agent authority and routing

Codex is the only agent that communicates with the user, decomposes work, approves dispatch, sequences workers, resolves conflicts, verifies results, and reports completion.

Claude Code handles bounded requirements, architecture, logical review, and final QA. Its advisor mode calls the built-in advisor exactly once in a fresh no-persistence session, using `advisorModel: "fable"` through the dedicated settings file. The adapter never invokes Fable as a direct `--model` value.

AGY handles bounded environment, deployment, dependency, and operational-risk checks. It is not an implementation owner.

Both bridges are non-recursive, do not contact the user, do not call each other, and return a structured result to Codex LO. Only one external worker may be in flight at a time.

## Workspace resolution

The global adapter receives a provider, a self-contained brief path, and the active workspace path.

1. Canonicalize the active workspace.
2. When the workspace is inside a Git worktree, resolve the worktree root with `git rev-parse --show-toplevel`.
3. Otherwise use the canonical Codex working directory as the workspace root.
4. Canonicalize the brief and require it to be a regular file below that workspace root.
5. Reject symlink escapes, missing files, extra arguments, unknown providers, and an empty brief before launching a provider.

Codex LO is responsible for supplying the existing brief file. A bridge must not create or edit the brief. This keeps bridge behavior read-only and makes the dispatched scope auditable.

## External adapter

The adapter is implemented with the Python standard library so capture, timeout, and JSON serialization do not require repository or `/tmp` writes.

Allowed providers are exactly:

- `claude`
- `claude-advisor`
- `agy`

Provider commands retain these boundaries:

- Claude worker: plan permission mode, `Read,Grep,Glob`, JSON output, no session persistence.
- Claude advisor: plan permission mode, only the built-in `advisor` tool, explicit dedicated Fable settings, stream JSON, no session persistence, exactly one advisor request in the prompt contract.
- AGY: sandbox and plan mode, active workspace added explicitly, bounded print timeout.

The adapter uses argument arrays without shell interpolation, a bounded timeout with explicit timeout provenance, in-memory stdout/stderr capture, fail-closed stderr secret redaction, and one JSON envelope. Raw provider stdout remains an explicitly documented trust boundary and is returned only to Codex LO for validation.

The adapter does not weaken the caller's sandbox. If provider execution requires network, home-directory state, or another permission unavailable to the bridge, it returns `BLOCK` evidence to Codex LO. Only Codex LO may request user approval and retry.

## Global Fable configuration

`~/.codex/multiagent/claude-advisor-settings.json` contains exactly:

```json
{
  "advisorModel": "fable"
}
```

This dedicated file avoids modifying or replacing the user's global `~/.claude/settings.json`. Claude worker mode continues to use normal Claude project/user settings discovery; only advisor mode passes the dedicated settings file explicitly.

## Installation and rollback

Installation is a two-phase operation.

### Phase 1: isolated validation

1. Build a temporary `CODEX_HOME` fixture.
2. Copy the user's current global config into the fixture.
3. Merge the proposed global keys into the fixture.
4. Install template files into the fixture.
5. Parse every TOML and JSON file.
6. Run mock Claude, advisor, and AGY adapter tests.
7. Run Codex strict configuration diagnostics from a neutral temporary project.
8. Run Codex diagnostics from MODE_11_hana and verify local specialization wins without duplicate agents.

Any failure ends before the real home is changed.

### Phase 2: atomic user-level install

1. Create a permission-restricted backup directory under `/tmp`.
2. Back up the current `~/.codex/config.toml` and any colliding managed files.
3. Write proposed files to sibling temporary paths under `~/.codex`.
4. Parse and hash the proposed files.
5. Replace only the managed destinations atomically.
6. Write `manifest.json` last.
7. Run the same neutral-project and MODE_11_hana configuration diagnostics against the installed files.

If post-install validation fails, restore all backed-up destinations and remove only files created by this installer. User-owned unrelated files remain untouched.

Re-running the installer is idempotent when managed file hashes and configuration values already match. If a managed destination has changed since installation, the installer stops and reports a conflict instead of overwriting it.

## Testing

Automated tests cover:

- preservation of existing model, reasoning, trust, MCP, plugin, and unrelated global configuration;
- rejection of malformed or conflicting global configuration;
- idempotent installation;
- rollback after injected failures at every atomic-install stage;
- absolute global agent paths;
- generic bridge text with no MODE_11_hana, HANA, frozen-track, BAT, protected-artifact, or OpenCode policy leakage;
- Codex-only LO authority and non-recursive sequential worker rules;
- Git and non-Git workspace resolution;
- brief containment and symlink-escape rejection;
- provider allowlisting, argv construction, bounded timeout, timeout provenance, redaction, and JSON envelope behavior;
- one-shot built-in Fable invocation without a direct model override;
- neutral-directory global loading;
- MODE_11_hana local specialization without duplicate agent registration;
- no change to repository protected artifacts or pre-existing user worktree changes.

Live provider calls are not required to prove installation correctness. If run, Claude, Fable advisor, and AGY smokes require separate Codex LO approval and must remain sequential.

## Startup behavior

After successful installation, a new CLI session started as follows loads the global base:

```bash
cd /path/to/any/project-or-directory
codex
```

"New session" means exiting the current Codex process and launching `codex` again. Existing sessions are not assumed to hot-reload global or project configuration.

The topology is available globally, but dispatch is demand-driven: Codex LO decides when a task benefits from Claude Code, the Fable advisor, or AGY. Provider binaries, authentication, runtime permissions, and network access must still be available when a worker is actually invoked.

## Acceptance criteria

The installation is accepted only when all of the following are demonstrated with fresh evidence:

1. `codex` strict configuration validation succeeds in a neutral Git project and a neutral non-Git directory.
2. `multi_agent` is enabled in those neutral locations.
3. Exactly one Claude bridge and one AGY bridge are registered globally.
4. Codex remains the only LO in all global bridge instructions.
5. The global files contain no MODE_11_hana-specific policy or absolute project path.
6. MODE_11_hana still loads its local hard gates and does not expose duplicate bridge roles.
7. Mock provider tests pass without repository or `/tmp` capture files.
8. Existing `~/.codex/config.toml` settings remain intact.
9. The install is idempotent and the tested rollback restores the pre-install state.
10. No protected repository artifact, BAT file, or pre-existing user change is modified.
