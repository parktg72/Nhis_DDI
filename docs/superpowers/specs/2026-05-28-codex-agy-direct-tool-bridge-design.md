# Codex-AGY Direct Tool Bridge Design

Date: 2026-05-28

## Goal

Make the Codex and AGY integration reliable as a direct request/response bridge:

- Codex can call AGY and immediately receive AGY's answer.
- AGY can call Codex and immediately receive Codex's answer.
- The existing Claude-AGY path remains untouched.

This is not a push chat, polling watcher, or shared live-session injection design.

## Non-Goals

- Do not change the current Claude-AGY behavior.
- Do not edit `agy-claude-mcp.ts`.
- Do not add Claude queue, channel, or watcher behavior for this work.
- Do not make Codex or AGY receive terminal push input into an already-open interactive session.

## Current Components

`codex-mcp.ts` exposes `send_to_agy` to Codex. It launches AGY in headless print mode and returns AGY's stdout as the tool result.

`agy-codex-mcp.ts` exposes `send_to_codex` to AGY. It launches `codex exec` in isolated non-interactive mode and returns Codex's final message.

`/home/ptg/.gemini/antigravity-cli/plugins/codex-bridge/mcp_config.json` registers the AGY-side MCP plugin for `send_to_codex`.

## Architecture

Codex to AGY:

1. Codex calls `send_to_agy(message, cd?, timeout_ms?)`.
2. `codex-mcp.ts` validates the message.
3. It runs AGY with `--dangerously-skip-permissions`, `--print-timeout`, and `--print`.
4. The prompt instructs AGY to return only the response intended for Codex.
5. The MCP tool returns AGY's output to Codex.

AGY to Codex:

1. AGY calls `send_to_codex(message, cd?, model?, timeout_ms?)`.
2. `agy-codex-mcp.ts` validates the message.
3. It runs `codex exec` with isolated config, workspace cwd, and `--output-last-message`.
4. The prompt instructs Codex to return only the response intended for AGY.
5. The MCP tool returns Codex's final message to AGY.

## Configuration Rules

- Keep AGY-side registration focused on `agy-codex-mcp.ts`.
- Keep `send_to_codex` eager/background enabled only if AGY needs to call it from normal assistant turns without extra prompting.
- Use `codex exec --ignore-user-config` from AGY to avoid recursive user MCP loading.
- Keep Codex-side `send_to_agy` authenticated through the existing AGY binary/wrapper and keyring preflight path.
- Pass explicit working directories from callers when the task depends on a repo.

## Error Handling

- Empty messages return structured MCP errors.
- AGY auth/keyring failures should fail before triggering repeated OAuth loops.
- Subprocess non-zero exits include bounded stderr/stdout details.
- Timeouts kill the child process and return an explicit failure.
- Empty subprocess output is treated as an error, not a successful blank reply.

## Verification

The implementation is correct when:

- Codex `send_to_agy` returns an AGY response.
- AGY `send_to_codex` returns a Codex response.
- MCP framing tests pass for both bridge servers.
- No Claude-AGY files or settings change.
- The AGY plugin config still points at `agy-codex-mcp.ts`.

## Implementation Scope

Inspect and adjust only Codex-AGY specific files and settings:

- `/home/ptg/codex-claude-bridge/codex-mcp.ts`
- `/home/ptg/codex-claude-bridge/agy-codex-mcp.ts`
- `/home/ptg/.gemini/antigravity-cli/plugins/codex-bridge/mcp_config.json`
- Codex MCP config only if needed to expose `send_to_agy`

Claude-AGY files are explicitly out of scope.
