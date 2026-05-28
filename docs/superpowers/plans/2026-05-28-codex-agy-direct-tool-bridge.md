# Codex-AGY Direct Tool Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden the Codex-AGY direct MCP request/response bridge without changing the existing Claude-AGY path.

**Architecture:** Keep Codex -> AGY on Codex's `send_to_agy` tool and AGY -> Codex on AGY's `send_to_codex` tool. Add focused regression checks for command ordering, config registration, isolation flags, auth/keyring behavior, and real smoke coverage.

**Tech Stack:** Bun, TypeScript MCP servers, Node test runner, AGY CLI, Codex CLI, AGY plugin JSON config.

---

## File Structure

- Modify: `/home/ptg/codex-claude-bridge/mcp-framing.test.mjs`
  - Responsibility: regression coverage for Codex-AGY MCP tool exposure and AGY plugin config precision.
- Modify if a test proves it is required: `/home/ptg/codex-claude-bridge/codex-mcp.ts`
  - Responsibility: Codex-side `send_to_agy` command construction, timeout, auth preflight, and tool schema.
- Modify if a test proves it is required: `/home/ptg/codex-claude-bridge/agy-codex-mcp.ts`
  - Responsibility: AGY-side `send_to_codex` command construction, Codex config isolation, timeout, and tool schema.
- Modify if a test proves it is required: `/home/ptg/.gemini/antigravity-cli/plugins/codex-bridge/mcp_config.json`
  - Responsibility: AGY plugin registration for `send_to_codex`.
- Do not modify: `/home/ptg/codex-claude-bridge/agy-claude-mcp.ts`, Claude plugin config, or Claude channel behavior.

### Task 1: Add AGY Plugin Config Regression Test

**Files:**
- Modify: `/home/ptg/codex-claude-bridge/mcp-framing.test.mjs`
- Test: `/home/ptg/codex-claude-bridge/mcp-framing.test.mjs`

- [ ] **Step 1: Write the failing test**

Add this test near the existing AGY/Codex reciprocal test:

```js
test('AGY plugin config registers only the direct Codex bridge', () => {
  const rawConfig = readFileSync('/home/ptg/.gemini/antigravity-cli/plugins/codex-bridge/mcp_config.json', 'utf8')
  const config = JSON.parse(rawConfig)
  const bridge = config.mcpServers?.['codex-bridge']

  assert.equal(bridge?.command, '/home/ptg/.bun/bin/bun')
  assert.deepEqual(bridge?.args, ['/home/ptg/codex-claude-bridge/agy-codex-mcp.ts'])
  assert.equal(bridge?.cwd, '/home/ptg')
  assert.equal(bridge?.env?.DBUS_SESSION_BUS_ADDRESS, 'unix:path=/run/user/1000/bus')
  assert.equal(bridge?.env?.DISPLAY, ':0')
  assert.equal(bridge?.env?.XDG_RUNTIME_DIR, '/run/user/1000')
  assert.equal(bridge?.tools?.send_to_codex?.background, 'always')
  assert.equal(bridge?.tools?.send_to_codex?.eager, true)
  assert(!rawConfig.includes('agy-claude-mcp.ts'), 'AGY plugin config must not switch to the Claude bridge')
})
```

- [ ] **Step 2: Run the targeted test file**

Run:

```bash
bun test /home/ptg/codex-claude-bridge/mcp-framing.test.mjs
```

Expected: either PASS if the config is already precise, or FAIL with the exact field that needs correction.

- [ ] **Step 3: Fix config only if the test fails**

If the test fails, edit `/home/ptg/.gemini/antigravity-cli/plugins/codex-bridge/mcp_config.json` so it exactly matches:

```json
{
  "mcpServers": {
    "codex-bridge": {
      "command": "/home/ptg/.bun/bin/bun",
      "args": [
        "/home/ptg/codex-claude-bridge/agy-codex-mcp.ts"
      ],
      "cwd": "/home/ptg",
      "env": {
        "DBUS_SESSION_BUS_ADDRESS": "unix:path=/run/user/1000/bus",
        "DISPLAY": ":0",
        "XDG_RUNTIME_DIR": "/run/user/1000"
      },
      "tools": {
        "send_to_codex": {
          "background": "always",
          "eager": true
        }
      }
    }
  }
}
```

- [ ] **Step 4: Re-run the targeted test file**

Run:

```bash
bun test /home/ptg/codex-claude-bridge/mcp-framing.test.mjs
```

Expected: `20 pass` before adding this task's new test becomes `21 pass`, with `0 fail`.

### Task 2: Add Direct Bridge Command Regression Tests

**Files:**
- Modify: `/home/ptg/codex-claude-bridge/mcp-framing.test.mjs`
- Test: `/home/ptg/codex-claude-bridge/mcp-framing.test.mjs`

- [ ] **Step 1: Write Codex -> AGY command test**

Add this test near the existing `Codex AGY bridge passes print timeout before print prompt` test:

```js
test('Codex send_to_agy remains a direct AGY print call', () => {
  const source = readFileSync(new URL('./codex-mcp.ts', import.meta.url), 'utf8')

  assert(source.includes("const AGY_BIN = process.env.AGY_BIN ?? '/home/ptg/.local/bin/agy-with-keyring'"))
  assert(source.includes('const AGY_DEFAULT_CWD = process.env.AGY_CODEX_BRIDGE_CWD ?? process.cwd()'))
  assert(source.includes('const AGY_TIMEOUT_MS = Number(process.env.AGY_CODEX_BRIDGE_TIMEOUT_MS ?? 300000)'))
  assert(source.includes('const authDiagnostic = await diagnoseAgyAuthEnvironment()'))
  assert(source.includes("'--dangerously-skip-permissions'"))
  assert(source.includes("'--print-timeout'"))
  assert(source.includes("'--print'"))
  assert(source.includes('Bun.spawn([AGY_BIN, ...command]'))
  assert(!source.includes('/api/to-agy-async'), 'Codex-AGY direct bridge must not use async hub endpoints')
})
```

- [ ] **Step 2: Write AGY -> Codex command test**

Add this test near the existing `AGY and Codex expose reciprocal auto-reply MCP tools` test:

```js
test('AGY send_to_codex remains an isolated Codex exec call', () => {
  const source = readFileSync(new URL('./agy-codex-mcp.ts', import.meta.url), 'utf8')

  assert(source.includes("const CODEX_BIN = process.env.CODEX_BIN ?? '/home/ptg/.nvm/versions/node/v20.20.2/bin/codex'"))
  assert(source.includes('const DEFAULT_CWD = process.env.AGY_CODEX_BRIDGE_CWD ?? process.cwd()'))
  assert(source.includes('const DEFAULT_TIMEOUT_MS = Number(process.env.AGY_CODEX_BRIDGE_TIMEOUT_MS ?? 300000)'))
  assert(source.includes("'exec'"))
  assert(source.includes("'--ignore-user-config'"))
  assert(source.includes("'--sandbox'"))
  assert(source.includes("'workspace-write'"))
  assert(source.includes("'--skip-git-repo-check'"))
  assert(source.includes("'--ignore-rules'"))
  assert(source.includes("'--output-last-message'"))
  assert(source.includes('delete env.CODEX_THREAD_ID'))
  assert(!source.includes('from-agy'), 'AGY-Codex direct bridge must not call the Claude hub from-agy endpoint')
})
```

- [ ] **Step 3: Run the targeted test file**

Run:

```bash
bun test /home/ptg/codex-claude-bridge/mcp-framing.test.mjs
```

Expected: `23 pass`, `0 fail` after Tasks 1 and 2 are present.

- [ ] **Step 4: Fix bridge files only if a test fails**

If a test fails, make the smallest edit in the named direct bridge file:

- `/home/ptg/codex-claude-bridge/codex-mcp.ts` for Codex -> AGY command construction.
- `/home/ptg/codex-claude-bridge/agy-codex-mcp.ts` for AGY -> Codex command construction.

Do not edit Claude bridge files.

### Task 3: Run Real Direct Smoke Tests

**Files:**
- No source files expected.

- [ ] **Step 1: Smoke test Codex -> AGY**

Run through the Codex MCP tool:

```text
send_to_agy(message="직접 호출 스모크 테스트입니다. 'Codex to AGY direct ok'만 출력하세요.", cd="/mnt/c/model/MODE_11_hana", timeout_ms=120000)
```

Expected: AGY returns `Codex to AGY direct ok` or the same phrase with only minimal surrounding text.

- [ ] **Step 2: Smoke test AGY -> Codex**

Run:

```bash
/home/ptg/.local/bin/agy --dangerously-skip-permissions --print-timeout 120s --print "MCP 직접 호출 스모크 테스트입니다. codex-bridge 도구 send_to_codex를 호출해서 message='AGY to Codex direct ok 라고만 답하세요.' 를 보내고, 도구 결과만 출력하세요. 도구가 없으면 'send_to_codex missing'이라고만 출력하세요."
```

Expected: output contains `AGY to Codex direct ok`.

- [ ] **Step 3: Confirm Claude-AGY files were not touched**

Run:

```bash
cd /home/ptg/codex-claude-bridge
git diff -- agy-claude-mcp.ts server.ts claude-bridges
```

Expected: no diff.

### Task 4: Commit the Direct Bridge Hardening

**Files:**
- Modify: `/home/ptg/codex-claude-bridge/mcp-framing.test.mjs`
- Modify only if tests required it: `/home/ptg/codex-claude-bridge/codex-mcp.ts`
- Modify only if tests required it: `/home/ptg/codex-claude-bridge/agy-codex-mcp.ts`
- Modify only if tests required it: `/home/ptg/.gemini/antigravity-cli/plugins/codex-bridge/mcp_config.json`

- [ ] **Step 1: Review changed files**

Run:

```bash
cd /home/ptg/codex-claude-bridge
git diff --stat
git diff -- mcp-framing.test.mjs codex-mcp.ts agy-codex-mcp.ts
```

Expected: only Codex-AGY test/direct bridge changes.

- [ ] **Step 2: Commit only owned files**

Run:

```bash
cd /home/ptg/codex-claude-bridge
git add mcp-framing.test.mjs
git add codex-mcp.ts agy-codex-mcp.ts
git commit -m "test: harden codex agy direct bridge"
```

If only the test file changed, omit unchanged files from `git add`.

Expected: commit contains no Claude-AGY changes.

## Self-Review

- Spec coverage: The plan covers direct Codex -> AGY, direct AGY -> Codex, AGY plugin registration, error/timeout/auth invariants, and a no-Claude-change check.
- Placeholder scan: No placeholder tasks remain.
- Type consistency: The plan uses the existing `send_to_agy`, `send_to_codex`, `message`, `cd`, `model`, and `timeout_ms` names.
