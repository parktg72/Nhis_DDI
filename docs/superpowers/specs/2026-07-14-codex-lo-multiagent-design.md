# Codex LO 멀티에이전트 시스템 설계

**작성일:** 2026-07-14
**상태:** 승인됨
**적용 범위:** MODE_11_hana 개발 저장소의 AI 오케스트레이션 계층
**대체 대상:** OpenCode LO 활성 구성과 Claude LO용 레거시 실행 구성

> **구현 메모:** 승인된 설계는 보호 아티팩트의 메타데이터만 스냅샷·검증하는 가드로 보안을 강화했으며, 커밋 표면은 Git으로 검증한다. 에이전트 토폴로지는 변경되지 않는다.

## 1. 목적

Codex를 MODE_11_hana의 유일한 L0/LO(Leading Orchestrator)로 두고, Claude Code와 AGY를 Codex가 통제하는 외부 모델 서브에이전트로 구성한다. Claude Code의 내장 advisor 모델은 Fable 5(`fable`, 실제 모델 `claude-fable-5`)로 설정한다.

Codex LO가 사용자 소통, 작업 분해, 순서 결정, 승인, 결과 검증, 충돌 해결, 최종 보고를 독점한다. Claude Code와 AGY는 자기완결적 brief를 받아 증거를 반환하며 사용자나 다른 워커에게 직접 연락하지 않는다.

## 2. 확정 결정

1. **LO는 Codex 하나뿐이다.** OpenCode, Claude Code, AGY는 LO가 될 수 없다.
2. **활성 OpenCode 레인은 제거한다.** OpenCode worker, handler, fallback, model policy, 라우팅 문구를 현재 설정과 운영 문서에서 삭제한다.
3. **Claude Code와 AGY만 외부 서브에이전트로 둔다.** Codex 자체 구현은 LO가 직접 수행하며 별도 Codex worker를 기본 경로로 만들지 않는다.
4. **Fable 5는 Claude Code 내장 advisor로 사용한다.** Fable을 일반 직접 호출 모델로 가장하지 않는다.
5. **advisor 호출은 매번 새 Claude Code 세션에서 정확히 한 번 수행한다.** 현재 환경에서 Fable advisor 호출 뒤 다른 tool round-trip을 거쳐 재호출하면 `unavailable`이 될 수 있으므로 같은 세션의 반복 호출을 금지한다.
6. **기본 advisor 게이트는 plan과 finish 두 지점이다.** 두 호출은 서로 다른 새 세션이다. 단순 질의나 무변경 보고처럼 advisor 가치가 없는 작업은 Codex LO가 생략 사유를 결과에 남길 수 있다.
7. **worker 간 직접 통신과 재귀 위임을 금지한다.** 모든 요청과 결과는 Codex LO를 거친다.
8. **현재 저장소의 안전 게이트를 그대로 보존한다.** 연구 동결, 보호 경로, Python 3.12, BAT CRLF/UTF-8, HANA schema 확인, train-serving parity 규칙은 오케스트레이터 변경으로 완화되지 않는다.

## 3. 접근법 비교

### A. Codex 네이티브 설정 + 외부 CLI 브리지 — 채택

프로젝트 `.codex/config.toml`과 `.codex/agents/*.toml`로 Codex의 역할과 서브에이전트 표면을 정의한다. Claude Code와 AGY 실행은 allowlist된 어댑터를 통해서만 수행한다.

장점은 Codex CLI의 공식 프로젝트 설정 표면을 사용하고, 역할·sandbox·depth를 저장소 단위로 추적할 수 있으며, 외부 CLI 결과를 일관된 envelope로 검증할 수 있다는 점이다. 단점은 Codex 네이티브 서브에이전트가 외부 모델을 직접 호스팅하지 않으므로 얇은 bridge agent와 adapter가 필요하다는 점이다.

### B. 기존 `.multiagent/`를 Codex flavor로 전환 — 미채택

현재 file-as-memory 패턴을 재사용할 수 있지만 `.multiagent/`는 gitignored 생성물이고 Claude Code 세션을 오케스트레이터로 가정한다. 실행 정본과 추적 정본이 분리되어 다시 drift가 생기므로 이번 목표에 맞지 않는다.

### C. 문서와 프롬프트만 변경 — 미채택

수정량은 가장 작지만 실제 모델 호출, 권한, timeout, 결과 검증을 강제하지 못한다. “Codex가 LO”라는 문구만 있고 실행 가능한 시스템이 없는 상태가 되므로 제외한다.

## 4. 아키텍처

```text
사용자
  |
  v
Codex LO
  |-- 직접 수행: 분해, 구현, 통합, 검증, 최종 보고
  |
  |-- claude-bridge (Codex custom subagent)
  |     |-- Claude Code worker: 요구사항·아키텍처·논리 QA
  |     `-- Claude Code advisor mode
  |           `-- built-in /advisor, advisorModel=fable, 새 세션 1회
  |
  `-- agy-bridge (Codex custom subagent)
        `-- AGY CLI: 환경·DevOps·Python 3.12·BAT·디스크·리스크
```

Codex custom subagent는 외부 모델인 척하지 않는다. bridge의 역할은 brief 검증, allowlist adapter 호출, 결과 envelope 확인, Codex LO로의 증거 반환이다. 실제 추론 제공자는 각각 Claude Code CLI와 AGY CLI이다.

## 5. 구성 요소

### 5.1 Codex LO 정책

`AGENTS.md`를 현재 오케스트레이션 정책의 최상위 정본으로 사용한다. 다음을 명시한다.

- Codex가 유일한 LO이자 사용자 소통 창구다.
- Claude Code와 AGY는 증거만 반환한다.
- 한 worker 호출이 진행 중이면 다른 worker로 새 outbound 메시지를 보내지 않는다.
- 서브에이전트가 서브에이전트를 호출하지 못하도록 depth 1을 유지한다.
- 결과에는 변경 파일, 명령·테스트, 검증 상태, 리스크, 다음 행동 하나가 반드시 포함된다.
- critical 변경은 Claude와 Codex의 cross-family 검토를 거친다. AGY는 환경·운영 게이트를 맡는다.

### 5.2 Codex 프로젝트 설정

`.codex/config.toml`은 다음 최소 설정만 소유한다.

- 안정 기능인 multi-agent 활성화
- 재귀 위임 방지를 위한 `agents.max_depth = 1`
- `claude-bridge`, `agy-bridge`의 프로젝트 agent 등록
- 저장소별 모델 핀은 두지 않고 현재 Codex 사용자 설정을 상속

`.codex/agents/claude-bridge.toml`과 `.codex/agents/agy-bridge.toml`은 공식 custom-agent 필수 필드인 `name`, `description`, `developer_instructions`를 제공한다. bridge는 자신이 외부 provider가 아니라는 점, 사용할 adapter, read-only 기본값, 보고 형식을 명시한다. 외부 adapter 실행은 Codex LO/사용자가 승인한 runtime permission을 전제로 하며, strict read-only bridge 안에서 항상 실행된다고 가정하지 않는다. 현재 runtime이 adapter를 실행할 수 없으면 bridge는 스스로 권한을 확대하거나 사용자에게 prompt/contact하거나 sandbox를 약화하지 않고, 시도한 정확한 명령과 runtime 증거를 포함한 `BLOCK`을 Codex LO에 반환한다. 로컬 Codex strict `:read-only` probe에서 `mktemp /tmp`가 차단된 것은 예상된 fail-closed 결과였으므로, live adapter dispatch에는 parent approval/runtime permission이 선행되어야 한다.

### 5.3 Claude Code worker와 Fable advisor

추적되는 프로젝트 `.claude/settings.json`은 공유 가능한 최소 설정인 `advisorModel: "fable"`만 포함한다. 기존 machine-local `permissions` 객체는 의미를 바꾸거나 권한을 확대하지 않고 gitignored `.claude/settings.local.json`으로 이동한다. 공유 파일에는 permission allowlist, `bypassPermissions`, host path가 들어가지 않는다. `claude-advisor` 모드만 `--settings <shared .claude/settings.json>`을 명시적으로 전달하며, `claude` worker 모드는 Claude Code의 정상적인 project settings discovery에 의존한다.

Claude adapter는 두 모드를 제공한다.

- `worker`: Claude Code가 요구사항, 아키텍처, label/schema/freeze 논리, 최종 QA를 read-only로 검토한다.
- `advisor`: 새 Claude Code print 세션에서 내장 advisor를 정확히 한 번 호출한 뒤, base Claude가 Fable의 advisor 결과를 구조화해 반환한다.

advisor prompt는 advisor를 첫 번째 추론 단계에서 한 번만 사용하고 재호출하지 말 것을 요구한다. plan과 finish가 모두 필요하면 adapter를 두 번 실행하여 세션을 분리한다.

### 5.4 AGY worker

AGY adapter는 기본적으로 sandbox + plan/read-only 모드에서 실행한다. 담당 범위는 다음으로 제한한다.

- Python 3.12 dev/prod parity
- Windows 폐쇄망 배포와 offline dependency 위험
- `.bat` CRLF 및 `chcp 65001`
- feature build temp disk 10GB+ preflight
- 보호 경로 및 연구 동결 위험 탐지
- 명시적으로 요청된 외부 조사

AGY가 코드 구현이나 최종 스펙 소유권을 가져서는 안 된다. 파일 수정이 필요한 제안은 Codex LO에 반환하며, 별도 승인 없이 직접 적용하지 않는다.

### 5.5 외부 CLI adapter

추적 가능한 단일 adapter entrypoint를 둔다. 입력은 명령행에 삽입한 자유 텍스트가 아니라 저장소 내부 brief 파일 경로로 받는다. adapter는 다음을 강제한다.

- provider allowlist: `claude`, `claude-advisor`, `agy`
- brief 경로 정규화 및 저장소 경계 확인
- 호출마다 하나의 bounded `AGENT_ADAPTER_TIMEOUT`과 별도의 bounded `AGENT_ADAPTER_KILL_AFTER` kill grace 적용
- stdout/stderr 분리 수집
- 프로세스 결과에서 파생한 status(`ok`/`timeout`/`error`), provider, 정규화된 canonical brief 경로, exit code, duration, raw stdout, sanitized stderr를 포함한 JSON process envelope
- secret-like 문자열을 stderr에서 최소한으로 마스킹
- `eval`, interactive permission prompt, background orphan process 금지

실제 모델 응답 stdout은 설계상 원문을 보존하며 adapter가 sanitize하지 않는다. stderr의 secret-like 문자열만 제한적으로 마스킹하므로 brief에 secret을 넣지 않고 envelope 전체를 민감한 실행 산출물로 취급한다. Process status는 실행 결과일 뿐 worker 응답의 의미적 성공이나 증거 형식 검증이 아니다. bridge와 Codex LO가 별도로 worker 응답 형식과 필수 evidence fields를 검증한 뒤에만 결과를 채택한다.

## 6. 데이터 흐름

1. 사용자가 Codex LO에 요청한다.
2. Codex가 저장소 규칙과 hard gate를 확인하고 작업을 분해한다.
3. plan advisor가 필요한 작업이면 Codex가 `claude-bridge`에 advisor brief를 보낸다.
4. bridge가 새 Claude Code 세션을 열고 내장 Fable advisor를 한 번 사용한다.
5. Codex가 advisor 결과를 검토한 뒤 직접 구현하거나 Claude/AGY worker를 순차 호출한다.
6. worker는 JSON envelope와 정형 결과를 반환한다.
7. Codex가 파일·명령·테스트 증거를 직접 재검증한다.
8. finish advisor가 필요한 작업이면 별도의 새 Claude Code 세션에서 한 번 호출한다.
9. Codex가 충돌을 해결하고 사용자에게 최종 결과를 보고한다.

worker 결과는 결정권이 없는 upstream evidence다. Codex LO가 검증하지 않은 성공 주장은 사용자에게 전달하지 않는다.

## 7. OpenCode 제거 범위

다음 활성 표면에서 OpenCode를 제거한다.

- `AGENTS.md`의 LO, role, trigger owner, routing 문구
- `CLAUDE.md`의 OpenCode LO와 model/fallback 정책
- `.agents/agents_config.json`의 OpenCode LO/HQ/worker/handler 설정
- `.agents/*.md`와 `.claude/agents/*.md`의 OpenCode LO 반환 대상
- `.agents/opencode_hq.md`
- `.claude/agents/opencode-worker.md`
- 활성 validation command의 OpenCode smoke 및 실행 파일 의존성

기존 Git commit 이력은 변경하지 않는다. 과거 설계 문서는 삭제해 provenance를 잃게 하지 않고, 문서 상단에 본 설계로 대체되었음을 표시한다. 현재 운영 규칙이나 실행 경로로 참조되지 않게 한다.

`.multiagent/`는 gitignored 레거시 생성물이므로 이번 변경의 실행 정본으로 사용하지 않는다. 사용자 로컬 생성물을 파괴적으로 삭제하지 않고, 현재 정책 문서에서 활성 경로로 참조하지 않게 한다.

## 8. 실패 처리

- **CLI 없음:** 호출 전 `command -v`로 확인하고 BLOCK 결과를 반환한다.
- **인증·quota·network 실패:** 최대 1회만 동일 provider로 재시도한다. 다시 실패하면 provider를 바꿔 성공한 것처럼 처리하지 않고 실패를 기록한다.
- **timeout:** 프로세스를 종료하고 timeout envelope를 반환한다. partial stdout은 보존한다.
- **Fable advisor unavailable:** 같은 세션에서 재호출하지 않는다. 새 세션 1회 재시도 후 실패를 Codex LO에 반환한다.
- **AGY sandbox 제약:** 필요한 읽기 작업이 sandbox에서 불가능하면 권한 확대를 자동 수행하지 않고 Codex LO가 사용자 승인을 요청한다.
- **bridge runtime 제약:** strict read-only runtime에서 temp file 또는 외부/home/socket 접근이 차단되면 bridge는 sandbox를 약화하거나 직접 사용자에게 권한을 요청하지 않고, 정확한 명령과 증거를 담아 Codex LO에 `BLOCK`을 반환한다.
- **결과 형식 불일치:** Codex가 결과를 채택하지 않고 한 번만 형식 교정 재요청한다.
- **worker 의견 충돌:** 다수결로 결정하지 않는다. Codex가 확인 가능한 파일·테스트·정책 근거를 우선하여 판정하고 불확실성을 보고한다.
- **hard gate 탐지:** 보호 경로나 연구 동결 위반이면 downstream 작업을 중단한다.

## 9. 보안과 권한

- 외부 worker는 기본 read-only다.
- adapter는 brief 파일과 허용된 repo 경로만 전달한다.
- 추적되는 `.claude/settings.json`은 Fable advisor 선택만 소유한다. 기존 machine permission은 ignored `.claude/settings.local.json`에만 보존하며 stage·commit하거나 공유 deliverable로 취급하지 않는다.
- Claude/AGY 실행에 `dangerously-skip-permissions`를 사용하지 않는다.
- worker 호출은 shell 문자열 조립이나 `eval`을 사용하지 않는다.
- model output을 shell 명령으로 재실행하지 않는다.
- `packages_win/py312/`, `mlruns/`, 생성 parquet, `out/`은 사용자 승인 없이 접근 범위를 확대하거나 수정하지 않는다.
- protected-artifact guard는 path, mode, size, mtime, symlink target만 비교하는 metadata-only change detector이며 content hash나 불변성 증명이 아니다. Tracked/untracked commit surface는 별도의 configured Git validator로 보완한다.
- 이 오케스트레이션 계층은 개발 도구이며 Windows 폐쇄망 production runtime dependency가 아니다.

## 10. 검증 전략

### 10.1 정적 검증

- `.codex/config.toml`과 agent TOML을 Python 3.12 `tomllib`로 파싱
- `.agents/agents_config.json`을 JSON parser로 검증
- adapter `bash -n` 및 가능하면 `shellcheck`
- active config/docs에서 `OpenCode`, `opencode-worker`, `opencode run` 잔존 검색
- 모든 agent 정의에서 hard gate와 Codex LO 반환 형식 확인
- configured `protected_diff`로 tracked, staged, untracked, branch commit을 포함한 Git commit surface의 보호 경로 변경을 검사
- 작업 전 `protected_snapshot`, 작업 후 `protected_verify`로 ignored 보호 artifact까지 metadata 기준으로 검사
- 변경된 `.bat`는 추가로 CRLF와 `chcp 65001`을 검증

### 10.2 자동 테스트

외부 CLI를 실제 호출하지 않는 fake executable 기반 테스트를 추가한다.

- provider allowlist
- 저장소 밖 brief 경로 거부
- 성공/실패/timeout envelope
- stderr redaction
- Claude advisor mode가 fresh one-shot invocation을 구성하는지 확인
- AGY mode가 sandbox/read-only 옵션을 구성하는지 확인

테스트는 Python 3.12에서 실행한다. 이 작업은 ETL, feature build, training을 실행하지 않으므로 HANA temp-disk preflight 대상이 아니다.

### 10.3 라이브 스모크

사용자 승인 아래 다음을 각각 한 번 실행한다.

1. Claude worker가 지정 문자열과 실제 model/session metadata를 반환
2. 새 Claude 세션에서 Fable 내장 advisor를 한 번 사용하고 advisor-backed 결과를 반환
3. AGY가 read-only로 지정 문자열과 gate 요약을 반환

라이브 스모크는 repository write를 허용하지 않는다. 결과에 실제 provider/model deviation, exit code, duration을 기록한다.

## 11. 완료 조건

- Codex가 모든 활성 문서와 설정에서 유일한 LO다.
- OpenCode 활성 agent, handler, routing, fallback, smoke 의존성이 없다.
- `.codex/config.toml`이 Claude/AGY bridge를 유효하게 등록한다.
- Claude Code project settings가 Fable 5를 내장 advisor로 지정한다.
- advisor 호출이 새 세션당 한 번으로 강제된다.
- Claude와 AGY 기본 호출이 read-only이며 JSON envelope를 반환한다.
- 정적·자동 테스트가 Python 3.12에서 통과한다.
- 승인된 라이브 스모크 3종이 통과하거나, 실패가 정확한 blocker로 기록된다.
- 보호 경로, `.bat`, 학습/서빙 schema, frozen holdout에는 변경이 없다.

## 12. 비범위

- ML 모델, feature, label, HANA query, serving 동작 변경
- ETL, training, holdout 평가 실행
- Windows production 번들 또는 offline wheel 변경
- OpenCode Git 이력 재작성
- worker가 worker를 호출하는 hierarchical delegation
- provider 장애를 숨기는 자동 cross-provider fallback
- commit, push, PR, merge, deploy
