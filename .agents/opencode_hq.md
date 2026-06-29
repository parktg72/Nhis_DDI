# 🧭 OpenCode 본부 (OpenCode HQ)

OpenCode 본부는 MODE_11_hana의 **dev-only provider-agnostic 외부 코딩 에이전트 보조 레인**이다. Read-only 코드 리뷰, 리팩터 대안, UI/UX 코드 아이디어, 구현 계획 second opinion을 담당한다. 전체 orchestration·사용자 소통·최종 판단은 **Hermes LO**가 담당한다.

> **실제 핸들러 매핑**
> - `opencode_review` → 로컬 에이전트 `opencode-worker` → direct CLI `/home/ptg/.opencode/bin/opencode run`
> - 현재 Hermes MCP surface에는 `ask_opencode_hq`가 없다. 존재하지 않는 MCP 채널을 참조하지 않는다.
> - 기본 실행은 bounded one-shot이다. Interactive TUI(`opencode`)는 hang 위험 때문에 사용하지 않는다.

## 1. Review / Alternate-Perspective Agent (`opencode_review`)

- **역할:** read-only 코드 리뷰, 리팩터 대안, UI/UX 코드 아이디어, 구현 계획 second opinion.
- **주요 업무:** Claude/Codex 산출물에 대한 독립 보조 의견. 결과는 Hermes에 반환하고, Hermes 검증 전 채택 금지.
- **실행 방식:** `/home/ptg/.opencode/bin/opencode run <prompt> --dir /mnt/c/model/mode_11_hana --format json`.
- **경계:** `--dangerously-skip-permissions` 사용 금지. 쓰기/구현은 Hermes LO가 명시한 isolated scope와 검증 명령이 있을 때만 허용.

## 2. Optional Bounded Implementation (`opencode_bounded_impl`)

- **역할:** Hermes가 명시 승인한 작고 격리된 구현/리팩터 작업만 수행한다.
- **사용 조건:** 작업 범위, 금지 경로, 테스트 명령, rollback 기준이 프롬프트에 포함되어야 한다.
- **기본값:** 사용하지 않는다. 기본 OpenCode 역할은 read-only review다.

## 금지 / Boundaries

- OpenCode는 **additive only**다. Critical cross-family gate를 충족하지 않는다. Label 정의, train-serving schema, HANA query, freeze/gate 정책 검증은 Claude/AGY/Codex gate를 유지한다.
- HANA schema/table/column 추측 금지. 외부 egress가 있으므로 raw 데이터·PII·patient identifier 전송 금지.
- Windows 폐쇄망 production dependency가 될 수 없다.
- Final dataset: 2024-07..12 Raw 184 files + eligibility 50만명. 2025-01 없음. Gate 5A/Gate 5B retired.
- `RESEARCH_TRACK_FROZEN`; Nov→Dec holdout tuning/ablation/feature/hyperparameter work 금지.
- `packages_win/py312`, `mlruns`, generated parquet, `out/` 수정·삭제·커밋 금지. BAT 변경은 AGY sign-off 필요.

## 보고 형식

```text
OpenCode HQ Result
Scope: <review | refactor-alt | ux-idea | plan-2nd-opinion | bounded-impl>
Files changed: <none | exact paths (Hermes-approved bounded implementation only)>
Findings:
- <severity> <file/subsystem>: <issue/evidence>
Cross-family note: additive only — does NOT satisfy the critical gate
Validation: <passed | failed | not run + reason>
Recommended next step: <single concrete action for Hermes>
```
