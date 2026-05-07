# Cross-Family Review Round Summary — 2026-05-07

**작성**: 2026-05-07
**범위**: 본 라운드는 cross-family AI(Codex / opencode / hermes / Claude advisor) 종합 리뷰 + Claude 독립 리뷰 + advisor sanity check 결과 12 commit. 어제(2026-05-06) Codex 4건 라운드 누적 합산 시 20 commit.
**제약**: 본 보고서는 사실 기반만 — 실제 commit / push / 회귀 테스트 결과 외 추측 없음.

---

## 1. 발단

직전 Codex (`docs/reports/2026-05-06-codex-review-for-claude.md`) 가 4건 review 보고. Claude 가 이를 처리한 뒤 (어제 8 commit) Codex 가 cross-family 종합 브리프 (Codex / opencode / hermes 합산) 송신. Claude advisor 검증 후 추가 finding 2건 (db.py 메타 helper / sunset 정책 부재). 본 라운드는 그 합의안 12 commit.

---

## 2. 작업 — 토픽별 commit

| # | 커밋 | 분류 | 토픽 |
|---|---|---|---|
| #3 | `57a128e` | P1 | 단일 ML schema strict validation (`_validate_feature_schema`) |
| #4 | `a4fb146` | P1/P2 | `db.py _execute_with_reconnect` helper + 4 데이터 helper |
| #5 | `9a7cbc2` | P2 | `MetricsWriter` retry/backoff (0.1/0.25/0.5) + lock_timeout 카운터 |
| #6 | `cd37299` | P2 | `BatchPredictResponse` `requested/success/failed` 분리 + `total` alias |
| #4-ext | `67c3914` | P1 | db.py 메타 helper(`get_schemas`/`get_tables`/`get_columns`) reconnect (#4 partial scope, Claude 독립 리뷰 발견) |
| #1-health | `df455d3` | P1 | `/health` `schema_drift` 노출 + status `degraded` 자동 전환 |
| #2-sidecar | `5604fb4` | P1 | scaler/selector sidecar `.sha256` 검증 (`_load_sidecar` helper) |
| #4-helper | `2acdecb` | P2 | HANA DataFrame schema validation helper + 8 fetch 적용 |
| #5-bound | `9e0f826` | P2 | batch 1000/1001 boundary 회귀 가드 (코드 변경 없이 테스트만) |
| #6-sunset | `b234832` | design | `FEATURE_SCHEMA_LENIENT` escape hatch sunset 강제 (`_is_feature_schema_lenient_allowed`) |
| #6-followup | `0140904` | followup | `/health` `feature_schema_lenient_allowed` + `feature_schema_lenient_sunset_date` 노출 (env 활성 vs 실제 효력 구분) |

---

## 3. 리스크 카테고리

### 3.1 학습-서빙 silent drift 방지
- **#3 단일 ML schema strict** — `feature_names ⊆ _BUILDER_KNOWN_COLS ∪ allowlist` 강제. unknown 컬럼 → 모델 로드 거부. `FEATURE_SCHEMA_LENIENT=1` 로 sunset 윈도우 안 일시 우회.
- **#2-sidecar artifact 무결성** — scaler/selector pickle 도 주 모델과 동일 정책: `.sha256` 부재/불일치/traversal/파일 부재 → 모델 로드 거부. partial state 오염 방지.
- **#4-helper HANA fetch column contract** — fetch 결과 DataFrame 의 `_validate_df_columns` 통과 강제. `patient_ids=[]` 시 빈 DF 도 columns 계약 유지.

### 3.2 운영 가용성/세션 관리
- **#4 + #4-ext db.py reconnect** — `query_df` 만 가지던 1회 reconnect+retry 가드를 7 helper(`get_row_count`/`preview`/`get_date_range`/`get_distinct_values`/`get_schemas`/`get_tables`/`get_columns`) 에 일관 확장. UI 카탈로그 탐색 hot path 보호.
- **#5 MetricsWriter retry/backoff** — `append()` 가 lock timeout 시 즉시 `Timeout` 전파하던 동작에 0.1/0.25/0.5 backoff retry. `lock_timeout_count` 누적 카운터 (threading.Lock 보호).

### 3.3 운영 visibility
- **#1-health + #6-followup** — `/health` 가 `schema_drift` / `feature_schema_lenient` / `feature_schema_lenient_allowed` / `feature_schema_lenient_sunset_date` / `degraded_reasons` 노출. drift / sunset 차단 시 `status="degraded"` 자동 전환.
- **/model/info** 도 `schema_drift` 노출 (디버깅/감사용).

### 3.4 API 계약 명확화
- **#6 batch response counts** — `total` (성공 건수) 의미 모호 → `requested_count` / `success_count` / `failed_count` 분리. `total` 은 backward compat 으로 `success_count` alias 유지.
- **#5-bound 1001 boundary** — `BatchPredictRequest.max_length=1000` 회귀 가드 (1000 OK, 1001 422 + predictor 미호출 검증).

### 3.5 Escape hatch 영구화 방지
- **#6-sunset** — `FEATURE_SCHEMA_LENIENT_SUNSET_DATE` env + 코드 default `2026-08-01`. `today >= sunset` → lenient 차단. invalid env → 안전 측 차단 (permissive 해석 금지).

---

## 4. 테스트 evidence

본 라운드 회귀 가드 누적 (작업별 신규 추가 케이스):

| 작업 | 신규 회귀 가드 | 테스트 파일 |
|---|---|---|
| #3 | 11 | `tests/test_serving/test_feature_schema_strict.py` |
| #4 | 13 | `tests/test_hana_app/test_db_reconnect.py` |
| #5 | 6 | `tests/test_monitoring/test_metrics_writer.py` |
| #6 | 7 | `tests/test_serving/test_batch_response_counts.py` |
| #4-ext | 7 | (test_db_reconnect 확장) |
| #1-health | 5 | `tests/test_serving/test_health_schema_drift.py` |
| #2-sidecar | 7 | `tests/test_serving/test_sidecar_hash.py` |
| #4-helper | 9 | `tests/test_hana_app/test_df_schema_validation.py` |
| #5-bound | 2 | (test_batch_response_counts 확장) |
| #6-sunset | 13 | `tests/test_serving/test_lenient_sunset.py` |
| #6-followup | 3 | (test_health_schema_drift 확장) |

전체 검증 (각 commit 후 측정 시점):
- `tests/test_serving/` 138 passed (#6-followup 직후)
- `tests/test_hana_app/` 226 passed (#4-helper 직후)
- `tests/test_monitoring/` 18 passed (#5 직후)

GAT (`tests/test_integration/test_gat_deploy.py`) 는 native lib 격리 이슈로 분리 실행 (메모리 `feedback_native_lib_test_isolation.md`).

---

## 5. 디시플린 패턴

본 라운드에서 반복 적용된 disciplinary patterns. 상세는 `~/.claude/projects/.../memory/` 메모리 참조.

### 5.1 Input contract validation at boundary
같은 패턴 3회 적용:
- `_normalize_yyyymmdd` (HANA NVARCHAR(8) date 입력 — 어제 commit `8c752b1`)
- `_validate_feature_schema` (model artifact feature_names — 본 라운드 #3)
- `_validate_df_columns` (HANA fetch 결과 DataFrame — 본 라운드 #4-helper)

공통 정책: boundary 에서 strict 가 기본, sunset 윈도우/legacy 호환 한정 lenient. invalid 입력은 fast fail.

### 5.2 Partial state cleanup on failure
`MLModel.load` (#2-sidecar), `_validate_feature_schema` 실패 시 (#3), `reload_hierarchical` schema 거부 시 (#3) — 모두 `_model=None`, `_feature_names=[]`, `_schema_drift=[]` 정리. 새 instance state 가 부분 적용된 상태로 운영 노출되지 않게.

### 5.3 Escape hatch 명시적 sunset
`FEATURE_SCHEMA_LENIENT` 가 silent 0.0 fallback 회피용 임시 우회구. `_FEATURE_SCHEMA_LENIENT_SUNSET_DEFAULT = date(2026, 8, 1)` 코드 default + env override. `today >= sunset` 부터 차단. invalid env → 안전 측 차단 (permissive 해석 금지).

### 5.4 Provisional 마크 + 1줄 revertable
`_INTENTIONAL_FEATURE_ALLOWLIST = frozenset({"dup_efmdc"})` — park 의 prod 모델 importance 측정 결과 전까지 잠정. 1줄 revertable 로 trail. `dup_efmdc` allowlist sunset 은 별도 design 합의 (본 라운드 범위 밖).

### 5.5 Cross-family 합의 → 1차 자료 우선
Codex/opencode/hermes 권고가 schema/yaml/wheel metadata 와 다르면 1차 자료를 따름 (메모리 `feedback_primary_source_overrides_ai.md`). 본 라운드에선 Codex 종합안의 일부 finding (drug_master yaml schema 등) 을 advisor 검증 후 drop, 다른 finding (db.py 메타 helper) 은 Claude 독립 리뷰로 추가.

---

## 6. 남은 항목 (park-dependent, blocked/pending)

본 라운드에서 처리하지 못한 작업. **park 답이 와야 진행 가능**:

### 6.1 dup_efmdc importance artifact 결론 (P1, blocked)
- 필요: prod 학습 모델 또는 최근 학습 artifact 경로 (`.pkl`/`.joblib`)
- 측정: `dup_efmdc` 의 feature_importance ≥ 1% 여부
- 결론:
  - ≥ 1% → `_INTENTIONAL_FEATURE_ALLOWLIST` 에서 `dup_efmdc` 제거 + serving 측에서 산출 (DrugMaster 로드 또는 별도 lookup)
  - ≈ 0 → 현행 allowlist 유지 + `docs/reports/` artifact 로 결정 trail 박음
- 함께 결정할 것: `dup_efmdc` allowlist sunset design (개념상 `FEATURE_SCHEMA_LENIENT` sunset 과 분리 — Codex 합의)

### 6.2 Dockerfile / Linux vs Windows 폐쇄망 배포 reality (P0/deprecated, blocked)
- 필요: 실제 배포 대상 결정
- Dockerfile 현 상태: `python:3.11-slim`, `packages_linux/py311/` (디렉터리 부재), `monitoring/` 미복사 — 마지막 commit 2026-04-03 (오늘의 작업과 한 달+ 격차)
- 결론:
  - Linux 컨테이너 실제 배포 → P0 풀 재빌드 (`python:3.12-slim` + `packages_linux/py312` + `monitoring/` + 누락 deps)
  - Windows 폐쇄망 only → Dockerfile 명시적 deprecated 처리 또는 제거 + 배포 가이드 정리

---

## 7. Bridge 통신 진단 trail (메타)

본 라운드 중 codex-bridge 통신 진단 일화 — engineering trail 보존:

- `reply` (id `claude-...`) → broadcast only. active pending Codex request 없으면 manual inbox 미저장
- `send_to_codex` (id `claude-init-...`) → 항상 queue
- 본 라운드 일부 보고가 inbox 도달 안 한 원인은 **session mismatch 가 아닌 tool semantics**
- Codex 가 server.ts 패치 (unmatched reply 도 queue) 적용 — bridge-enabled Claude session 재시작 후 발효
- 본 보고서 작성 시점에는 미적용 — proactive 보고는 `send_to_codex` 사용

---

## 8. Index — 코드/테스트/메모리 위치

| 영역 | 위치 |
|---|---|
| feature schema validation | `serving/predictor.py:_validate_feature_schema`, `_is_feature_schema_lenient_allowed`, `_FEATURE_SCHEMA_LENIENT_SUNSET_DEFAULT` |
| sidecar hash | `serving/predictor.py:MLModel._load_sidecar`, `_verify_hash` |
| HANA DataFrame schema | `hana_app/core/hana_etl.py:_validate_df_columns`, `_normalize_yyyymmdd` |
| HANA reconnect | `hana_app/core/db.py:_execute_with_reconnect` |
| MetricsWriter retry | `monitoring/metrics_writer.py:_APPEND_RETRY_BACKOFFS`, `lock_timeout_count` |
| /health 로직 | `serving/routers/health.py:health_check`, `_collect_schema_drift`, `_lenient_sunset_date_iso` |
| 응답 스키마 | `serving/schemas.py:HealthResponse`, `ModelInfoResponse`, `BatchPredictResponse` |
| 운영자 매뉴얼 | `docs/ops/lenient-sunset-degraded-checklist.md` (본 라운드 동시 작성) |
| 메모리 (디시플린) | `~/.claude/projects/.../memory/feedback_*.md` |
