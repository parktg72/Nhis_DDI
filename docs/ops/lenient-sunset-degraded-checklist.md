# Operator Checklist — Feature Schema Lenient · Sunset · /health Degraded

**작성**: 2026-05-07 (cross-family 라운드 후속)
**대상**: serving 운영팀
**범위**: `FEATURE_SCHEMA_LENIENT` escape hatch / sunset deadline / `/health` degraded 신호 운영 매뉴얼
**근거 commit**: `df455d3` (#1-health), `b234832` (#6-sunset), `0140904` (#6-followup), `5604fb4` (#2-sidecar)

본 문서는 **운영자 행동 매뉴얼**이며 코드 변경/동작 변경을 도입하지 않는다. 코드 진실은 `serving/predictor.py:_validate_feature_schema`, `serving/routers/health.py:health_check`, `serving/schemas.py:HealthResponse`.

---

## 1. 핵심 개념

### 1.1 Strict 가 기본
모델 로드 시 `feature_names ⊆ _BUILDER_KNOWN_COLS ∪ _INTENTIONAL_FEATURE_ALLOWLIST` 검증. 미허용 컬럼이 있으면 **로드 거부 (False return)**.

이유: 학습 모델이 `RequestFeatureBuilder` 가 산출하지 못하는 컬럼을 사용 중이면 serving 에서 silent 0.0 fallback 으로 prediction 이 잘못된 확률 반환.

### 1.2 Lenient 는 일시 escape hatch
`FEATURE_SCHEMA_LENIENT=1` 환경 변수로 strict 우회. 미허용 컬럼이 있어도 모델 로드되고 0.0 fallback 적용 + `_schema_drift` trail 기록.

**용도**: legacy 모델 호환 sunset 윈도우. **영구 활성 금지**.

### 1.3 Sunset deadline
`FEATURE_SCHEMA_LENIENT_SUNSET_DATE` 환경 변수 (YYYY-MM-DD) 또는 코드 default `2026-08-01`. `today >= sunset` 이면 lenient 무시되고 strict 강제.

`invalid env date` (잘못된 형식) → **안전 측 차단** (lenient 비활성).

---

## 2. 환경변수 매트릭스

| `FEATURE_SCHEMA_LENIENT` | `FEATURE_SCHEMA_LENIENT_SUNSET_DATE` | today vs sunset | `_validate_feature_schema` 동작 (unknown feature 모델 기준) |
|---|---|---|---|
| unset | (any) | (any) | strict — load False |
| 1 | unset | < default `2026-08-01` | lenient — load True, `_schema_drift` 기록 |
| 1 | unset | ≥ default `2026-08-01` | strict 강제 (lenient 차단) |
| 1 | YYYY-MM-DD valid | < env date | lenient |
| 1 | YYYY-MM-DD valid | ≥ env date | strict 강제 |
| 1 | invalid (`garbage`, `2026-13-40`, `MM/DD/YYYY` 등) | (any) | strict 강제 (안전 측 차단) |

---

## 3. `/health` 응답 신호 해석

### 3.1 정상
```json
{
  "status": "ok",
  "feature_schema_lenient": false,
  "feature_schema_lenient_allowed": false,
  "feature_schema_lenient_sunset_date": "2026-08-01",
  "schema_drift": [],
  "degraded_reasons": []
}
```
조치 없음. (`sunset_date` 는 default 노출 — 정상 trail.)

### 3.2 lenient 활성, drift 없음
```json
{
  "status": "ok",
  "feature_schema_lenient": true,
  "feature_schema_lenient_allowed": true,
  "schema_drift": [],
  "degraded_reasons": []
}
```
의미: env 켜졌지만 현재 모델은 strict 통과. 정책 OK 이지만 **운영자 점검**: `FEATURE_SCHEMA_LENIENT=1` 가 의도적으로 켜져 있는지 확인. 아니면 끄기 권장.

### 3.3 silent drift 발생 (degraded)
```json
{
  "status": "degraded",
  "feature_schema_lenient": true,
  "feature_schema_lenient_allowed": true,
  "schema_drift": ["fake_xyz", "legacy_feat_a"],
  "degraded_reasons": [
    "feature_schema_drift: 2 unknown columns (fake_xyz, legacy_feat_a)"
  ]
}
```
의미: lenient 통과한 모델이 builder 미산출 컬럼 사용 중 → **prediction 이 silent 0.0 fallback 으로 잘못된 확률 가능**.

조치:
1. 즉시 알림 발송
2. 학습 파이프라인이 `_BUILDER_KNOWN_COLS ∪ _INTENTIONAL_FEATURE_ALLOWLIST` 외 컬럼 사용 — 학습/서빙 schema 정렬 필요
3. 옵션:
   - serving `RequestFeatureBuilder` 가 missing 컬럼 산출하도록 보강
   - 학습에서 미허용 컬럼 제외 후 재학습
   - missing 컬럼이 의도된 serving 미산출 (DrugMaster 미로드 등) 이면 `_INTENTIONAL_FEATURE_ALLOWLIST` 등록 검토 (단 design 결정 필요)

### 3.4 lenient 차단 (sunset 통과)
```json
{
  "status": "degraded",
  "feature_schema_lenient": true,
  "feature_schema_lenient_allowed": false,
  "feature_schema_lenient_sunset_date": "2026-08-01",
  "degraded_reasons": [
    "feature_schema_lenient_blocked_by_sunset: 2026-08-01"
  ]
}
```
의미: env 는 켜져 있으나 sunset 통과로 실제 lenient 차단 — 모델이 strict 만 통과 (drift 없음 시 prediction 정상).

조치:
1. 모델 정상화 우선 — 학습/서빙 schema 정렬 후 `FEATURE_SCHEMA_LENIENT=` (unset) 으로 escape hatch 회수
2. 정상화 불가 + 한시적 연장 필요 → `FEATURE_SCHEMA_LENIENT_SUNSET_DATE` 명시 연장 (단 5절 원칙 준수)

### 3.5 invalid sunset env
```json
{
  "feature_schema_lenient_sunset_date": "garbage-date",
  "feature_schema_lenient_allowed": false,
  "degraded_reasons": [...]
}
```
의미: env 형식 오류 → 안전 차단. **즉시 env 형식 정정** (YYYY-MM-DD).

---

## 4. Rollback / Triage 단계

### 4.1 monitoring → degraded 발생
1. `/health` 응답 polling 으로 `status` 변동 감지
2. `degraded_reasons` 분류:
   - `feature_schema_drift` → §3.3 조치
   - `feature_schema_lenient_blocked_by_sunset` → §3.4 조치
   - `predictor_not_initialized` → 서버 startup 실패 (lifespan 점검)

### 4.2 model 핫스왑 후 갑작스런 degraded
1. `/admin/reload` 또는 `/admin/reload/hierarchical` 직후 → 새 모델 schema 가 builder 와 misalign 가능
2. 신속 rollback: 직전 모델로 `/admin/reload` 재시도
3. 새 모델 schema 점검 → 학습팀과 정렬 후 재배포

### 4.3 sidecar (scaler/selector) 무결성 실패
신호: 모델 로드 자체가 실패 (`/admin/reload` 가 400 반환).
원인: scaler/selector pickle 의 `.sha256` 부재/불일치/path traversal/파일 부재.
조치: artifact 디렉터리 무결성 점검 → 학습 파이프라인의 artifact 생성 단계 재실행.

---

## 5. "Lenient 연장은 명시적 승인 없이 금지" 원칙

`FEATURE_SCHEMA_LENIENT_SUNSET_DATE` 연장 결정 시 다음 모두 충족:

1. **운영팀 + 엔지니어링 리드 명시 승인** (메일/티켓 trail)
2. **연장 사유 문서화** — 왜 strict 정렬이 본 deadline 안에 불가능했는가
3. **새 deadline 명시** — 무기한 연장 금지, 최대 30~60일 권장
4. **모델 정상화 plan 함께** — 새 deadline 까지 schema 정렬 단계 정의

이유: lenient 자체가 silent drift 회피용 escape hatch. 영구 활성 시 학습-서빙 정렬이 무한 미뤄지고 실제 prediction 품질 저하가 운영자 모르게 누적될 위험.

---

## 6. 부록 — 코드 진실 위치

| 항목 | 위치 |
|---|---|
| feature schema validation | `serving/predictor.py:_validate_feature_schema` |
| sunset helper | `serving/predictor.py:_is_feature_schema_lenient_allowed` |
| code default deadline | `serving/predictor.py:_FEATURE_SCHEMA_LENIENT_SUNSET_DEFAULT` (`date(2026, 8, 1)`) |
| `/health` 로직 | `serving/routers/health.py:health_check` |
| HealthResponse 스키마 | `serving/schemas.py:HealthResponse` |
| sidecar hash 검증 | `serving/predictor.py:MLModel._load_sidecar` |
| 회귀 가드 | `tests/test_serving/test_lenient_sunset.py`, `tests/test_serving/test_health_schema_drift.py`, `tests/test_serving/test_feature_schema_strict.py`, `tests/test_serving/test_sidecar_hash.py` |
