# 구조적 개선 설계 — 환경변수·MODEL_DIR·배포 무결성·통합 테스트

**날짜:** 2026-04-03  
**리뷰어:** Codex, Gemini (3-way 합의)  
**상태:** 승인됨

---

## 배경 및 목적

Codex+Gemini 3-way 리뷰 사이클에서 반복 발견된 4가지 구조적 약점을 제거한다.

| # | 문제 | 위험 |
|---|------|------|
| 1 | 환경변수명 불일치 가능성 (DAG ↔ serving) | 배포 후 reload 실패 |
| 2 | `MODEL_DIR` 기본값 6개 위치 중복 정의 | 경로 변경 시 다수 파일 수정 필요, 불일치 재발 |
| 3 | 커밋 80bca32 원자성 미보장 + 백업 불완전 | 부분 배포 상태 가능, 롤백 불가 |
| 4 | DAG↔serving 통합 테스트 미흡 | 환경변수 계약 위반을 CI가 잡지 못함 |

---

## 설계

### 1. `config/settings.py` — 중앙 설정 모듈

#### 구조

```
config/
  __init__.py
  settings.py
```

#### 구현

```python
# config/settings.py
"""
중앙 환경변수 설정 모듈.

장수 프로세스(Airflow webserver, uvicorn)에서 환경변수 동적 변경은
반영되지 않는다. 런타임 오버라이드가 필요한 경우 각 호출 지점에서
os.environ.get()을 직접 사용할 것.
"""
import os
from pathlib import Path

# 모델 및 데이터 경로
MODEL_DIR     = Path(os.environ.get("MODEL_DIR",         "/app/models"))
FEATURES_DIR  = Path(os.environ.get("DDI_FEATURES_DIR",  "/app/data/features"))
PROCESSED_DIR = Path(os.environ.get("DDI_PROCESSED_DIR", "/app/data/processed"))
PREDICTIONS_DIR = Path(os.environ.get("DDI_PREDICTIONS_DIR", "/app/data/predictions"))
RAW_DATA_DIR  = Path(os.environ.get("DDI_RAW_DATA_DIR",  "/app/data/raw"))

# 파생 경로
MODEL_PROD_PATH   = MODEL_DIR / "model_prod.pkl"
MODEL_BACKUP_PATH = MODEL_DIR / "model_backup.pkl"

# API / 서비스
ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY", "")
SERVING_URL   = os.environ.get("DDI_SERVING_URL", "http://localhost:8000")

# 훈련 파라미터
TRAIN_WEEKS      = int(os.environ.get("DDI_TRAIN_WEEKS",    "4"))
MODEL_TYPE       = os.environ.get("DDI_MODEL_TYPE",         "ensemble")
OPTUNA_TRIALS    = int(os.environ.get("DDI_OPTUNA_TRIALS",  "50"))
RECALL_THRESHOLD = float(os.environ.get("DDI_RECALL_THRESHOLD", "0.90"))
AUC_THRESHOLD    = float(os.environ.get("DDI_AUC_THRESHOLD",    "0.85"))
```

#### 적용 범위

| 파일 | 변경 내용 |
|------|-----------|
| `dags/ddi_train_dag.py` | 상단 `os.environ.get(...)` 블록 → `from config.settings import ...` |
| `dags/ddi_etl_dag.py` | 동일 |
| `dags/ddi_feature_dag.py` | 동일 |
| `dags/ddi_batch_predict_dag.py` | 동일 |
| `serving/main.py` | `MODEL_DIR`, `MODEL_PATH` 로직 → import |
| `serving/routers/health.py` | `ADMIN_API_KEY`, `MODEL_DIR` → import |

Dockerfile `ENV MODEL_DIR=/app/models` 는 **유지** (런타임 오버라이드 진입점).

#### 모듈 레벨 캐시 제약 (Codex 지적)

`settings.py` 상수는 프로세스 시작 시 1회 평가된다. 장수 프로세스에서 환경변수를 바꿔도 반영되지 않으므로:
- Airflow DAG 태스크는 새 worker에서 실행되어 문제 없음
- uvicorn의 경우 `/admin/reload` 엔드포인트가 경로를 파라미터로 받으므로 문제 없음
- 테스트에서는 `monkeypatch.setenv` 후 `importlib.reload(settings)` 또는 직접 상수 패치 사용

#### 순환 import 방지

`config/settings.py`는 프로젝트 내 다른 모듈을 일절 import하지 않는다. 순수 `os`, `pathlib` 의존.

---

### 2. 커밋 80bca32 보강 — 원자적 배포 + 완전 백업

#### 현재 문제 (Codex 지적)

```
현재 순서:
1. model_prod.pkl 복사       ← 여기서 성공
2. model_prod.pkl.sha256 복사
3. 서브모델(.xgb.pkl) 복사
4. 서브모델.sha256 없으면 RuntimeError  ← 이미 1번은 교체된 상태
```

디스크가 부분 갱신된 채 RuntimeError가 발생하면 serving이 불일치 상태의 파일을 읽을 수 있다.

#### 수정 후 순서 (전체 선검증 → 임시 디렉터리 → 원자적 rename)

```python
def _deploy_model(**context):
    # Phase 1: 전체 아티팩트 존재 검증 (복사 없음)
    required = [main_pkl, main_sha, *submodel_pkls, *submodel_shas]
    for path in required:
        if not os.path.exists(path):
            raise RuntimeError(f"배포 중단 — 필수 아티팩트 없음: {path}")

    # Phase 2: 임시 디렉터리에 전체 복사
    tmp_dir = MODEL_DIR / ".deploy_tmp"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True)
    for src, dst_name in artifacts:
        shutil.copy2(src, tmp_dir / dst_name)

    # Phase 3: 기존 모델 전체 백업 (메인 + 서브모델 + sha256)
    backup_dir = MODEL_DIR / "backup"
    backup_dir.mkdir(exist_ok=True)
    for f in MODEL_DIR.glob("model_prod*"):
        shutil.copy2(f, backup_dir / f.name)

    # Phase 4: 원자적 promote (rename)
    for f in tmp_dir.iterdir():
        os.replace(f, MODEL_DIR / f.name)
    shutil.rmtree(tmp_dir, ignore_errors=True)
```

#### 백업 범위 확장 (Codex 지적)

현재 `model_backup.pkl` 단일 파일 → `backup/` 디렉터리에 타임스탬프 없이 최신 1세대 보관.  
보관 대상: `model_prod.pkl`, `model_prod.pkl.sha256`, `model_prod.xgb.pkl`, `model_prod.xgb.pkl.sha256`, `model_prod.lgb.pkl`, `model_prod.lgb.pkl.sha256`

---

### 3. 통합 테스트 — `tests/test_integration/`

#### 신규 파일 목록

```
tests/test_integration/
  __init__.py
  test_env_contract.py
  test_deploy_integrity.py
  test_model_path_fallback.py
  test_concurrent_reload.py
```

#### `test_env_contract.py`

| 테스트 | 검증 |
|--------|------|
| `test_model_dir_key_consistent` | `config.settings.MODEL_DIR` 키명이 DAG, serving 양쪽 소스에서 동일한 `"MODEL_DIR"` 환경변수를 읽음을 AST/소스 검사로 확인 |
| `test_admin_api_key_no_drift` | `ADMIN_API_KEY` 키명이 DAG `_deploy_model`과 `serving/routers/health.py` 양쪽에서 동일함 확인 — **기존 `DDI_ADMIN_API_KEY` 잔재 제거 검증** 포함 |
| `test_no_hardcoded_app_models` | `/app/models` 리터럴 문자열이 `config/settings.py` 외 Python 소스에 존재하지 않음 |

#### `test_deploy_integrity.py`

| 테스트 | 검증 |
|--------|------|
| `test_deploy_aborts_on_missing_submodel_sha256` | 서브모델 sha256 누락 시 `_deploy_model`이 `RuntimeError` 발생, **main pkl은 교체되지 않음** |
| `test_deploy_aborts_on_missing_main_sha256` | 메인 sha256 누락 시 동일 |
| `test_deploy_atomic_on_partial_artifact` | 아티팩트 중 하나라도 없으면 MODEL_DIR 내 기존 파일 변경 없음 |
| `test_hotswap_failure_raises` | serving `/admin/reload`가 503 반환 시 DAG가 예외 발생 |
| `test_hotswap_timeout_raises` | network timeout 시 DAG가 예외 발생 (transient 구분) |

#### `test_model_path_fallback.py`

| 테스트 | 검증 |
|--------|------|
| `test_model_path_fallback_uses_model_dir` | `MODEL_PATH` 환경변수 미설정 시 `MODEL_DIR/model_prod.pkl` 경로 사용 |
| `test_model_path_explicit_overrides_dir` | `MODEL_PATH` 설정 시 해당 경로 사용 |

#### `test_concurrent_reload.py` (Gemini 지적)

| 테스트 | 검증 |
|--------|------|
| `test_concurrent_predict_during_reload` | `reload_model` 실행 중 동시 `/predict` 요청이 에러 없이 처리됨 (lock 패턴 검증) |

#### 기존 테스트 수정

- `tests/test_serving/test_admin_cors.py:115` — `DDI_ADMIN_API_KEY` → `ADMIN_API_KEY` 픽스처 키 수정

---

### 4. ADMIN_API_KEY 보안 처리 (Gemini 지적)

운영 환경(`ENVIRONMENT=production` 또는 `ADMIN_API_KEY` 미설정)에서 serving 시작 시 명시적 경고 로그 출력. ValueError로 강제 종료는 하지 않음 (기존 동작 — 미설정 시 `/admin/reload` 엔드포인트 비활성화 — 이 이미 안전함).

`serving/main.py` lifespan에서:
```python
if not settings.ADMIN_API_KEY:
    logger.warning("ADMIN_API_KEY 미설정 — /admin/reload 비활성화됨")
```

---

## 구현 순서

1. `config/` 모듈 신설 및 기존 파일 import 교체
2. 기존 `test_admin_cors.py` 키 드리프트 수정
3. `tests/test_integration/` 테스트 추가
4. DAG `_deploy_model` 원자적 배포 + 완전 백업 로직 수정
5. `serving/main.py` ADMIN_API_KEY 경고 로그 추가
6. 전체 테스트 통과 확인

---

## 비범위 (이번 작업 제외)

- Pydantic BaseSettings 도입 (필요 시 별도 스펙)
- 모델 버전 디렉터리 / 심볼릭 링크 promote 방식 (별도 스펙)
- DAG retry 정책 세분화 (별도 스펙)
- XCom 페이로드 보안 검토 (별도 스펙)
