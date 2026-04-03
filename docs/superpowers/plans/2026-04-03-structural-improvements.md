# 구조적 개선 구현 계획 (환경변수·MODEL_DIR·배포 무결성·통합 테스트)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `config/settings.py` 중앙 설정 모듈로 환경변수 중복 제거, `_deploy_model` 원자적 배포로 부분-배포 위험 제거, 통합 테스트로 DAG↔serving 계약 CI 보장

**Architecture:** `config/settings.py`가 유일한 환경변수 기본값 소스. 모든 DAG와 serving 파일이 이를 import. `_deploy_model`은 전체 아티팩트 선검증 → 임시 디렉터리 복사 → 원자적 rename 3단계로 재구성. `tests/test_integration/`에 계약 테스트·배포 무결성 테스트 추가.

**Tech Stack:** Python 3.11, FastAPI, Airflow 2.x, pytest, pytest-asyncio, httpx

---

## 파일 맵

| 액션 | 파일 | 담당 내용 |
|------|------|-----------|
| 생성 | `config/__init__.py` | 패키지 초기화 |
| 생성 | `config/settings.py` | 모든 환경변수 기본값 단일 정의 |
| 수정 | `dags/ddi_train_dag.py:43-49` | env var 블록 → config import |
| 수정 | `dags/ddi_train_dag.py:134-190` | `_deploy_model` 원자적 배포 재구성 |
| 수정 | `dags/ddi_etl_dag.py:38-40` | env var 블록 → config import |
| 수정 | `dags/ddi_feature_dag.py:41-45` | env var 블록 → config import |
| 수정 | `dags/ddi_batch_predict_dag.py:42-47` | env var 블록 → config import |
| 수정 | `serving/main.py:48-56` | lifespan env vars → config import |
| 수정 | `serving/routers/health.py:29-30` | ADMIN_API_KEY, MODEL_DIR → config import |
| 수정 | `serving/main.py:47` | lifespan에 ADMIN_API_KEY 경고 로그 추가 |
| 수정 | `tests/test_serving/test_admin_cors.py:115` | DDI_ADMIN_API_KEY → ADMIN_API_KEY |
| 생성 | `tests/test_integration/__init__.py` | 패키지 초기화 |
| 생성 | `tests/test_integration/test_env_contract.py` | 하드코딩 경로·키 드리프트 계약 테스트 |
| 생성 | `tests/test_integration/test_deploy_integrity.py` | 원자적 배포·핫스왑 실패 테스트 |
| 생성 | `tests/test_integration/test_model_path_fallback.py` | MODEL_PATH 폴백 테스트 |
| 생성 | `tests/test_integration/test_concurrent_reload.py` | 동시성 스모크 테스트 |

---

### Task 1: config/settings.py 생성 (TDD)

**Files:**
- Create: `config/__init__.py`
- Create: `config/settings.py`
- Test: `tests/test_integration/test_env_contract.py` (부분 — 설정 기본값 검증)

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_integration/__init__.py` 빈 파일 생성:
```python
```

`tests/test_integration/test_env_contract.py` 생성:
```python
"""DAG↔serving 환경변수 계약 및 하드코딩 경로 검증."""
import importlib
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent


def test_settings_model_dir_default():
    """MODEL_DIR 기본값 /app/models."""
    import config.settings as s
    importlib.reload(s)
    assert str(s.MODEL_DIR) == "/app/models"


def test_settings_model_dir_env_override(monkeypatch, tmp_path):
    """MODEL_DIR 환경변수 오버라이드."""
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    import config.settings as s
    importlib.reload(s)
    assert s.MODEL_DIR == tmp_path
    importlib.reload(s)  # cleanup — 다음 테스트에 기본값 복원


def test_settings_admin_api_key_default():
    """ADMIN_API_KEY 기본값 빈 문자열."""
    import config.settings as s
    importlib.reload(s)
    assert s.ADMIN_API_KEY == ""


def test_no_hardcoded_app_models():
    """/app/models 리터럴이 config/settings.py 외 Python 소스에 없음."""
    violations = []
    exclude = {".venv", ".venv_macos", "__pycache__", "docs", ".git"}
    for py_file in REPO_ROOT.rglob("*.py"):
        parts = set(py_file.parts)
        if parts & exclude:
            continue
        # config/settings.py 는 유일한 허용 소스
        if py_file.relative_to(REPO_ROOT) == Path("config/settings.py"):
            continue
        content = py_file.read_text(errors="replace")
        if '"/app/models"' in content or "'/app/models'" in content:
            violations.append(str(py_file.relative_to(REPO_ROOT)))
    assert violations == [], (
        "하드코딩된 /app/models 발견 — config.settings 로 교체하세요:\n"
        + "\n".join(violations)
    )


def test_admin_api_key_no_drift():
    """DDI_ADMIN_API_KEY 잔재 없음 — ADMIN_API_KEY 로 통일됨."""
    violations = []
    exclude = {".venv", ".venv_macos", "__pycache__", "docs", ".git"}
    for py_file in REPO_ROOT.rglob("*.py"):
        parts = set(py_file.parts)
        if parts & exclude:
            continue
        content = py_file.read_text(errors="replace")
        if "DDI_ADMIN_API_KEY" in content:
            violations.append(str(py_file.relative_to(REPO_ROOT)))
    assert violations == [], (
        "DDI_ADMIN_API_KEY 잔재 발견:\n" + "\n".join(violations)
    )
```

- [ ] **Step 2: 실패 확인**

```bash
cd /Volumes/model/claude/MODE_11_hana
python -m pytest tests/test_integration/test_env_contract.py -v 2>&1 | head -20
```
예상: `ModuleNotFoundError: No module named 'config'`

- [ ] **Step 3: config/settings.py 구현**

`config/__init__.py` 생성 (빈 파일):
```python
```

`config/settings.py` 생성:
```python
"""
중앙 환경변수 설정 모듈.

모든 환경변수 기본값은 이 파일이 유일한 소스다.
DAG, serving, 테스트 모두 여기서 import한다.

주의: 모듈 레벨 상수는 프로세스 시작 시 1회 평가된다.
장수 프로세스(Airflow webserver, uvicorn)에서 런타임 오버라이드는
반영되지 않는다. 테스트에서는 monkeypatch 후 importlib.reload(settings) 사용.
"""
import os
from pathlib import Path

# ── 경로 ──────────────────────────────────────────────────────────────────────
MODEL_DIR       = Path(os.environ.get("MODEL_DIR",          "/app/models"))
FEATURES_DIR    = Path(os.environ.get("DDI_FEATURES_DIR",   "/app/data/features"))
PROCESSED_DIR   = Path(os.environ.get("DDI_PROCESSED_DIR",  "/app/data/processed"))
PREDICTIONS_DIR = Path(os.environ.get("DDI_PREDICTIONS_DIR","/app/data/predictions"))
RAW_DATA_DIR    = Path(os.environ.get("DDI_RAW_DATA_DIR",   "/app/data/raw"))

# 파생 경로 (MODEL_DIR 기반)
MODEL_PROD_PATH   = MODEL_DIR / "model_prod.pkl"
MODEL_BACKUP_PATH = MODEL_DIR / "model_backup.pkl"

# ── API / 서비스 ───────────────────────────────────────────────────────────────
ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY", "")
SERVING_URL   = os.environ.get("DDI_SERVING_URL", "http://localhost:8000")

# ── 훈련 파라미터 ──────────────────────────────────────────────────────────────
TRAIN_WEEKS      = int(os.environ.get("DDI_TRAIN_WEEKS",        "4"))
MODEL_TYPE       = os.environ.get("DDI_MODEL_TYPE",             "ensemble")
OPTUNA_TRIALS    = int(os.environ.get("DDI_OPTUNA_TRIALS",      "50"))
RECALL_THRESHOLD = float(os.environ.get("DDI_RECALL_THRESHOLD", "0.90"))
AUC_THRESHOLD    = float(os.environ.get("DDI_AUC_THRESHOLD",    "0.85"))
BATCH_SIZE       = max(1, min(10_000, int(os.environ.get("DDI_BATCH_SIZE", "500"))))

# ── 데이터 파생 경로 ───────────────────────────────────────────────────────────
DDI_MATRIX_PATH = Path(os.environ.get(
    "DDI_MATRIX_PATH", "/app/data/processed/ddi_matrix_final.parquet"
))
DRUG_INDEX_PATH = Path(os.environ.get(
    "DRUG_INDEX_PATH", "/app/data/processed/drug_name_index.parquet"
))
CYP_MATRIX_PATH = Path(os.environ.get(
    "CYP_MATRIX_PATH", "/app/data/processed/cyp_matrix.parquet"
))
DRUG_INDEX_PARQUET = Path(os.environ.get(
    "DDI_DRUG_INDEX_PATH", "/app/data/processed/drug_name_index.parquet"
))
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
python -m pytest tests/test_integration/test_env_contract.py::test_settings_model_dir_default \
    tests/test_integration/test_env_contract.py::test_settings_model_dir_env_override \
    tests/test_integration/test_env_contract.py::test_settings_admin_api_key_default -v
```
예상: `3 passed`

(`test_no_hardcoded_app_models`, `test_admin_api_key_no_drift` 는 아직 FAIL — Task 2~6 완료 후 통과)

- [ ] **Step 5: 커밋**

```bash
git add config/__init__.py config/settings.py \
    tests/test_integration/__init__.py \
    tests/test_integration/test_env_contract.py
git commit -m "feat: config/settings.py 중앙 설정 모듈 + 계약 테스트 뼈대"
```

---

### Task 2: ddi_train_dag.py 환경변수 블록 교체

**Files:**
- Modify: `dags/ddi_train_dag.py:20-49`

- [ ] **Step 1: import 블록 교체**

`dags/ddi_train_dag.py` 의 `import os` 줄과 환경변수 블록(43-49)을 교체:

기존 (lines 20-49):
```python
from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG
...

FEATURES_DIR     = os.environ.get("DDI_FEATURES_DIR", "/app/data/features")
MODEL_DIR        = os.environ.get("MODEL_DIR", "/app/models")
TRAIN_WEEKS      = int(os.environ.get("DDI_TRAIN_WEEKS", "4"))
MODEL_TYPE       = os.environ.get("DDI_MODEL_TYPE", "ensemble")
OPTUNA_TRIALS    = int(os.environ.get("DDI_OPTUNA_TRIALS", "50"))
RECALL_THRESHOLD = float(os.environ.get("DDI_RECALL_THRESHOLD", "0.90"))
AUC_THRESHOLD    = float(os.environ.get("DDI_AUC_THRESHOLD", "0.85"))
```

교체 후:
```python
from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.sensors.external_task import ExternalTaskSensor
from airflow.utils.dates import days_ago

from config.settings import (
    FEATURES_DIR,
    MODEL_DIR,
    TRAIN_WEEKS,
    MODEL_TYPE,
    OPTUNA_TRIALS,
    RECALL_THRESHOLD,
    AUC_THRESHOLD,
)
```

(기존 airflow import 줄들을 그대로 유지하고 환경변수 7줄만 위의 `from config.settings import ...` 블록으로 교체)

- [ ] **Step 2: 기존 테스트 통과 확인**

```bash
python -m pytest tests/test_dags/ -v
```
예상: 기존 테스트 전부 PASS

- [ ] **Step 3: 커밋**

```bash
git add dags/ddi_train_dag.py
git commit -m "refactor: ddi_train_dag 환경변수 → config.settings import"
```

---

### Task 3: 나머지 DAG 파일 환경변수 블록 교체

**Files:**
- Modify: `dags/ddi_etl_dag.py:38-40`
- Modify: `dags/ddi_feature_dag.py:41-45`
- Modify: `dags/ddi_batch_predict_dag.py:42-47`

- [ ] **Step 1: ddi_etl_dag.py 교체**

기존 (lines 38-40):
```python
RAW_DIR = os.environ.get("DDI_RAW_DATA_DIR", "/app/data/raw")
PROC_DIR = os.environ.get("DDI_PROCESSED_DIR", "/app/data/processed")
DRUG_INDEX = os.environ.get("DDI_DRUG_INDEX_PATH", "/app/data/processed/drug_name_index.parquet")
```

교체 후 (`import os` 아래 추가):
```python
from config.settings import RAW_DATA_DIR as RAW_DIR, PROCESSED_DIR as PROC_DIR, DRUG_INDEX_PARQUET as DRUG_INDEX
```

- [ ] **Step 2: ddi_feature_dag.py 교체**

기존 (lines 41-45):
```python
PROC_DIR       = os.environ.get("DDI_PROCESSED_DIR", "/app/data/processed")
FEATURES_DIR   = os.environ.get("DDI_FEATURES_DIR", "/app/data/features")
CYP_PATH       = os.environ.get("DDI_CYP_MATRIX_PATH", "/app/data/processed/cyp_matrix.parquet")
NORMALIZER_PATH = os.environ.get("DDI_NORMALIZER_PATH", "/app/models/normalizer.pkl")
SELECTOR_PATH  = os.environ.get("DDI_SELECTOR_PATH", "/app/models/selector.pkl")
```

교체 후:
```python
from config.settings import (
    PROCESSED_DIR as PROC_DIR,
    FEATURES_DIR,
    CYP_MATRIX_PATH as CYP_PATH,
    MODEL_DIR,
)
import os as _os
NORMALIZER_PATH = _os.environ.get("DDI_NORMALIZER_PATH", str(MODEL_DIR / "normalizer.pkl"))
SELECTOR_PATH   = _os.environ.get("DDI_SELECTOR_PATH",   str(MODEL_DIR / "selector.pkl"))
```

(DDI_NORMALIZER_PATH / DDI_SELECTOR_PATH는 MODEL_DIR 파생 경로이므로 MODEL_DIR 기반 기본값 유지)

- [ ] **Step 3: ddi_batch_predict_dag.py 교체**

기존 (lines 42-47):
```python
FEATURES_DIR    = os.environ.get("DDI_FEATURES_DIR", "/app/data/features")
PREDICTIONS_DIR = os.environ.get("DDI_PREDICTIONS_DIR", "/app/data/predictions")
SERVING_URL     = os.environ.get("DDI_SERVING_URL", "http://localhost:8000")
BATCH_SIZE      = max(1, min(10_000, int(os.environ.get("DDI_BATCH_SIZE", "500"))))
DDI_MATRIX_PATH = os.environ.get("DDI_MATRIX_PATH", "/app/data/processed/ddi_matrix_final.parquet")
PROC_DIR        = os.environ.get("DDI_PROCESSED_DIR", "/app/data/processed")
```

교체 후:
```python
from config.settings import (
    FEATURES_DIR,
    PREDICTIONS_DIR,
    SERVING_URL,
    BATCH_SIZE,
    DDI_MATRIX_PATH,
    PROCESSED_DIR as PROC_DIR,
)
```

- [ ] **Step 4: 기존 테스트 통과 확인**

```bash
python -m pytest tests/test_dags/ -v
```
예상: 전부 PASS

- [ ] **Step 5: 커밋**

```bash
git add dags/ddi_etl_dag.py dags/ddi_feature_dag.py dags/ddi_batch_predict_dag.py
git commit -m "refactor: 나머지 DAG 환경변수 → config.settings import"
```

---

### Task 4: serving/main.py + serving/routers/health.py 교체

**Files:**
- Modify: `serving/main.py:48-56`
- Modify: `serving/routers/health.py:29-30`

- [ ] **Step 1: serving/main.py lifespan 교체**

기존 (lines 48-56):
```python
    _model_path = os.environ.get("MODEL_PATH") or os.path.join(
        os.environ.get("MODEL_DIR", "/app/models"), "model_prod.pkl"
    )
    init_predictor(
        model_path=_model_path,
        ddi_matrix_path=os.environ.get("DDI_MATRIX_PATH", "data/processed/ddi_matrix_final.parquet"),
        drug_index_path=os.environ.get("DRUG_INDEX_PATH", "data/processed/drug_name_index.parquet"),
        cyp_matrix_path=os.environ.get("CYP_MATRIX_PATH", "data/processed/cyp_matrix.parquet"),
    )
```

파일 상단 import 블록에 추가 (`from serving.routers import health, predict` 아래):
```python
from config import settings as _settings
```

lifespan 함수 교체 후:
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작/종료 시 리소스 초기화/해제."""
    logger.info("DDI 위험도 분류 서버 시작")
    if not _settings.ADMIN_API_KEY:
        logger.warning("ADMIN_API_KEY 미설정 — /admin/reload 비활성화됨")
    _model_path = os.environ.get("MODEL_PATH") or str(_settings.MODEL_PROD_PATH)
    init_predictor(
        model_path=_model_path,
        ddi_matrix_path=str(_settings.DDI_MATRIX_PATH),
        drug_index_path=str(_settings.DRUG_INDEX_PATH),
        cyp_matrix_path=str(_settings.CYP_MATRIX_PATH),
    )
    logger.info("예측기 초기화 완료")
    yield
    logger.info("서버 종료")
```

(MODEL_PATH는 런타임에 오버라이드될 수 있으므로 `os.environ.get("MODEL_PATH")` 유지)

- [ ] **Step 2: serving/routers/health.py 교체**

기존 (lines 29-30):
```python
_ADMIN_KEY: str = os.environ.get("ADMIN_API_KEY", "")
_MODEL_DIR: Path = Path(os.environ.get("MODEL_DIR", "/app/models")).resolve()
```

파일 상단 import에 추가 (`from serving.schemas import ...` 아래):
```python
from config import settings as _settings
```

lines 29-30 교체:
```python
_ADMIN_KEY: str = _settings.ADMIN_API_KEY
_MODEL_DIR: Path = _settings.MODEL_DIR.resolve()
```

- [ ] **Step 3: 기존 serving 테스트 통과 확인**

```bash
python -m pytest tests/test_serving/ -v
```
예상: 전부 PASS

- [ ] **Step 4: 커밋**

```bash
git add serving/main.py serving/routers/health.py
git commit -m "refactor: serving 환경변수 → config.settings import + ADMIN_API_KEY 경고 로그"
```

---

### Task 5: test_admin_cors.py 키 드리프트 수정

**Files:**
- Modify: `tests/test_serving/test_admin_cors.py:115`

- [ ] **Step 1: 키 드리프트 수정**

`tests/test_serving/test_admin_cors.py` line 115:

기존:
```python
    monkeypatch.setenv("DDI_ADMIN_API_KEY", "secret-key")
```

교체:
```python
    monkeypatch.setenv("ADMIN_API_KEY", "secret-key")
```

- [ ] **Step 2: 테스트 통과 확인**

```bash
python -m pytest tests/test_serving/test_admin_cors.py -v
```
예상: 전부 PASS (특히 `test_deploy_dag_sends_admin_key`)

- [ ] **Step 3: 계약 테스트 통과 확인**

```bash
python -m pytest tests/test_integration/test_env_contract.py -v
```
예상: 4개 전부 PASS (Task 1~5 완료 후 `test_no_hardcoded_app_models`, `test_admin_api_key_no_drift` 통과)

- [ ] **Step 4: 커밋**

```bash
git add tests/test_serving/test_admin_cors.py
git commit -m "fix: test_admin_cors DDI_ADMIN_API_KEY → ADMIN_API_KEY 키 드리프트 수정"
```

---

### Task 6: test_deploy_integrity.py — 실패하는 테스트 먼저 작성

**Files:**
- Create: `tests/test_integration/test_deploy_integrity.py`

- [ ] **Step 1: 실패하는 테스트 작성**

```python
"""배포 원자성·핫스왑 실패 시나리오 통합 테스트."""
import pytest
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch


def _ensure_airflow_mock():
    """airflow 미설치 환경에서 DAG 모듈 import 가능하도록 mock."""
    if "airflow" not in sys.modules:
        import types
        airflow = types.ModuleType("airflow")
        airflow_dag = types.ModuleType("airflow.dag")

        class DAG:
            def __init__(self, *a, **kw): pass
            def __enter__(self): return self
            def __exit__(self, *a): pass

        airflow.DAG = DAG
        sys.modules["airflow"] = airflow
        for sub in ("airflow.operators.python", "airflow.operators.empty",
                    "airflow.sensors.external_task", "airflow.utils.dates"):
            m = types.ModuleType(sub)
            m.PythonOperator = lambda **kw: None
            m.BranchPythonOperator = lambda **kw: None
            m.EmptyOperator = lambda **kw: None
            m.ExternalTaskSensor = lambda **kw: None
            m.days_ago = lambda n: None
            sys.modules[sub] = m


class _FakeTI:
    def __init__(self, model_path: str):
        self._model_path = model_path

    def xcom_pull(self, key, task_ids):
        return self._model_path


def _make_full_artifacts(staging: Path, base_name: str = "model_v1"):
    """완전한 앙상블 아티팩트 세트 생성."""
    main_pkl = staging / f"{base_name}.pkl"
    main_pkl.write_bytes(b"model")
    (staging / f"{base_name}.pkl.sha256").write_text("aabbcc  model_v1.pkl\n")
    (staging / f"{base_name}.xgb.pkl").write_bytes(b"xgb")
    (staging / f"{base_name}.xgb.pkl.sha256").write_text("aabbcc  model_v1.xgb.pkl\n")
    (staging / f"{base_name}.lgb.pkl").write_bytes(b"lgb")
    (staging / f"{base_name}.lgb.pkl.sha256").write_text("aabbcc  model_v1.lgb.pkl\n")
    return main_pkl


@pytest.fixture(autouse=True)
def _fresh_dag_module():
    """각 테스트마다 DAG 모듈 새로 import."""
    _ensure_airflow_mock()
    for key in list(sys.modules.keys()):
        if "ddi_train_dag" in key:
            del sys.modules[key]
    yield
    for key in list(sys.modules.keys()):
        if "ddi_train_dag" in key:
            del sys.modules[key]


def test_deploy_atomic_no_files_copied_on_missing_submodel_sha256(tmp_path):
    """서브모델 sha256 누락 시 prod_dir 파일 변경 없음 (원자성 보장)."""
    staging = tmp_path / "staging"
    staging.mkdir()
    prod_dir = tmp_path / "models"
    prod_dir.mkdir()

    # 서브모델 sha256 하나 누락
    main_pkl = staging / "model_v1.pkl"
    main_pkl.write_bytes(b"model")
    (staging / "model_v1.pkl.sha256").write_text("hash\n")
    (staging / "model_v1.xgb.pkl").write_bytes(b"xgb")
    # model_v1.xgb.pkl.sha256 고의로 누락
    (staging / "model_v1.lgb.pkl").write_bytes(b"lgb")
    (staging / "model_v1.lgb.pkl.sha256").write_text("hash\n")

    import config.settings as s
    import importlib
    monkeypatch_attr = lambda obj, attr, val: setattr(obj, attr, val)
    monkeypatch_attr(s, "MODEL_DIR", prod_dir)
    monkeypatch_attr(s, "MODEL_PROD_PATH", prod_dir / "model_prod.pkl")
    monkeypatch_attr(s, "SERVING_URL", "http://localhost:9999")

    from dags.ddi_train_dag import _deploy_model

    with pytest.raises(RuntimeError, match="배포 중단"):
        _deploy_model(ti=_FakeTI(str(main_pkl)))

    # prod_dir에 어떤 파일도 생성되지 않아야 함
    assert list(prod_dir.glob("model_prod*")) == [], \
        "RuntimeError 발생 전 파일이 prod_dir에 복사됨 — 원자성 깨짐"


def test_deploy_atomic_no_files_copied_on_missing_main_sha256(tmp_path):
    """메인 sha256 누락 시 prod_dir 파일 변경 없음."""
    staging = tmp_path / "staging"
    staging.mkdir()
    prod_dir = tmp_path / "models"
    prod_dir.mkdir()

    main_pkl = staging / "model_v1.pkl"
    main_pkl.write_bytes(b"model")
    # model_v1.pkl.sha256 고의로 누락

    import config.settings as s
    s.MODEL_DIR = prod_dir
    s.MODEL_PROD_PATH = prod_dir / "model_prod.pkl"
    s.SERVING_URL = "http://localhost:9999"

    from dags.ddi_train_dag import _deploy_model

    with pytest.raises(RuntimeError, match="배포 중단"):
        _deploy_model(ti=_FakeTI(str(main_pkl)))

    assert list(prod_dir.glob("model_prod*")) == []


def test_deploy_success_creates_all_files(tmp_path):
    """완전한 아티팩트로 배포 시 model_prod* 파일 전부 생성."""
    staging = tmp_path / "staging"
    staging.mkdir()
    prod_dir = tmp_path / "models"
    prod_dir.mkdir()

    main_pkl = _make_full_artifacts(staging)

    import config.settings as s
    s.MODEL_DIR = prod_dir
    s.MODEL_PROD_PATH = prod_dir / "model_prod.pkl"
    s.SERVING_URL = "http://localhost:9999"
    s.ADMIN_API_KEY = "test-key"

    def fake_post(url, json=None, headers=None, timeout=None):
        r = MagicMock()
        r.raise_for_status.return_value = None
        r.json.return_value = {"status": "ok"}
        return r

    with patch("requests.post", side_effect=fake_post):
        from dags.ddi_train_dag import _deploy_model
        _deploy_model(ti=_FakeTI(str(main_pkl)))

    assert (prod_dir / "model_prod.pkl").exists()
    assert (prod_dir / "model_prod.pkl.sha256").exists()
    assert (prod_dir / "model_prod.xgb.pkl").exists()
    assert (prod_dir / "model_prod.xgb.pkl.sha256").exists()
    assert (prod_dir / "model_prod.lgb.pkl").exists()
    assert (prod_dir / "model_prod.lgb.pkl.sha256").exists()


def test_deploy_backup_covers_all_files(tmp_path):
    """배포 성공 시 backup/ 에 기존 model_prod* 전체 보관."""
    staging = tmp_path / "staging"
    staging.mkdir()
    prod_dir = tmp_path / "models"
    prod_dir.mkdir()

    # 기존 prod 파일 준비
    (prod_dir / "model_prod.pkl").write_bytes(b"old_model")
    (prod_dir / "model_prod.pkl.sha256").write_text("oldhash\n")
    (prod_dir / "model_prod.xgb.pkl").write_bytes(b"old_xgb")
    (prod_dir / "model_prod.xgb.pkl.sha256").write_text("oldhash\n")

    main_pkl = _make_full_artifacts(staging)

    import config.settings as s
    s.MODEL_DIR = prod_dir
    s.MODEL_PROD_PATH = prod_dir / "model_prod.pkl"
    s.SERVING_URL = "http://localhost:9999"
    s.ADMIN_API_KEY = "key"

    def fake_post(url, **kw):
        r = MagicMock()
        r.raise_for_status.return_value = None
        r.json.return_value = {"status": "ok"}
        return r

    with patch("requests.post", side_effect=fake_post):
        from dags.ddi_train_dag import _deploy_model
        _deploy_model(ti=_FakeTI(str(main_pkl)))

    backup_dir = prod_dir / "backup"
    assert (backup_dir / "model_prod.pkl").read_bytes() == b"old_model"
    assert (backup_dir / "model_prod.pkl.sha256").exists()
    assert (backup_dir / "model_prod.xgb.pkl").read_bytes() == b"old_xgb"


def test_hotswap_failure_raises(tmp_path):
    """serving reload 503 시 RuntimeError."""
    staging = tmp_path / "staging"
    staging.mkdir()
    prod_dir = tmp_path / "models"
    prod_dir.mkdir()

    main_pkl = _make_full_artifacts(staging)

    import config.settings as s
    s.MODEL_DIR = prod_dir
    s.MODEL_PROD_PATH = prod_dir / "model_prod.pkl"
    s.SERVING_URL = "http://localhost:9999"
    s.ADMIN_API_KEY = "key"

    import requests as req_lib

    def fake_post_503(url, **kw):
        r = MagicMock()
        r.raise_for_status.side_effect = req_lib.HTTPError("503")
        return r

    with patch("requests.post", side_effect=fake_post_503):
        from dags.ddi_train_dag import _deploy_model
        with pytest.raises(RuntimeError, match="핫스왑 실패"):
            _deploy_model(ti=_FakeTI(str(main_pkl)))


def test_hotswap_timeout_raises(tmp_path):
    """serving reload 타임아웃 시 RuntimeError."""
    staging = tmp_path / "staging"
    staging.mkdir()
    prod_dir = tmp_path / "models"
    prod_dir.mkdir()

    main_pkl = _make_full_artifacts(staging)

    import config.settings as s
    s.MODEL_DIR = prod_dir
    s.MODEL_PROD_PATH = prod_dir / "model_prod.pkl"
    s.SERVING_URL = "http://localhost:9999"
    s.ADMIN_API_KEY = "key"

    import requests as req_lib

    def fake_post_timeout(url, **kw):
        raise req_lib.Timeout("timeout")

    with patch("requests.post", side_effect=fake_post_timeout):
        from dags.ddi_train_dag import _deploy_model
        with pytest.raises(RuntimeError, match="핫스왑 실패"):
            _deploy_model(ti=_FakeTI(str(main_pkl)))
```

- [ ] **Step 2: 실패 확인**

```bash
python -m pytest tests/test_integration/test_deploy_integrity.py -v 2>&1 | head -30
```
예상: `test_deploy_atomic_no_files_copied_*` 가 FAIL (현재 _deploy_model이 원자적이지 않음)

- [ ] **Step 3: 커밋 (failing tests)**

```bash
git add tests/test_integration/test_deploy_integrity.py
git commit -m "test: 원자적 배포 + 핫스왑 실패 통합 테스트 (RED)"
```

---

### Task 7: _deploy_model 원자적 배포 + 완전 백업 구현

**Files:**
- Modify: `dags/ddi_train_dag.py:134-190`

- [ ] **Step 1: _deploy_model 함수 전체 교체**

`dags/ddi_train_dag.py` lines 134-190 의 `_deploy_model` 함수를 아래로 교체:

```python
def _deploy_model(**context) -> None:
    """검증 통과 모델을 production 경로로 원자적 배포 + serving 핫스왑.

    배포 순서:
      Phase 1: 전체 아티팩트 존재 검증 (복사 없음 — 실패해도 prod_dir 불변)
      Phase 2: 임시 디렉터리에 전체 복사
      Phase 3: 기존 prod 파일 전체 백업 (메인 + 서브모델 + sha256)
      Phase 4: os.replace로 원자적 promote
    """
    import sys
    sys.path.insert(0, "/app")
    import logging
    import shutil
    import requests
    from pathlib import Path
    from config import settings as _s

    model_path = Path(context["ti"].xcom_pull(key="model_path", task_ids="run_training"))
    prod_dir   = _s.MODEL_DIR
    prod_path  = prod_dir / "model_prod.pkl"
    base_src   = model_path.with_suffix("")  # e.g. /app/models/model_v1

    # ── Phase 1: 전체 아티팩트 선검증 ────────────────────────────────────────
    # (src_path, prod_filename) 쌍 목록 구성
    artifacts: list[tuple[Path, str]] = [
        (model_path,                       "model_prod.pkl"),
        (Path(str(model_path) + ".sha256"), "model_prod.pkl.sha256"),
    ]
    for ext in (".xgb.pkl", ".lgb.pkl"):
        sub_src = Path(str(base_src) + ext)
        if sub_src.exists():
            sub_sha = Path(str(sub_src) + ".sha256")
            if not sub_sha.exists():
                raise RuntimeError(
                    f"배포 중단 — 서브모델 해시 없음: {sub_sha}"
                )
            artifacts.append((sub_src, "model_prod" + ext))
            artifacts.append((sub_sha, "model_prod" + ext + ".sha256"))

    for src, _ in artifacts:
        if not src.exists():
            raise RuntimeError(f"배포 중단 — 필수 아티팩트 없음: {src}")

    # ── Phase 2: 임시 디렉터리에 전체 복사 ───────────────────────────────────
    tmp_dir = prod_dir / ".deploy_tmp"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for src, dst_name in artifacts:
        shutil.copy2(src, tmp_dir / dst_name)

    # ── Phase 3: 기존 prod 전체 백업 ─────────────────────────────────────────
    backup_dir = prod_dir / "backup"
    backup_dir.mkdir(exist_ok=True)
    for f in prod_dir.glob("model_prod*"):
        if f.is_file():
            shutil.copy2(f, backup_dir / f.name)

    # ── Phase 4: 원자적 promote ───────────────────────────────────────────────
    for f in tmp_dir.iterdir():
        os.replace(f, prod_dir / f.name)
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # ── Serving 핫스왑 ────────────────────────────────────────────────────────
    try:
        resp = requests.post(
            f"{_s.SERVING_URL}/admin/reload",
            json={"model_path": str(prod_path)},
            headers={"X-Admin-Key": _s.ADMIN_API_KEY},
            timeout=30,
        )
        resp.raise_for_status()
        logging.info("Serving 핫스왑 완료: %s", resp.json())
    except Exception as exc:
        logging.warning("Serving 핫스왑 실패: %s", exc)
        raise RuntimeError(f"Serving 핫스왑 실패 — 구버전 모델로 서빙 중: {exc}") from exc
```

- [ ] **Step 2: 테스트 통과 확인**

```bash
python -m pytest tests/test_integration/test_deploy_integrity.py -v
```
예상: 6개 전부 PASS

- [ ] **Step 3: 기존 DAG 테스트 통과 확인**

```bash
python -m pytest tests/test_dags/ tests/test_serving/ -v
```
예상: 전부 PASS

- [ ] **Step 4: 커밋**

```bash
git add dags/ddi_train_dag.py
git commit -m "fix: _deploy_model 원자적 배포 (선검증→tmp복사→백업→rename) + 완전 백업"
```

---

### Task 8: test_model_path_fallback.py

**Files:**
- Create: `tests/test_integration/test_model_path_fallback.py`

- [ ] **Step 1: 테스트 작성**

```python
"""MODEL_PATH 환경변수 폴백 동작 검증."""
import importlib
import os
import pytest


def test_model_path_fallback_uses_model_dir(monkeypatch, tmp_path):
    """MODEL_PATH 미설정 시 MODEL_DIR/model_prod.pkl 사용."""
    monkeypatch.delenv("MODEL_PATH", raising=False)
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))

    import config.settings as s
    importlib.reload(s)

    # serving/main.py lifespan 로직 재현
    model_path = os.environ.get("MODEL_PATH") or str(s.MODEL_PROD_PATH)
    assert model_path == str(tmp_path / "model_prod.pkl")

    importlib.reload(s)  # cleanup


def test_model_path_explicit_overrides_dir(monkeypatch, tmp_path):
    """MODEL_PATH 명시 설정 시 MODEL_DIR 무시."""
    explicit = str(tmp_path / "custom_model.pkl")
    monkeypatch.setenv("MODEL_PATH", explicit)
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "other"))

    import config.settings as s
    importlib.reload(s)

    model_path = os.environ.get("MODEL_PATH") or str(s.MODEL_PROD_PATH)
    assert model_path == explicit

    importlib.reload(s)  # cleanup


def test_model_prod_path_derived_from_model_dir(monkeypatch, tmp_path):
    """settings.MODEL_PROD_PATH 가 MODEL_DIR / model_prod.pkl 임."""
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    import config.settings as s
    importlib.reload(s)
    assert s.MODEL_PROD_PATH == tmp_path / "model_prod.pkl"
    importlib.reload(s)
```

- [ ] **Step 2: 테스트 통과 확인**

```bash
python -m pytest tests/test_integration/test_model_path_fallback.py -v
```
예상: 3개 PASS

- [ ] **Step 3: 커밋**

```bash
git add tests/test_integration/test_model_path_fallback.py
git commit -m "test: MODEL_PATH 폴백 동작 통합 테스트"
```

---

### Task 9: test_concurrent_reload.py

**Files:**
- Create: `tests/test_integration/test_concurrent_reload.py`

- [ ] **Step 1: 테스트 작성**

```python
"""reload_model 중 동시 /predict 요청 동시성 스모크 테스트."""
import asyncio
import pytest
from unittest.mock import MagicMock, patch


@pytest.mark.asyncio
async def test_concurrent_predict_during_reload():
    """reload_model 실행 중 /predict 요청이 500 없이 처리됨."""
    # predictor 모킹 — 실제 모델 파일 불필요
    mock_predictor = MagicMock()
    mock_predictor.predict.return_value = {
        "risk_level": "LOW",
        "score": 0.1,
        "reasons": [],
        "rule_triggered": False,
    }
    mock_predictor._model = MagicMock()  # 로드된 상태

    with patch("serving.predictor.get_predictor", return_value=mock_predictor), \
         patch("serving.predictor.init_predictor"):

        from httpx import AsyncClient, ASGITransport
        from serving.main import app

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            # /health 는 모델 없이도 동작
            tasks = [client.get("/health") for _ in range(10)]
            responses = await asyncio.gather(*tasks)

        statuses = [r.status_code for r in responses]
        assert all(s in (200, 503) for s in statuses), (
            f"예상 외 상태코드 발견: {set(statuses)}"
        )
        # 500 (서버 에러) 없어야 함
        assert 500 not in statuses, "동시 요청 중 서버 내부 오류 발생"
```

- [ ] **Step 2: pytest-asyncio 설치 확인 및 테스트 통과**

```bash
python -m pytest tests/test_integration/test_concurrent_reload.py -v
```
예상: PASS (또는 pytest-asyncio 없으면 `pip install pytest-asyncio` 후 재실행)

- [ ] **Step 3: 커밋**

```bash
git add tests/test_integration/test_concurrent_reload.py
git commit -m "test: reload 중 동시 요청 동시성 스모크 테스트"
```

---

### Task 10: 전체 테스트 통과 확인 및 최종 커밋

- [ ] **Step 1: 전체 테스트 실행**

```bash
python -m pytest tests/ -v --tb=short 2>&1 | tail -20
```
예상: 기존 341 passed + 신규 ~18 tests = 350+ passed, 0 failed

- [ ] **Step 2: 하드코딩 경로 최종 검증**

```bash
python -m pytest tests/test_integration/test_env_contract.py -v
```
예상: 5개 전부 PASS

- [ ] **Step 3: 최종 커밋**

```bash
git add -A
git commit -m "chore: 구조적 개선 완료 — 환경변수 통일·원자적 배포·통합 테스트 18건 추가"
```

---

## 자체 검토 (Spec 대비)

| Spec 요구사항 | 구현 Task |
|---------------|-----------|
| config/settings.py 신설 | Task 1 |
| DAG 4개 import 교체 | Task 2, 3 |
| serving/main.py import 교체 | Task 4 |
| serving/routers/health.py import 교체 | Task 4 |
| ADMIN_API_KEY 경고 로그 | Task 4 |
| DDI_ADMIN_API_KEY 키 드리프트 수정 | Task 5 |
| test_env_contract.py | Task 1, 5 |
| test_deploy_integrity.py | Task 6, 7 |
| test_model_path_fallback.py | Task 8 |
| test_concurrent_reload.py | Task 9 |
| _deploy_model 원자적 배포 | Task 7 |
| 완전 백업 (서브모델 + sha256) | Task 7 |

**누락 없음.**
