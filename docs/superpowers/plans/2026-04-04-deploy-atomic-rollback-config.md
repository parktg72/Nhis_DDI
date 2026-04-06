# Deploy Atomicity, Rollback, and Config Completion Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Codex+Gemini HIGH/MEDIUM 5건 수정 — 번들 원자성(symlink), hotswap 롤백, config 완성, 다중 인스턴스 reload, backup 보존 정책.

**Architecture:** 
Phase 4를 `os.rename(tmp_dir → .v_TIMESTAMP/)` + `current` 심링크 원자 교체로 전환한다. `MODEL_PROD_PATH`는 `MODEL_DIR/current/model_prod.pkl`로 변경. Hotswap 실패 시 `backup/` 파일을 새 versioned 디렉터리로 복원 후 `current` 심링크를 되돌린다. `LOG_LEVEL`·`CORS_ORIGINS`·`SERVING_URLS`·`BACKUP_KEEP_N`을 settings.py에 추가한다.

**Tech Stack:** Python 3.9, FastAPI, Airflow, pytest, os.symlink/os.replace

---

## File Structure

| 파일 | 변경 내용 |
|------|-----------|
| `config/settings.py` | LOG_LEVEL, CORS_ORIGINS, SERVING_URLS, BACKUP_KEEP_N 추가 |
| `dags/ddi_train_dag.py` | Phase 3·4 재작성, hotswap 롤백, 다중 URL, versioned dir 정리 |
| `serving/main.py` | `_settings.LOG_LEVEL`, `_settings.CORS_ORIGINS` 사용 |
| `tests/test_integration/test_deploy_integrity.py` | symlink 구조 반영, rollback 테스트 추가 |
| `tests/test_integration/test_env_contract.py` | 새 settings 키 계약 테스트 추가 |

`serving/routers/health.py`는 이미 `.resolve()`로 symlink를 처리하므로 변경 불필요.

---

### Task 1: HIGH — config.settings에 LOG_LEVEL·CORS_ORIGINS·SERVING_URLS·BACKUP_KEEP_N 추가

**Files:**
- Modify: `config/settings.py`
- Test: `tests/test_integration/test_env_contract.py`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_integration/test_env_contract.py` 끝에 추가:

```python
def test_settings_log_level_default(monkeypatch):
    """LOG_LEVEL 기본값은 INFO."""
    import importlib
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    import config.settings as s
    importlib.reload(s)
    assert s.LOG_LEVEL == "INFO"
    importlib.reload(s)


def test_settings_cors_origins_default(monkeypatch):
    """CORS_ORIGINS 기본값은 빈 문자열."""
    import importlib
    monkeypatch.delenv("CORS_ORIGINS", raising=False)
    import config.settings as s
    importlib.reload(s)
    assert s.CORS_ORIGINS == ""
    importlib.reload(s)


def test_settings_serving_urls_single(monkeypatch):
    """DDI_SERVING_URLS 미설정 시 DDI_SERVING_URL 단일 URL 사용."""
    import importlib
    monkeypatch.delenv("DDI_SERVING_URLS", raising=False)
    monkeypatch.setenv("DDI_SERVING_URL", "http://host1:8000")
    import config.settings as s
    importlib.reload(s)
    assert s.SERVING_URLS == ["http://host1:8000"]
    importlib.reload(s)


def test_settings_serving_urls_multi(monkeypatch):
    """DDI_SERVING_URLS 설정 시 쉼표 구분 URL 목록 반환."""
    import importlib
    monkeypatch.setenv("DDI_SERVING_URLS", "http://a:8000,http://b:8000")
    import config.settings as s
    importlib.reload(s)
    assert s.SERVING_URLS == ["http://a:8000", "http://b:8000"]
    importlib.reload(s)


def test_settings_backup_keep_n_default(monkeypatch):
    """BACKUP_KEEP_N 기본값은 5."""
    import importlib
    monkeypatch.delenv("DDI_BACKUP_KEEP_N", raising=False)
    import config.settings as s
    importlib.reload(s)
    assert s.BACKUP_KEEP_N == 5
    importlib.reload(s)
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
python3 -m pytest tests/test_integration/test_env_contract.py -v --tb=short 2>&1 | tail -15
```

Expected: 5건 FAILED (AttributeError: module has no attribute)

- [ ] **Step 3: settings.py에 신규 상수 추가**

`config/settings.py`의 `# ── API / 서비스` 섹션에 추가:

```python
# ── 로깅 / CORS ────────────────────────────────────────────────────────────────
LOG_LEVEL    = os.environ.get("LOG_LEVEL",    "INFO")
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "")

# ── 다중 인스턴스 핫스왑 ───────────────────────────────────────────────────────
# DDI_SERVING_URLS=http://inst1:8000,http://inst2:8000  (쉼표 구분)
# 미설정 시 DDI_SERVING_URL 단일 인스턴스 사용
_serving_urls_raw = os.environ.get("DDI_SERVING_URLS", "")
SERVING_URLS: list = (
    [u.strip() for u in _serving_urls_raw.split(",") if u.strip()]
    if _serving_urls_raw
    else ([SERVING_URL] if SERVING_URL else [])
)

# ── 배포 보존 정책 ──────────────────────────────────────────────────────────────
BACKUP_KEEP_N = max(1, int(os.environ.get("DDI_BACKUP_KEEP_N", "5")))
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
python3 -m pytest tests/test_integration/test_env_contract.py -v --tb=short 2>&1 | tail -10
```

Expected: 전체 PASSED

- [ ] **Step 5: serving/main.py에서 os.environ 직접 참조 제거**

`serving/main.py`의 로깅 설정 블록 교체:

```python
# 변경 전
logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
```

```python
# 변경 후 (os.environ 직접 참조 제거)
logging.basicConfig(
    level=getattr(logging, _settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
```

`serving/main.py`의 CORS 블록 교체:

```python
# 변경 전
_cors_origins_env = os.environ.get("CORS_ORIGINS", "")
_cors_origins = [o.strip() for o in _cors_origins_env.split(",") if o.strip()]
if not _cors_origins:
    logger.info("CORS_ORIGINS 미설정 — 외부 오리진 차단")
```

```python
# 변경 후
_cors_origins_env = _settings.CORS_ORIGINS
_cors_origins = [o.strip() for o in _cors_origins_env.split(",") if o.strip()]
if not _cors_origins:
    logger.info("CORS_ORIGINS 미설정 — 외부 오리진 차단")
```

`MODEL_PATH`는 프로세스 시작 시 1회 오버라이드용 런타임 변수로 settings.py 에 포함하지 않는다. 해당 줄에 주석 추가:

```python
# MODEL_PATH: 런타임 오버라이드 전용 — settings.py에 포함하지 않음
_model_path = os.environ.get("MODEL_PATH") or str(_settings.MODEL_PROD_PATH)
```

- [ ] **Step 6: CORS 기존 테스트 통과 확인**

```bash
python3 -m pytest tests/test_serving/test_admin_cors.py::test_cors_default_is_not_wildcard -v
```

Expected: PASSED

- [ ] **Step 7: 커밋**

```bash
git add config/settings.py serving/main.py tests/test_integration/test_env_contract.py
git commit -m "feat: config.settings에 LOG_LEVEL·CORS_ORIGINS·SERVING_URLS·BACKUP_KEEP_N 추가"
```

---

### Task 2: HIGH — Phase 4 번들 원자성: symlink 기반 버전 디렉터리 교체

**Files:**
- Modify: `dags/ddi_train_dag.py`
- Modify: `config/settings.py` (MODEL_PROD_PATH 경로 변경)
- Modify: `tests/test_integration/test_deploy_integrity.py` (어설션 업데이트)

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_integration/test_deploy_integrity.py`의 `test_deploy_success_creates_all_files`를 아래로 교체:

```python
def test_deploy_success_creates_all_files(tmp_path):
    """완전한 아티팩트로 배포 시 current 심링크 + 버전 디렉터리에 파일 전부 생성."""
    staging = tmp_path / "staging"
    staging.mkdir()
    prod_dir = tmp_path / "models"
    prod_dir.mkdir()

    main_pkl = _make_full_artifacts(staging)

    import config.settings as s
    s.MODEL_DIR = prod_dir
    s.MODEL_PROD_PATH = prod_dir / "current" / "model_prod.pkl"
    s.SERVING_URL = "http://localhost:9999"
    s.SERVING_URLS = ["http://localhost:9999"]
    s.ADMIN_API_KEY = "test-key"

    def fake_post(url, **kw):
        r = MagicMock()
        r.raise_for_status.return_value = None
        r.json.return_value = {"status": "ok"}
        return r

    with patch("requests.post", side_effect=fake_post):
        from dags.ddi_train_dag import _deploy_model
        _deploy_model(ti=_FakeTI(str(main_pkl)))

    # current 심링크가 존재하고 버전 디렉터리를 가리켜야 함
    current = prod_dir / "current"
    assert current.is_symlink(), "current 심링크 없음"
    assert (current / "model_prod.pkl").exists()
    assert (current / "model_prod.pkl.sha256").exists()
    assert (current / "model_prod.xgb.pkl").exists()
    assert (current / "model_prod.xgb.pkl.sha256").exists()
    assert (current / "model_prod.lgb.pkl").exists()
    assert (current / "model_prod.lgb.pkl.sha256").exists()
```

`test_deploy_atomic_no_files_on_missing_submodel_sha256` 어설션도 교체 (기존 glob 체크 → current 심링크 체크):

```python
    # 기존: assert list(prod_dir.glob("model_prod*")) == []
    assert not (prod_dir / "current").exists(), \
        "RuntimeError 발생 전 current 심링크가 생성됨 — 원자성 깨짐"
    assert not list(prod_dir.glob(".v_*")), \
        "RuntimeError 발생 전 버전 디렉터리가 생성됨 — 원자성 깨짐"
```

`test_deploy_atomic_no_files_on_missing_main_sha256` 도 동일하게 교체:

```python
    # 기존: assert list(prod_dir.glob("model_prod*")) == []
    assert not (prod_dir / "current").exists()
    assert not list(prod_dir.glob(".v_*"))
```

`test_deploy_backup_covers_all_files`의 setup 및 어설션 교체:

```python
def test_deploy_backup_covers_all_files(tmp_path):
    """배포 성공 시 backup/ 에 기존 model_prod* 전체 보관."""
    staging = tmp_path / "staging"
    staging.mkdir()
    prod_dir = tmp_path / "models"
    prod_dir.mkdir()

    # 기존 prod: .v_old 버전 디렉터리 + current 심링크
    old_dir = prod_dir / ".v_old"
    old_dir.mkdir()
    (old_dir / "model_prod.pkl").write_bytes(b"old_model")
    (old_dir / "model_prod.pkl.sha256").write_text("oldhash\n")
    (old_dir / "model_prod.xgb.pkl").write_bytes(b"old_xgb")
    (old_dir / "model_prod.xgb.pkl.sha256").write_text("oldhash\n")
    (prod_dir / "current").symlink_to(".v_old")

    main_pkl = _make_full_artifacts(staging)

    import config.settings as s
    s.MODEL_DIR = prod_dir
    s.MODEL_PROD_PATH = prod_dir / "current" / "model_prod.pkl"
    s.SERVING_URL = "http://localhost:9999"
    s.SERVING_URLS = ["http://localhost:9999"]
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
```

기존 `test_hotswap_failure_raises`와 `test_hotswap_timeout_raises`에도 `s.SERVING_URLS = [s.SERVING_URL]` 추가:

```python
    s.SERVING_URL = "http://localhost:9999"
    s.SERVING_URLS = ["http://localhost:9999"]  # 추가
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
python3 -m pytest tests/test_integration/test_deploy_integrity.py -v --tb=short 2>&1 | tail -15
```

Expected: 여러 FAILED (current symlink not found, 기존 코드가 flat 구조 사용)

- [ ] **Step 3: config/settings.py MODEL_PROD_PATH 경로 변경**

```python
# 변경 전
MODEL_PROD_PATH   = MODEL_DIR / "model_prod.pkl"

# 변경 후
MODEL_PROD_PATH   = MODEL_DIR / "current" / "model_prod.pkl"
```

- [ ] **Step 4: ddi_train_dag.py — _atomic_symlink_update 헬퍼 추가 및 Phase 3·4 재작성**

`_deploy_model` 함수 바로 위(136번째 줄 앞)에 헬퍼 함수 추가:

```python
def _atomic_symlink_update(link_path, target_name: str) -> None:
    """link_path 심링크를 target_name으로 원자적으로 교체.

    target_name은 link_path.parent 내 상대 이름이어야 한다.
    tmp 심링크 생성 후 os.replace로 원자 교체한다 (POSIX 보장).
    """
    from pathlib import Path
    tmp = Path(str(link_path) + "._tmp")
    if tmp.is_symlink():
        tmp.unlink()
    tmp.symlink_to(target_name)
    import os as _os
    _os.replace(str(tmp), str(link_path))
```

`_deploy_model` 함수의 import 블록에 `import time` 추가:

```python
    import sys
    sys.path.insert(0, "/app")
    import os
    import time                    # 추가
    import logging
    import shutil
    import requests
    from pathlib import Path
    from config import settings as _s
```

`prod_path` 계산 라인 교체:

```python
    # 변경 전
    prod_path  = prod_dir / "model_prod.pkl"
    
    # 변경 후
    prod_path  = prod_dir / "current" / "model_prod.pkl"
```

Phase 3 (백업) 전체 교체:

```python
    # ── Phase 3: 기존 prod 전체 백업 ─────────────────────────────────────────
    backup_dir = prod_dir / "backup"
    backup_dir.mkdir(exist_ok=True)
    current_link = prod_dir / "current"
    if current_link.is_symlink() or current_link.is_dir():
        src_dir = current_link.resolve() if current_link.is_symlink() else current_link
        for f in src_dir.iterdir():
            if f.is_file():
                shutil.copy2(f, backup_dir / f.name)
```

Phase 4 (원자적 promote) 전체 교체:

```python
    # ── Phase 4: 버전 디렉터리 rename + current 심링크 원자 교체 ──────────────
    versioned_name = f".v_{int(time.time())}"
    versioned_dir = prod_dir / versioned_name
    os.rename(str(tmp_dir), str(versioned_dir))          # 디렉터리 단위 원자 rename
    _atomic_symlink_update(prod_dir / "current", versioned_name)  # 심링크 원자 교체
```

- [ ] **Step 5: 테스트 통과 확인**

```bash
python3 -m pytest tests/test_integration/test_deploy_integrity.py -v --tb=short 2>&1 | tail -20
```

Expected: `test_deploy_success_creates_all_files`, `test_deploy_backup_covers_all_files` PASSED  
(hotswap 실패 테스트는 Task 3에서 처리)

- [ ] **Step 6: 전체 테스트 스위트 확인**

```bash
python3 -m pytest tests/ -v --tb=short 2>&1 | tail -10
```

Expected: 전체 PASSED (일부 hotswap 테스트는 Task 3 전까지 adjusted 여부 확인)

- [ ] **Step 7: 커밋**

```bash
git add config/settings.py dags/ddi_train_dag.py tests/test_integration/test_deploy_integrity.py
git commit -m "feat: Phase 4 번들 원자성 — symlink 기반 버전 디렉터리 교체 (HIGH #1)"
```

---

### Task 3: HIGH — Hotswap 실패 시 자동 롤백 + SERVING_URLS 브로드캐스트

**Files:**
- Modify: `dags/ddi_train_dag.py` (hotswap 섹션)
- Modify: `tests/test_integration/test_deploy_integrity.py` (rollback 테스트 추가)

- [ ] **Step 1: rollback 테스트 추가**

`tests/test_integration/test_deploy_integrity.py` 끝에 추가:

```python
def test_hotswap_failure_rolls_back_current_symlink(tmp_path):
    """핫스왑 실패 시 current 심링크가 이전 버전(backup)으로 복원된다."""
    staging = tmp_path / "staging"
    staging.mkdir()
    prod_dir = tmp_path / "models"
    prod_dir.mkdir()

    # 기존 prod: .v_old + current 심링크
    old_dir = prod_dir / ".v_old"
    old_dir.mkdir()
    (old_dir / "model_prod.pkl").write_bytes(b"old_model")
    (old_dir / "model_prod.pkl.sha256").write_text("oldhash\n")
    (prod_dir / "current").symlink_to(".v_old")

    main_pkl = _make_full_artifacts(staging)

    import config.settings as s
    s.MODEL_DIR = prod_dir
    s.MODEL_PROD_PATH = prod_dir / "current" / "model_prod.pkl"
    s.SERVING_URL = "http://localhost:9999"
    s.SERVING_URLS = ["http://localhost:9999"]
    s.ADMIN_API_KEY = "key"

    import requests as _req
    def fake_post_503(url, **kw):
        r = MagicMock()
        r.raise_for_status.side_effect = _req.HTTPError("503")
        return r

    with patch("requests.post", side_effect=fake_post_503):
        from dags.ddi_train_dag import _deploy_model
        with pytest.raises(RuntimeError, match="핫스왑 실패"):
            _deploy_model(ti=_FakeTI(str(main_pkl)))

    current = prod_dir / "current"
    assert current.is_symlink(), "current 심링크 없음"
    assert (current / "model_prod.pkl").read_bytes() == b"old_model", \
        "롤백 실패: current가 여전히 새 버전을 가리킴"


def test_hotswap_multi_url_all_called(tmp_path):
    """SERVING_URLS 의 모든 인스턴스에 reload 요청을 보낸다."""
    staging = tmp_path / "staging"
    staging.mkdir()
    prod_dir = tmp_path / "models"
    prod_dir.mkdir()
    main_pkl = _make_full_artifacts(staging)

    import config.settings as s
    s.MODEL_DIR = prod_dir
    s.MODEL_PROD_PATH = prod_dir / "current" / "model_prod.pkl"
    s.SERVING_URL = "http://inst1:8000"
    s.SERVING_URLS = ["http://inst1:8000", "http://inst2:8000"]
    s.ADMIN_API_KEY = "key"

    called_urls: list[str] = []
    def fake_post(url, **kw):
        called_urls.append(url)
        r = MagicMock()
        r.raise_for_status.return_value = None
        r.json.return_value = {"status": "ok"}
        return r

    with patch("requests.post", side_effect=fake_post):
        from dags.ddi_train_dag import _deploy_model
        _deploy_model(ti=_FakeTI(str(main_pkl)))

    assert "http://inst1:8000/admin/reload" in called_urls
    assert "http://inst2:8000/admin/reload" in called_urls
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
python3 -m pytest tests/test_integration/test_deploy_integrity.py::test_hotswap_failure_rolls_back_current_symlink tests/test_integration/test_deploy_integrity.py::test_hotswap_multi_url_all_called -v --tb=short
```

Expected: 2건 FAILED

- [ ] **Step 3: _deploy_model hotswap 섹션 전체 교체**

기존 `# ── Serving 핫스왑` 블록을 아래로 교체:

```python
    # ── Serving 핫스왑 (전체 인스턴스 브로드캐스트) ──────────────────────────
    failed_urls: list = []
    for url in _s.SERVING_URLS:
        try:
            resp = requests.post(
                f"{url}/admin/reload",
                json={"model_path": str(prod_path)},
                headers={"X-Admin-Key": _s.ADMIN_API_KEY},
                timeout=30,
            )
            resp.raise_for_status()
            logging.info("핫스왑 완료 [%s]: %s", url, resp.json())
        except Exception as exc:
            logging.warning("핫스왑 실패 [%s]: %s", url, exc)
            failed_urls.append(url)

    if failed_urls:
        # 디스크 롤백: backup/ 파일을 새 versioned 디렉터리로 복원 후 current 심링크 교체
        try:
            rollback_name = f".v_rollback_{int(time.time())}"
            rollback_dir = prod_dir / rollback_name
            rollback_dir.mkdir(parents=True, exist_ok=True)
            for f in backup_dir.iterdir():
                if f.is_file():
                    shutil.copy2(f, rollback_dir / f.name)
            _atomic_symlink_update(prod_dir / "current", rollback_name)
            logging.warning("디스크 롤백 완료: %s → %s", versioned_name, rollback_name)
        except Exception as rb_exc:
            logging.error("롤백 실패 (수동 복구 필요): %s", rb_exc)
        raise RuntimeError(
            f"핫스왑 실패 — 롤백 완료, 구버전 모델 서빙 중. 실패 인스턴스: {failed_urls}"
        )
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
python3 -m pytest tests/test_integration/test_deploy_integrity.py -v --tb=short 2>&1 | tail -15
```

Expected: 전체 PASSED

- [ ] **Step 5: 전체 테스트 스위트 확인**

```bash
python3 -m pytest tests/ --tb=short 2>&1 | tail -5
```

Expected: N passed, 0 failed

- [ ] **Step 6: 커밋**

```bash
git add dags/ddi_train_dag.py tests/test_integration/test_deploy_integrity.py
git commit -m "feat: hotswap 실패 시 자동 롤백 + SERVING_URLS 다중 인스턴스 브로드캐스트 (HIGH #2, MEDIUM #2)"
```

---

### Task 4: MEDIUM — Backup 버전 디렉터리 보존 정책 (keep N)

**Files:**
- Modify: `dags/ddi_train_dag.py` (`_prune_old_versioned_dirs` 추가 + 호출)
- Test: `tests/test_integration/test_deploy_integrity.py`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_integration/test_deploy_integrity.py` 끝에 추가:

```python
def test_prune_keeps_n_versioned_dirs(tmp_path):
    """_prune_old_versioned_dirs는 .v_ 버전 디렉터리를 최근 N개만 보존한다."""
    prod_dir = tmp_path / "models"
    prod_dir.mkdir()

    # .v_ 디렉터리 7개 생성
    import time as _time
    dirs = []
    for i in range(7):
        d = prod_dir / f".v_{1000 + i}"
        d.mkdir()
        (d / "model_prod.pkl").write_bytes(b"x")
        dirs.append(d)
        _time.sleep(0.01)  # mtime 차이 확보

    # current는 마지막 dir을 가리킴
    (prod_dir / "current").symlink_to(dirs[-1].name)

    from dags.ddi_train_dag import _prune_old_versioned_dirs
    _prune_old_versioned_dirs(prod_dir, keep_n=3)

    remaining = sorted([d.name for d in prod_dir.iterdir()
                        if d.is_dir() and d.name.startswith(".v_")])
    assert len(remaining) == 3, f"expected 3, got {remaining}"
    # 최근 3개가 남아야 함
    assert dirs[-1].name in remaining
    assert dirs[-2].name in remaining
    assert dirs[-3].name in remaining
    # 오래된 4개는 삭제되어야 함
    for old_dir in dirs[:-3]:
        assert not old_dir.exists(), f"{old_dir.name} 이 삭제되지 않음"
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
python3 -m pytest tests/test_integration/test_deploy_integrity.py::test_prune_keeps_n_versioned_dirs -v --tb=short
```

Expected: FAILED (ImportError: cannot import name _prune_old_versioned_dirs)

- [ ] **Step 3: _prune_old_versioned_dirs 구현 및 배포 후 호출**

`_atomic_symlink_update` 함수 아래에 추가:

```python
def _prune_old_versioned_dirs(prod_dir, keep_n: int) -> None:
    """prod_dir 내 .v_ 버전 디렉터리를 최근 keep_n 개만 보존하고 나머지 삭제.

    current 심링크가 가리키는 디렉터리는 정책 외로 무조건 보존된다.
    """
    import logging as _log
    import shutil as _shutil
    from pathlib import Path as _Path

    prod = _Path(prod_dir)
    current_target: str | None = None
    current_link = prod / "current"
    if current_link.is_symlink():
        current_target = current_link.resolve().name

    candidates = sorted(
        [d for d in prod.iterdir()
         if d.is_dir() and d.name.startswith(".v_") and d.name != current_target],
        key=lambda d: d.stat().st_mtime,
    )
    for old in candidates[:-keep_n] if len(candidates) > keep_n else []:
        _shutil.rmtree(old, ignore_errors=True)
        _log.info("구버전 디렉터리 정리: %s", old.name)
```

`_deploy_model`의 hotswap 성공 직후(failed_urls가 없을 때) 호출 추가.  
hotswap 섹션의 for 루프 아래, `if failed_urls:` 블록 다음에:

```python
    # ── 구버전 디렉터리 정리 ───────────────────────────────────────────────────
    _prune_old_versioned_dirs(prod_dir, _s.BACKUP_KEEP_N)
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
python3 -m pytest tests/test_integration/test_deploy_integrity.py::test_prune_keeps_n_versioned_dirs -v
```

Expected: PASSED

- [ ] **Step 5: 전체 테스트 스위트 확인**

```bash
python3 -m pytest tests/ --tb=short 2>&1 | tail -5
```

Expected: N passed, 0 failed

- [ ] **Step 6: 커밋**

```bash
git add dags/ddi_train_dag.py tests/test_integration/test_deploy_integrity.py
git commit -m "feat: backup 버전 디렉터리 보존 정책 (BACKUP_KEEP_N, 기본 5개) (MEDIUM #3)"
```

---

### Task 5: 최종 검증 및 머지

**Files:** 없음 (검증만)

- [ ] **Step 1: 전체 테스트 스위트 최종 확인**

```bash
python3 -m pytest tests/ -v 2>&1 | tail -10
```

Expected: N passed, 0 failed (N ≥ 370)

- [ ] **Step 2: feature 브랜치 → main 머지**

```bash
git checkout main
git pull
git merge feature/structural-improvements --no-ff -m "merge: HIGH 2건·MEDIUM 3건 수정 — 번들 원자성·hotswap 롤백·config 완성·다중 인스턴스·backup 보존"
```

- [ ] **Step 3: 머지 후 테스트 재확인**

```bash
python3 -m pytest tests/ --tb=short 2>&1 | tail -5
```

Expected: N passed, 0 failed

- [ ] **Step 4: Codex + Gemini 리뷰 요청**

Claude에게 "완료된 내용에 대해 codex와 gemini에게 의견을 받아 정리해 주세요" 전달.
