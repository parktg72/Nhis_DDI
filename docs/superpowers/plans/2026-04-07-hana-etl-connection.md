# HANA ETL 연결 안정화 + 테이블 검증 Wizard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 폐쇄망 SAP HANA DB에서 ETL → 학습 전체 흐름을 안정적으로 실행할 수 있도록, 세션 격리 연결 관리 + 자동 재연결 + 테이블/컬럼 검증 wizard를 구현한다.

**Architecture:** `db.py` 모듈 레벨 싱글톤을 `session_state` 기반 격리로 교체하고 5초 TTL 캐시가 포함된 `ensure_connected()`를 추가한다. `table_validator.py` 헬퍼 모듈을 신규 생성해 컬럼 매핑 검증 로직을 분리한다. Page 1에 "🔍 테이블 검증" 탭을 추가하고, Page 3에 `validated` 가드와 ETL 예외 상세 표시를 추가한다. Pages 2·5는 `get_connection(st.session_state)` 교체만 수행한다.

**Tech Stack:** Python 3.12 (Windows 타겟), Streamlit ≥ 1.26, hdbcli (HANA), pytest

---

## File Structure

| 파일 | 변경 내용 |
|------|-----------|
| `hana_app/core/db.py` | `get_connection(session_state)` 시그니처 변경, `ensure_connected()` 추가, `_global_conn` → `_fallback_conn` 이름 변경 |
| `hana_app/core/config.py` | `DEFAULT_CONFIG`에 `validated`, `validated_at`, `validated_host` 추가 |
| `hana_app/core/table_validator.py` | **신규** — `check_column_mapping()`, `validate_all_identifiers()` |
| `hana_app/pages/1_🔌_연결_및_테이블설정.py` | line 19·26 교체, 연결 성공 시 `hana_creds` 저장, "🔍 테이블 검증" 탭 추가 |
| `hana_app/pages/2_🔍_데이터_미리보기.py` | line 28 `get_connection()` → `get_connection(st.session_state)` |
| `hana_app/pages/3_🤖_모델_학습.py` | lines 17·29 교체, 페이지 상단 가드 + ETL 예외 표시 교체 |
| `hana_app/pages/5_🗄️_분석DB_관리.py` | line 32 `get_connection()` → `get_connection(st.session_state)` |
| `tests/test_hana_app/__init__.py` | 신규 (빈 파일) |
| `tests/test_hana_app/conftest.py` | `reset_fallback_conn` autouse 픽스처 |
| `tests/test_hana_app/test_db.py` | `get_connection`, `ensure_connected` 단위 테스트 |
| `tests/test_hana_app/test_config.py` | `validated` 플래그 단위 테스트 |
| `tests/test_hana_app/test_table_validator.py` | `check_column_mapping`, `validate_all_identifiers` 단위 테스트 |

---

### Task 1: 테스트 인프라 — `tests/test_hana_app/` 디렉토리 + conftest

**Files:**
- Create: `tests/test_hana_app/__init__.py`
- Create: `tests/test_hana_app/conftest.py`

- [ ] **Step 1: 디렉토리 및 `__init__.py` 생성**

```bash
mkdir tests\test_hana_app
type nul > tests\test_hana_app\__init__.py
```

- [ ] **Step 2: `conftest.py` 작성**

`tests/test_hana_app/conftest.py`:

```python
"""tests/test_hana_app 전용 픽스처."""
import pytest


@pytest.fixture(autouse=True)
def reset_fallback_conn():
    """각 테스트 전후로 _fallback_conn을 새 인스턴스로 교체.

    _fallback_conn은 모듈 레벨 객체이므로 테스트 간 연결 상태가
    누출되는 것을 방지한다.
    """
    from hana_app.core import db as _db_module
    from hana_app.core.db import HANAConnection

    _db_module._fallback_conn = HANAConnection()
    yield
    _db_module._fallback_conn = HANAConnection()
```

- [ ] **Step 3: 픽스처 동작 확인**

```bash
python3 -m pytest tests/test_hana_app/ -v --collect-only
```

Expected: `0 items / 0 errors` (아직 테스트 없음, 수집만 확인)

- [ ] **Step 4: 커밋**

```bash
git add tests/test_hana_app/__init__.py tests/test_hana_app/conftest.py
git commit -m "test: test_hana_app 테스트 디렉토리 + reset_fallback_conn autouse 픽스처"
```

---

### Task 2: `db.py` — `get_connection(session_state)` + `ensure_connected()`

**Files:**
- Modify: `hana_app/core/db.py` (lines 236–240)
- Create: `tests/test_hana_app/test_db.py`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_hana_app/test_db.py`:

```python
"""hana_app/core/db.py 단위 테스트."""
from unittest.mock import patch, MagicMock
import pytest

from hana_app.core.db import HANAConnection, get_connection, _fallback_conn


class TestGetConnection:
    def test_none_returns_fallback(self):
        """session_state=None → _fallback_conn 반환."""
        from hana_app.core import db as _db_module
        result = get_connection(None)
        assert result is _db_module._fallback_conn

    def test_no_arg_returns_fallback(self):
        """인자 없이 호출 → _fallback_conn 반환 (하위 호환)."""
        from hana_app.core import db as _db_module
        result = get_connection()
        assert result is _db_module._fallback_conn

    def test_creates_conn_per_session(self):
        """session_state별로 별도 HANAConnection 생성."""
        s1: dict = {}
        s2: dict = {}
        c1 = get_connection(s1)
        c2 = get_connection(s2)
        assert isinstance(c1, HANAConnection)
        assert isinstance(c2, HANAConnection)
        assert c1 is not c2

    def test_same_session_returns_same_conn(self):
        """동일 session_state는 같은 객체 반환."""
        s: dict = {}
        c1 = get_connection(s)
        c2 = get_connection(s)
        assert c1 is c2

    def test_stores_conn_in_session_state(self):
        """연결 객체가 session_state['hana_conn']에 저장됨."""
        s: dict = {}
        conn = get_connection(s)
        assert s["hana_conn"] is conn


class TestEnsureConnected:
    CREDS = {"host": "h", "port": 30015, "user": "u", "password": "p"}

    def test_reconnects_when_disconnected(self):
        """is_connected()=False → connect() 호출."""
        conn = HANAConnection()
        with patch.object(conn, "is_connected", return_value=False):
            with patch.object(conn, "connect") as mock_connect:
                conn.ensure_connected(self.CREDS)
        mock_connect.assert_called_once_with(
            host="h", port=30015, user="u", password="p"
        )

    def test_skips_when_already_connected(self):
        """is_connected()=True → connect() 미호출."""
        conn = HANAConnection()
        with patch.object(conn, "is_connected", return_value=True):
            with patch.object(conn, "connect") as mock_connect:
                conn.ensure_connected(self.CREDS)
        mock_connect.assert_not_called()

    def test_ttl_cache_skips_is_connected(self):
        """TTL 캐시 유효 시 is_connected() 호출 없이 통과."""
        import time
        conn = HANAConnection()
        session: dict = {"_conn_ok_until": time.monotonic() + 100}
        with patch.object(conn, "is_connected") as mock_check:
            conn.ensure_connected(self.CREDS, session_state=session)
        mock_check.assert_not_called()

    def test_ttl_cache_set_after_connect(self):
        """연결 후 session_state['_conn_ok_until'] 가 미래 시각으로 설정됨."""
        import time
        conn = HANAConnection()
        session: dict = {}
        with patch.object(conn, "is_connected", return_value=False):
            with patch.object(conn, "connect"):
                conn.ensure_connected(self.CREDS, session_state=session, ttl_seconds=5)
        assert session.get("_conn_ok_until", 0) > time.monotonic()

    def test_propagates_connect_exception(self):
        """connect() 실패 시 예외 전파."""
        conn = HANAConnection()
        with patch.object(conn, "is_connected", return_value=False):
            with patch.object(conn, "connect", side_effect=RuntimeError("DB down")):
                with pytest.raises(RuntimeError, match="DB down"):
                    conn.ensure_connected(self.CREDS)
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
python3 -m pytest tests/test_hana_app/test_db.py -v --tb=short 2>&1 | tail -20
```

Expected: 여러 FAILED (`ensure_connected` 미구현, `get_connection` 시그니처 불일치)

- [ ] **Step 3: `db.py` 수정**

`hana_app/core/db.py` 파일 끝 부분(lines 234–240)을 아래로 교체:

```python
# ── 전역 폴백 (테스트 / CLI / 비Streamlit 환경용) ─────────────────────────────
_fallback_conn = HANAConnection()


def get_connection(session_state: dict | None = None) -> HANAConnection:
    """세션별 격리된 HANAConnection 반환.

    session_state가 None이거나 생략되면 _fallback_conn 반환
    (테스트 / CLI / 비Streamlit 환경 하위 호환).
    Streamlit 환경에서는 반드시 st.session_state를 전달한다.
    """
    if session_state is None:
        return _fallback_conn
    if "hana_conn" not in session_state:
        session_state["hana_conn"] = HANAConnection()
    return session_state["hana_conn"]
```

`HANAConnection` 클래스 내부(`close()` 메서드 아래)에 `ensure_connected()` 추가:

```python
    def ensure_connected(
        self,
        creds: dict,
        session_state: dict | None = None,
        ttl_seconds: int = 5,
    ) -> None:
        """연결이 끊겼으면 creds로 자동 재연결.

        creds 구조: {"host": str, "port": int, "user": str, "password": str}

        session_state가 제공되면 TTL 캐시를 사용해 is_connected() DB 왕복을
        ttl_seconds 동안 생략한다 (Streamlit rerun 마다 호출되는 경우 성능 보호).

        이미 연결된 상태면 아무것도 하지 않는다.
        재연결 실패 시 hdbcli 예외를 그대로 전파한다.
        """
        import time

        now = time.monotonic()
        cache_key = "_conn_ok_until"

        # TTL 캐시 유효 → is_connected() 생략
        if session_state is not None:
            if now < session_state.get(cache_key, 0):
                return

        if not self.is_connected():
            self.connect(
                host=creds["host"],
                port=int(creds["port"]),
                user=creds["user"],
                password=creds["password"],
            )

        # 연결 확인 후 TTL 갱신
        if session_state is not None:
            session_state[cache_key] = now + ttl_seconds
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
python3 -m pytest tests/test_hana_app/test_db.py -v --tb=short
```

Expected: `10 passed`

- [ ] **Step 5: 전체 테스트 스위트 회귀 확인**

```bash
python3 -m pytest tests/ --tb=short -q 2>&1 | tail -5
```

Expected: 기존 테스트 모두 통과

- [ ] **Step 6: 커밋**

```bash
git add hana_app/core/db.py tests/test_hana_app/test_db.py
git commit -m "feat: db.py session_state 격리 + ensure_connected() TTL 캐시 (테스트 10건)"
```

---

### Task 3: `config.py` — `validated` / `validated_at` / `validated_host` 플래그

**Files:**
- Modify: `hana_app/core/config.py`
- Create: `tests/test_hana_app/test_config.py`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_hana_app/test_config.py`:

```python
"""hana_app/core/config.py validated 플래그 테스트."""
import importlib
import json
from pathlib import Path

import pytest

from hana_app.core.config import DEFAULT_CONFIG, load_config, save_config


class TestValidatedFlag:
    def test_default_config_has_validated_false(self):
        """DEFAULT_CONFIG에 validated=False가 있다."""
        assert DEFAULT_CONFIG.get("validated") is False

    def test_default_config_has_validated_at(self):
        """DEFAULT_CONFIG에 validated_at 키가 있다 (빈 문자열)."""
        assert "validated_at" in DEFAULT_CONFIG
        assert DEFAULT_CONFIG["validated_at"] == ""

    def test_default_config_has_validated_host(self):
        """DEFAULT_CONFIG에 validated_host 키가 있다 (빈 문자열)."""
        assert "validated_host" in DEFAULT_CONFIG
        assert DEFAULT_CONFIG["validated_host"] == ""

    def test_load_config_merges_validated_keys_if_missing(self, tmp_path, monkeypatch):
        """기존 config 파일에 validated 키 없으면 기본값으로 병합."""
        cfg_file = tmp_path / "hana_config.json"
        # validated 없는 구버전 config
        cfg_file.write_text(json.dumps({"connection": {"host": "h"}}), encoding="utf-8")

        monkeypatch.setattr(
            "hana_app.core.config.CONFIG_FILE", cfg_file
        )
        import hana_app.core.config as _cfg_mod
        importlib.reload(_cfg_mod)
        loaded = _cfg_mod.load_config()
        assert loaded.get("validated") is False
        assert "validated_at" in loaded
        assert "validated_host" in loaded

    def test_save_and_load_validated_true(self, tmp_path, monkeypatch):
        """validated=True로 저장 후 다시 로드하면 True."""
        cfg_file = tmp_path / "hana_config.json"
        monkeypatch.setattr("hana_app.core.config.CONFIG_FILE", cfg_file)
        import hana_app.core.config as _cfg_mod
        importlib.reload(_cfg_mod)

        cfg = _cfg_mod.load_config()
        cfg["validated"] = True
        cfg["validated_at"] = "2026-04-07T09:00:00"
        cfg["validated_host"] = "192.168.1.1"
        _cfg_mod.save_config(cfg)

        loaded = _cfg_mod.load_config()
        assert loaded["validated"] is True
        assert loaded["validated_at"] == "2026-04-07T09:00:00"
        assert loaded["validated_host"] == "192.168.1.1"
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
python3 -m pytest tests/test_hana_app/test_config.py -v --tb=short 2>&1 | tail -15
```

Expected: 여러 FAILED (`validated`, `validated_at`, `validated_host` 키 없음)

- [ ] **Step 3: `config.py` 수정 — `DEFAULT_CONFIG`에 플래그 추가**

`hana_app/core/config.py`의 `DEFAULT_CONFIG` 딕셔너리 내 `"training":` 섹션 바로 위에 추가:

```python
    # ── 테이블 검증 상태 ──────────────────────────────────────────────────
    # Page 1 wizard 완료 시 True. 검증 DB 호스트가 변경되면 False로 초기화.
    "validated":      False,
    "validated_at":   "",    # ISO 8601 (예: "2026-04-07T09:00:00")
    "validated_host": "",    # 검증 시 사용된 HANA 호스트
```

- [ ] **Step 4: `load_config()`에 신규 키 병합 로직 추가**

`load_config()` 함수 내 `# 누락 최상위 키 병합` 블록이 이미 `for key, val in DEFAULT_CONFIG.items(): if key not in data: data[key] = val`를 수행하므로 **추가 코드 없음** — `DEFAULT_CONFIG`에 키를 추가하면 자동으로 병합됩니다.

- [ ] **Step 5: 테스트 통과 확인**

```bash
python3 -m pytest tests/test_hana_app/test_config.py -v --tb=short
```

Expected: `5 passed`

- [ ] **Step 6: 커밋**

```bash
git add hana_app/core/config.py tests/test_hana_app/test_config.py
git commit -m "feat: config.py validated/validated_at/validated_host 플래그 추가 (테스트 5건)"
```

---

### Task 4: `table_validator.py` 신규 생성

**Files:**
- Create: `hana_app/core/table_validator.py`
- Create: `tests/test_hana_app/test_table_validator.py`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_hana_app/test_table_validator.py`:

```python
"""hana_app/core/table_validator.py 단위 테스트."""
import pytest


class TestCheckColumnMapping:
    """check_column_mapping(actual_cols, expected_map) -> {"ok": [...], "missing": [...]}"""

    def test_all_match(self):
        from hana_app.core.table_validator import check_column_mapping
        actual = ["INDI_DSCM_NO", "CMN_KEY", "MDCARE_STRT_DT"]
        expected = {"patient_id": "INDI_DSCM_NO", "bill_no": "CMN_KEY"}
        result = check_column_mapping(actual, expected)
        assert result["ok"] == ["patient_id", "bill_no"]
        assert result["missing"] == []

    def test_some_missing(self):
        from hana_app.core.table_validator import check_column_mapping
        actual = ["INDI_DSCM_NO"]
        expected = {"patient_id": "INDI_DSCM_NO", "bill_no": "CMN_KEY"}
        result = check_column_mapping(actual, expected)
        assert "patient_id" in result["ok"]
        assert "bill_no" in result["missing"]

    def test_all_missing(self):
        from hana_app.core.table_validator import check_column_mapping
        actual = ["OTHER_COL"]
        expected = {"patient_id": "INDI_DSCM_NO"}
        result = check_column_mapping(actual, expected)
        assert result["ok"] == []
        assert "patient_id" in result["missing"]

    def test_empty_expected(self):
        from hana_app.core.table_validator import check_column_mapping
        result = check_column_mapping(["COL_A"], {})
        assert result == {"ok": [], "missing": []}

    def test_empty_actual(self):
        from hana_app.core.table_validator import check_column_mapping
        result = check_column_mapping([], {"patient_id": "INDI_DSCM_NO"})
        assert result["missing"] == ["patient_id"]
        assert result["ok"] == []

    def test_case_sensitive(self):
        """컬럼명 비교는 대소문자를 구분한다."""
        from hana_app.core.table_validator import check_column_mapping
        actual = ["indi_dscm_no"]   # 소문자
        expected = {"patient_id": "INDI_DSCM_NO"}  # 대문자
        result = check_column_mapping(actual, expected)
        assert "patient_id" in result["missing"]


class TestValidateAllIdentifiers:
    """validate_all_identifiers(column_map) — 안전하지 않은 식별자 있으면 ValueError."""

    def test_all_safe(self):
        from hana_app.core.table_validator import validate_all_identifiers
        # 예외 없어야 함
        validate_all_identifiers({"patient_id": "INDI_DSCM_NO", "bill_no": "CMN_KEY"})

    def test_unsafe_value_raises(self):
        from hana_app.core.table_validator import validate_all_identifiers
        with pytest.raises(ValueError, match="안전하지 않은"):
            validate_all_identifiers({"patient_id": "col'; DROP TABLE--"})

    def test_unsafe_key_raises(self):
        from hana_app.core.table_validator import validate_all_identifiers
        with pytest.raises(ValueError):
            validate_all_identifiers({"bad key!": "INDI_DSCM_NO"})

    def test_empty_map_passes(self):
        from hana_app.core.table_validator import validate_all_identifiers
        validate_all_identifiers({})  # 예외 없어야 함

    def test_dollar_and_hash_allowed(self):
        """HANA는 $·# 허용."""
        from hana_app.core.table_validator import validate_all_identifiers
        validate_all_identifiers({"col_a": "COL$NAME", "col_b": "COL#2"})
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
python3 -m pytest tests/test_hana_app/test_table_validator.py -v --tb=short 2>&1 | tail -15
```

Expected: 여러 FAILED (모듈 없음)

- [ ] **Step 3: `table_validator.py` 구현**

`hana_app/core/table_validator.py` (신규):

```python
"""HANA 테이블·컬럼 매핑 검증 헬퍼.

Page 1 wizard에서 호출되며, Streamlit 없이도 단위 테스트 가능하도록
순수 Python 함수로 작성한다.
"""
from __future__ import annotations

from hana_app.core.db import _assert_safe_identifier


def check_column_mapping(
    actual_cols: list[str],
    expected_map: dict[str, str],
) -> dict[str, list[str]]:
    """논리명 → 실제 DB 컬럼명 매핑을 검증한다.

    Args:
        actual_cols: DB에서 조회한 실제 컬럼명 목록.
        expected_map: {논리명: 기대 DB 컬럼명} 딕셔너리.

    Returns:
        {"ok": [일치한 논리명 목록], "missing": [불일치 논리명 목록]}
    """
    actual_set = set(actual_cols)
    ok: list[str] = []
    missing: list[str] = []
    for logical_name, db_col in expected_map.items():
        if db_col in actual_set:
            ok.append(logical_name)
        else:
            missing.append(logical_name)
    return {"ok": ok, "missing": missing}


def validate_all_identifiers(column_map: dict[str, str]) -> None:
    """column_map의 모든 키·값이 HANA 안전 식별자인지 검증한다.

    _assert_safe_identifier()를 통과하지 못하는 항목이 있으면 ValueError 발생.
    저장 직전 일괄 호출해 SQL 인젝션 방어선으로 사용한다.

    Args:
        column_map: {논리명: DB 컬럼명} 딕셔너리.

    Raises:
        ValueError: 안전하지 않은 식별자가 포함된 경우.
    """
    for logical_name, db_col in column_map.items():
        _assert_safe_identifier(logical_name, "논리명")
        _assert_safe_identifier(db_col, "DB 컬럼명")
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
python3 -m pytest tests/test_hana_app/test_table_validator.py -v --tb=short
```

Expected: `11 passed`

- [ ] **Step 5: 커밋**

```bash
git add hana_app/core/table_validator.py tests/test_hana_app/test_table_validator.py
git commit -m "feat: table_validator.py — check_column_mapping + validate_all_identifiers (테스트 11건)"
```

---

### Task 5: Page 1 — 연결 코드 교체 + `hana_creds` 저장

**Files:**
- Modify: `hana_app/pages/1_🔌_연결_및_테이블설정.py` (lines 19, 26, 107–130)

- [ ] **Step 1: import 및 모듈 레벨 `get_connection()` 교체**

`hana_app/pages/1_🔌_연결_및_테이블설정.py` line 19를 교체:

```python
# 변경 전
from hana_app.core.db import get_connection

# 변경 후
from hana_app.core.db import get_connection
```
*(import는 동일 — 시그니처만 변경)*

line 26을 교체:

```python
# 변경 전
conn = get_connection()

# 변경 후
conn = get_connection(st.session_state)
```

- [ ] **Step 2: 연결 성공 시 `hana_creds` 저장**

line 117 (`st.session_state.connected = True`) 바로 아래에 추가:

```python
                            st.session_state.connected = True
                            # ↓ 추가: 자동 재연결용 creds 저장 (비밀번호는 session_state에만 보관)
                            st.session_state["hana_creds"] = {
                                "host": host,
                                "port": int(port),
                                "user": user,
                                "password": password,
                            }
```

- [ ] **Step 3: 앱 기동 확인 (수동)**

```bash
python3 -m streamlit run hana_app/main.py
```

1번 페이지 진입 → 오류 없이 렌더링 확인. HANA 연결 테스트 버튼 클릭 후 `st.session_state`에 `hana_creds` 저장 여부 확인 (Streamlit 디버그 모드 또는 `st.write(st.session_state)` 임시 추가).

- [ ] **Step 4: 커밋**

```bash
git add "hana_app/pages/1_🔌_연결_및_테이블설정.py"
git commit -m "feat: page1 get_connection(session_state) 교체 + hana_creds session_state 저장"
```

---

### Task 6: Page 1 — "🔍 테이블 검증" 탭 추가 (wizard 4단계)

**Files:**
- Modify: `hana_app/pages/1_🔌_연결_및_테이블설정.py`

탭 정의 부분(line 57)을 교체하고 새 탭 블록을 추가합니다.

- [ ] **Step 1: 탭 정의에 "🔍 테이블 검증" 추가**

line 57–62를 교체:

```python
# 변경 전
tab_hana, tab_sas, tab_tbl, tab_col = st.tabs([
    "🗄️ HANA DB 연결",
    "📂 SAS 파일 설정",
    "📋 테이블 위치 (HANA)",
    "🗂️ 컬럼 매핑",
])

# 변경 후
tab_hana, tab_sas, tab_tbl, tab_col, tab_validate = st.tabs([
    "🗄️ HANA DB 연결",
    "📂 SAS 파일 설정",
    "📋 테이블 위치 (HANA)",
    "🗂️ 컬럼 매핑",
    "🔍 테이블 검증",
])
```

import 블록 상단에 추가:

```python
import datetime

from hana_app.core.table_validator import check_column_mapping, validate_all_identifiers
```

- [ ] **Step 2: `tab_validate` 블록 추가 (파일 끝에 append)**

SAS 설정 저장 블록(`st.button("💾 SAS 설정 저장"`) 이후 파일 끝에 추가:

```python

# ─────────────────────────────────────────────────────────────────────────────
# 탭 5: 테이블 검증 Wizard (HANA 전용)
# ─────────────────────────────────────────────────────────────────────────────
with tab_validate:
    st.subheader("🔍 HANA 테이블 검증")
    st.caption(
        "실제 HANA DB의 테이블·컬럼이 학습 코드와 일치하는지 확인합니다. "
        "3번 페이지(모델 학습)에서 데이터를 추출하기 전에 반드시 완료해야 합니다."
    )

    if not (st.session_state.get("connected") and conn.is_connected()):
        st.warning("⚠️ HANA DB에 먼저 연결하세요. (🗄️ HANA DB 연결 탭)")
        st.stop()

    # ── 현재 검증 상태 표시 ───────────────────────────────────────────────
    if cfg.get("validated"):
        st.success(
            f"✅ 검증 완료  |  "
            f"{cfg.get('validated_at', '')}  |  "
            f"호스트: {cfg.get('validated_host', '')}"
        )
        if cfg.get("validated_host") and cfg["validated_host"] != cfg["connection"]["host"]:
            st.warning(
                "⚠️ 검증된 호스트와 현재 연결 호스트가 다릅니다. 재검증을 권장합니다."
            )
    else:
        st.info("ℹ️ 아직 검증되지 않았습니다. 아래 단계를 순서대로 진행하세요.")

    st.markdown("---")

    TABLE_LOGICAL = {
        "t20":    "T20 (요양명세서)",
        "t30":    "T30 (원내 약품)",
        "t40":    "T40 (상병내역)",
        "t60":    "T60 (원외처방)",
        "yoyang": "요양기관",
    }

    # ── Step 1: 스키마 선택 ───────────────────────────────────────────────
    st.markdown("#### Step 1: 스키마 선택")

    col_refresh1, _ = st.columns([1, 5])
    with col_refresh1:
        if st.button("🔄 스키마 새로고침", key="refresh_schemas"):
            st.session_state.pop("_wizard_schemas", None)

    if "_wizard_schemas" not in st.session_state:
        with st.spinner("스키마 목록 조회 중..."):
            try:
                st.session_state["_wizard_schemas"] = conn.get_schemas()
            except Exception as e:
                st.error(f"❌ 스키마 조회 실패: {e}")
                st.stop()

    schema_list = st.session_state["_wizard_schemas"]
    if not schema_list:
        st.error("❌ 접근 가능한 스키마가 없습니다. 계정 권한을 확인하세요.")
        st.stop()

    schema_selections: dict[str, str] = {}
    cols_s = st.columns(len(TABLE_LOGICAL))
    for (tbl_key, tbl_label), col in zip(TABLE_LOGICAL.items(), cols_s):
        current_schema = cfg["tables"].get(tbl_key, {}).get("schema", "")
        default_idx = schema_list.index(current_schema) if current_schema in schema_list else 0
        with col:
            schema_selections[tbl_key] = st.selectbox(
                tbl_label,
                options=schema_list,
                index=default_idx,
                key=f"wiz_schema_{tbl_key}",
            )

    # ── Step 2: 테이블 선택 ───────────────────────────────────────────────
    st.markdown("#### Step 2: 테이블 선택")

    table_selections: dict[str, str] = {}
    for tbl_key, tbl_label in TABLE_LOGICAL.items():
        schema = schema_selections[tbl_key]
        cache_key = f"_wizard_tables_{schema}"
        col_lbl, col_sel, col_ref = st.columns([1, 3, 1])
        with col_ref:
            if st.button("🔄", key=f"refresh_tbl_{tbl_key}", help="테이블 목록 새로고침"):
                st.session_state.pop(cache_key, None)
        if cache_key not in st.session_state:
            with st.spinner(f"{schema} 테이블 목록 조회 중..."):
                try:
                    st.session_state[cache_key] = conn.get_tables(schema)
                except Exception as e:
                    st.session_state[cache_key] = []
                    st.error(f"❌ {schema} 테이블 조회 실패: {e}")
        tbl_list = st.session_state[cache_key]
        current_tbl = cfg["tables"].get(tbl_key, {}).get("table", "")
        default_idx = tbl_list.index(current_tbl) if current_tbl in tbl_list else 0
        with col_lbl:
            st.markdown(f"**{tbl_label}**")
        with col_sel:
            if tbl_list:
                table_selections[tbl_key] = st.selectbox(
                    "테이블",
                    options=tbl_list,
                    index=default_idx,
                    key=f"wiz_table_{tbl_key}",
                    label_visibility="collapsed",
                )
            else:
                st.error(f"❌ {schema} 에 테이블이 없습니다")
                table_selections[tbl_key] = ""

    # ── Step 3: 컬럼 매핑 검증 ───────────────────────────────────────────
    st.markdown("#### Step 3: 컬럼 매핑 검증")
    st.caption("ETL에 필요한 컬럼만 검증합니다. 🔴 항목은 드롭다운으로 실제 컬럼을 선택하세요.")

    from hana_app.core.config import DEFAULT_TABLE_COLS

    # wizard에서 선택한 컬럼 매핑 (논리명 → 실제 DB 컬럼명)
    # DEFAULT_TABLE_COLS를 기본값으로 사용
    updated_col_map: dict[str, dict[str, str]] = {}

    for tbl_key, tbl_label in TABLE_LOGICAL.items():
        schema = schema_selections[tbl_key]
        table = table_selections.get(tbl_key, "")
        if not table:
            continue

        cache_key = f"_wizard_cols_{schema}_{table}"
        if cache_key not in st.session_state:
            with st.spinner(f"{table} 컬럼 조회 중..."):
                try:
                    col_info = conn.get_columns(schema, table)
                    st.session_state[cache_key] = [c["name"] for c in col_info]
                except Exception as e:
                    st.session_state[cache_key] = []
                    st.error(f"❌ {table} 컬럼 조회 실패: {e}")

        actual_cols = st.session_state[cache_key]
        expected_map: dict[str, str] = cfg.get("columns", DEFAULT_TABLE_COLS).get(tbl_key, {})
        if not expected_map:
            expected_map = DEFAULT_TABLE_COLS.get(tbl_key, {})

        check_result = check_column_mapping(actual_cols, expected_map)

        with st.expander(
            f"**{tbl_label}** — "
            f"✅ {len(check_result['ok'])}개 일치 / "
            f"{'🔴 ' + str(len(check_result['missing'])) + '개 불일치' if check_result['missing'] else '전체 일치'}",
            expanded=bool(check_result["missing"]),
        ):
            tbl_col_map: dict[str, str] = {}
            for logical_name, db_col in expected_map.items():
                status = "✅" if logical_name in check_result["ok"] else "🔴"
                c1, c2, c3 = st.columns([1, 2, 2])
                with c1:
                    st.markdown(status)
                with c2:
                    st.markdown(f"`{logical_name}`")
                with c3:
                    if logical_name in check_result["ok"]:
                        st.markdown(f"`{db_col}`")
                        tbl_col_map[logical_name] = db_col
                    else:
                        # 불일치: 실제 컬럼 중에서 선택
                        opts = actual_cols if actual_cols else ["(컬럼 없음)"]
                        sel = st.selectbox(
                            f"{logical_name} 대체 컬럼",
                            options=opts,
                            key=f"wiz_col_{tbl_key}_{logical_name}",
                            label_visibility="collapsed",
                        )
                        tbl_col_map[logical_name] = sel

            updated_col_map[tbl_key] = tbl_col_map

    # ── Step 4: 저장 ─────────────────────────────────────────────────────
    st.markdown("#### Step 4: 저장")

    col_save, col_revalidate = st.columns(2)
    with col_save:
        if st.button("✅ 검증 완료 & 저장", type="primary", use_container_width=True):
            # 저장 전 일괄 식별자 재검증
            try:
                for tbl_key, col_map in updated_col_map.items():
                    validate_all_identifiers(col_map)
            except ValueError as e:
                st.error(f"❌ 식별자 검증 실패: {e}")
                st.stop()

            # config 업데이트
            for tbl_key, tbl_label in TABLE_LOGICAL.items():
                cfg["tables"][tbl_key] = {
                    "schema": schema_selections.get(tbl_key, cfg["tables"].get(tbl_key, {}).get("schema", "")),
                    "table":  table_selections.get(tbl_key, cfg["tables"].get(tbl_key, {}).get("table", "")),
                }
                if tbl_key in updated_col_map:
                    if "columns" not in cfg:
                        cfg["columns"] = {}
                    cfg["columns"][tbl_key] = updated_col_map[tbl_key]

            cfg["validated"] = True
            cfg["validated_at"] = datetime.datetime.now().isoformat(timespec="seconds")
            cfg["validated_host"] = cfg["connection"]["host"]
            save_config(cfg)
            st.success("✅ 검증 완료 — 3번 페이지에서 학습을 시작할 수 있습니다.")
            st.rerun()

    with col_revalidate:
        if cfg.get("validated") and st.button(
            "🔄 재검증", use_container_width=True,
            help="DB 스키마 변경 후 재검증"
        ):
            cfg["validated"] = False
            save_config(cfg)
            for key in list(st.session_state.keys()):
                if key.startswith("_wizard_"):
                    del st.session_state[key]
            st.rerun()
```

- [ ] **Step 3: 앱 기동 확인 (수동)**

```bash
python3 -m streamlit run hana_app/main.py
```

1번 페이지 → "🔍 테이블 검증" 탭이 보임 확인. HANA 미연결 상태에서 탭 진입 시 경고 메시지 표시 확인.

- [ ] **Step 4: 커밋**

```bash
git add "hana_app/pages/1_🔌_연결_및_테이블설정.py"
git commit -m "feat: page1 테이블 검증 wizard 4단계 추가 (스키마→테이블→컬럼→저장)"
```

---

### Task 7: Pages 2, 5 — `get_connection(st.session_state)` 교체

**Files:**
- Modify: `hana_app/pages/2_🔍_데이터_미리보기.py` (line 28)
- Modify: `hana_app/pages/5_🗄️_분석DB_관리.py` (line 32)

- [ ] **Step 1: Page 2 교체**

`hana_app/pages/2_🔍_데이터_미리보기.py` line 28:

```python
# 변경 전
conn = get_connection()

# 변경 후
conn = get_connection(st.session_state)
```

- [ ] **Step 2: Page 5 교체**

`hana_app/pages/5_🗄️_분석DB_관리.py` line 32:

```python
# 변경 전
conn = get_connection()

# 변경 후
conn = get_connection(st.session_state)
```

- [ ] **Step 3: 앱 기동 확인 (수동)**

```bash
python3 -m streamlit run hana_app/main.py
```

2번·5번 페이지 진입 시 오류 없이 렌더링 확인.

- [ ] **Step 4: 커밋**

```bash
git add "hana_app/pages/2_🔍_데이터_미리보기.py" "hana_app/pages/5_🗄️_분석DB_관리.py"
git commit -m "feat: page2, page5 get_connection(st.session_state) 교체"
```

---

### Task 8: Page 3 — 가드 로직 + ETL 예외 상세 표시

**Files:**
- Modify: `hana_app/pages/3_🤖_모델_학습.py` (lines 17, 29, ETL 호출 블록)

- [ ] **Step 1: import 및 `get_connection()` 교체**

line 17 (import 블록) — `get_connection` 옆에 아무것도 추가 안 해도 됨, 시그니처만 바뀜.

line 29를 교체:

```python
# 변경 전
conn = get_connection()

# 변경 후
conn = get_connection(st.session_state)
```

- [ ] **Step 2: 페이지 상단 가드 추가**

line 29 (`conn = get_connection(st.session_state)`) 바로 아래에 추가:

```python
# ── validated 가드 ────────────────────────────────────────────────────────
_cfg_for_guard = load_config()
if is_hana(_cfg_for_guard) and not _cfg_for_guard.get("validated"):
    st.warning("⚠️ HANA 테이블 검증이 완료되지 않았습니다.")
    st.page_link(
        "pages/1_🔌_연결_및_테이블설정.py",
        label="👉 1번 페이지 → 🔍 테이블 검증 탭에서 완료 후 돌아오세요",
    )
    st.stop()

if is_hana(_cfg_for_guard):
    if _cfg_for_guard.get("validated_host") and \
       _cfg_for_guard["validated_host"] != _cfg_for_guard["connection"]["host"]:
        st.warning("⚠️ 검증된 DB 호스트와 현재 연결 호스트가 다릅니다. 1번 페이지에서 재검증을 권장합니다.")

# ── 자동 재연결 ───────────────────────────────────────────────────────────
_hana_creds = st.session_state.get("hana_creds")
if is_hana(_cfg_for_guard):
    if _hana_creds:
        try:
            conn.ensure_connected(_hana_creds, session_state=st.session_state)
        except Exception as _conn_err:
            st.error(f"❌ DB 재연결 실패: {_conn_err}")
            st.stop()
    elif not conn.is_connected():
        st.error("❌ DB 연결이 없습니다. 1번 페이지에서 먼저 연결하세요.")
        st.stop()
```

- [ ] **Step 3: ETL 예외 블록에 `st.exception()` 추가**

Page 3에서 `extract_prescriptions` 또는 `build_patient_features`를 호출하는 `try/except` 블록을 찾아 교체합니다. 현재 코드에서 `with st.spinner` 내부 예외가 묻히는 블록을 아래 패턴으로 교체:

```python
# 패턴: 기존 except 블록 교체 (단일 예외 처리)
# 변경 전
    except Exception as e:
        st.error(f"오류: {e}")

# 변경 후
    except Exception as e:
        st.error("❌ 데이터 추출 실패")
        st.exception(e)
        st.info("💡 오류가 지속되면 1번 페이지 → 🔍 테이블 검증 탭에서 재검증하세요.")
        st.stop()
```

Page 3 파일에서 `except Exception` 블록을 모두 검색(`grep -n "except Exception" "hana_app/pages/3_🤖_모델_학습.py"`)하여 ETL 관련 블록에 위 패턴을 적용합니다.

- [ ] **Step 4: 앱 기동 확인 (수동)**

```bash
python3 -m streamlit run hana_app/main.py
```

3번 페이지 진입 시:
- HANA 미검증 상태 → 경고 + 1번 페이지 링크 표시 확인
- validated=True 상태 → 정상 렌더링 확인

- [ ] **Step 5: 커밋**

```bash
git add "hana_app/pages/3_🤖_모델_학습.py"
git commit -m "feat: page3 validated 가드 + 자동 재연결 + ETL 예외 st.exception() 표시"
```

---

### Task 9: 전체 테스트 스위트 통과 확인

**Files:** 없음 (검증만)

- [ ] **Step 1: `test_hana_app` 전체 실행**

```bash
python3 -m pytest tests/test_hana_app/ -v --tb=short
```

Expected: `26 passed` (Task 2: 10건, Task 3: 5건, Task 4: 11건)

- [ ] **Step 2: 전체 테스트 스위트 회귀 확인**

```bash
python3 -m pytest tests/ --tb=short -q 2>&1 | tail -10
```

Expected: N passed, 0 failed

- [ ] **Step 3: (선택) Windows 환경 확인**

Windows 환경에서:

```cmd
.venv_hana\Scripts\python -m pytest tests/test_hana_app/ -v --tb=short
```

Expected: `26 passed`

- [ ] **Step 4: 최종 커밋**

```bash
git commit --allow-empty -m "chore: HANA ETL 연결 안정화 + 테이블 검증 wizard 구현 완료"
```
