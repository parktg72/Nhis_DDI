# HANA ETL 연결 안정화 + 테이블 검증 Wizard 설계 스펙

**날짜:** 2026-04-06
**목표:** 폐쇄망 SAP HANA DB에서 ETL → 학습 전체 흐름을 안정적으로 실행
**범위:** `hana_app/core/db.py`, `hana_app/core/config.py`, Page 1, Page 3
**범위 외:** `hana_etl.py`, `ml_runner.py` (config를 소비만 하므로 수정 없음)

---

## 1. 문제 정의

### 1.1 증상
Streamlit 3번 페이지에서 "데이터 추출 시작" 버튼 클릭 후 스피너가 돌다가 오류 발생.
오류 내용이 표시되지 않아 원인 파악 불가.

### 1.2 근본 원인 (3가지)

| # | 문제 | 위치 | 심각도 |
|---|------|------|--------|
| 1 | 모듈 레벨 싱글톤 — 멀티 사용자 세션 간 연결 공유 | `db.py:236` | HIGH |
| 2 | 테이블/컬럼 검증 단계 없음 — 기본값이 실제 DB와 불일치 시 SQL 오류 | `hana_etl.py` | HIGH |
| 3 | ETL 예외가 상세 표시 없이 묻힘 — 오류 원인 구별 불가 | `page 3` | MEDIUM |

---

## 2. 아키텍처

```
┌─────────────────────────────────────────────────────┐
│  Page 1 (연결 및 테이블설정)                          │
│  ① connect() → st.session_state["hana_conn"]        │
│  ② 스키마/테이블 탐색 wizard (4단계)                  │
│  ③ 컬럼 매핑 검증 → hana_config.json {validated:true}│
└──────────────────────┬──────────────────────────────┘
                       │ config + session_state 공유
┌──────────────────────▼──────────────────────────────┐
│  db.py — session_state 기반 격리                     │
│  • session_state["hana_conn"] 개별 HANAConnection    │
│  • ensure_connected(creds) 자동 재연결               │
│  • _fallback_conn: 테스트/CLI 환경 하위 호환          │
└──────────────────────┬──────────────────────────────┘
                       │ get_connection(st.session_state)
┌──────────────────────▼──────────────────────────────┐
│  Page 3 (모델 학습)                                   │
│  • validated 플래그 미확인 시 → Page 1 안내           │
│  • ensure_connected() 자동 재연결                    │
│  • ETL 예외 → st.exception() 전체 스택트레이스 표시   │
└─────────────────────────────────────────────────────┘
```

---

## 3. 컴포넌트 상세 설계

### 3.1 db.py — SessionConnectionManager

#### 변경 1: get_connection() 시그니처

```python
# 변경 전
_global_conn = HANAConnection()

def get_connection() -> HANAConnection:
    return _global_conn

# 변경 후
_fallback_conn = HANAConnection()   # 테스트/CLI 환경용 (이름만 변경)

def get_connection(session_state: dict | None = None) -> HANAConnection:
    """세션별 격리된 HANAConnection 반환.

    session_state가 None이면 _fallback_conn 반환 (테스트/비Streamlit 환경 하위 호환).
    Streamlit 환경에서는 반드시 st.session_state를 전달해야 한다.
    """
    if session_state is None:
        return _fallback_conn
    if "hana_conn" not in session_state:
        session_state["hana_conn"] = HANAConnection()
    return session_state["hana_conn"]
```

#### 변경 2: HANAConnection.ensure_connected()

```python
def ensure_connected(self, creds: dict) -> None:
    """연결이 끊겼으면 creds로 자동 재연결.

    creds 구조:
        {"host": str, "port": int, "user": str, "password": str}

    이미 연결된 상태면 아무것도 하지 않는다.
    재연결 실패 시 hdbcli 예외를 그대로 전파한다.
    """
    if not self.is_connected():
        self.connect(
            host=creds["host"],
            port=int(creds["port"]),
            user=creds["user"],
            password=creds["password"],
        )
```

#### creds 저장 위치
Page 1에서 연결 성공 시 `st.session_state["hana_creds"]`에 저장.
비밀번호는 session_state에만 보관 (JSON에 기록하지 않음 — 기존 Keychain 정책 유지).

#### 하위 호환
- `get_connection()` 인자 없이 호출 → `_fallback_conn` 반환
- 기존 테스트에서 `get_connection()` 호출 패턴 수정 불필요

---

### 3.2 config.py — validated 플래그

`DEFAULT_CONFIG`에 `"validated": False` 추가:

```python
DEFAULT_CONFIG: dict[str, Any] = {
    ...
    "validated": False,   # Page 1 테이블 검증 완료 시 True로 저장
    ...
}
```

Page 1 검증 완료 시 `cfg["validated"] = True` 후 `save_config(cfg)` 호출.
테이블 설정 변경 시 `cfg["validated"] = False`로 초기화.

---

### 3.3 Page 1 — 테이블 검증 Wizard

기존 연결 섹션 아래에 **"🔍 테이블 검증"** 섹션 추가.
연결된 상태에서만 활성화. 4단계 순차 진행:

#### Page 1 상단 연결 코드 교체 (필수)
현재 Page 1에도 `conn = get_connection()` (인자 없음) 호출이 있으므로 교체 필요:
```python
# 변경 전 (page 1 line 29)
conn = get_connection()
# 변경 후
conn = get_connection(st.session_state)
```
연결 성공 시 session_state에 creds 저장:
```python
# connect() 성공 직후
st.session_state["hana_creds"] = {
    "host": host, "port": port, "user": user, "password": password
}
```

#### Step 1: 스키마 선택
- `conn.get_schemas()` 호출 → 사용 가능한 스키마 목록 표시
- t20 / t30 / t40 / t60 / yoyang 각각 selectbox로 스키마 선택
- 현재 config 값을 기본 선택으로 pre-fill

#### Step 2: 테이블 선택
- 선택된 스키마별 `conn.get_tables(schema)` 호출
- 각 논리 테이블에 실제 테이블명 selectbox 매핑
- 현재 config 값을 기본 선택으로 pre-fill

#### Step 3: 컬럼 매핑 검증
- 선택된 각 테이블의 `conn.get_columns(schema, table)` 호출
- 코드가 기대하는 논리 컬럼명 ↔ 실제 DB 컬럼명 비교 표 표시:

| 논리명 | 기대 컬럼 | 실제 존재 | 상태 |
|--------|-----------|-----------|------|
| patient_id | INDI_DSCM_NO | INDI_DSCM_NO | ✅ |
| bill_no | CMN_KEY | KEY_NO | 🔴 → selectbox |

- 불일치 컬럼은 실제 컬럼 중 selectbox로 대체 선택
- 모든 선택값에 `_assert_safe_identifier()` 서버 사이드 재검증

#### Step 4: 저장 및 완료
- "✅ 검증 완료 & 저장" 버튼
- config["tables"] + config["columns"] 업데이트
- config["validated"] = True
- `save_config(cfg)` 호출
- `st.success("✅ 검증 완료 — 3번 페이지에서 학습을 시작할 수 있습니다.")`

#### 검증 오류 메시지 세분화

| 체크 | 실패 메시지 |
|------|------------|
| 스키마 존재 | `❌ {schema} 스키마 없음 — 권한 또는 이름 확인` |
| 테이블 존재 | `❌ {table} 테이블 없음 — 실제 테이블을 선택하세요` |
| 필수 컬럼 존재 | `❌ {col} 컬럼 없음 — 대체 컬럼을 선택하세요` |
| 식별자 안전성 | `❌ 허용되지 않는 문자 포함 (영문자·숫자·_·$·# 만 허용)` |

---

### 3.4 Page 3 — 가드 + 오류 표시

#### 페이지 상단 전처리 (기존 `conn = get_connection()` 교체)

```python
# 1. session_state 기반 연결
conn = get_connection(st.session_state)

# 2. validated 플래그 확인
cfg = load_config()
if not cfg.get("validated"):
    st.warning("⚠️ 테이블 검증이 완료되지 않았습니다.")
    st.page_link(
        "pages/1_🔌_연결_및_테이블설정.py",
        label="👉 1번 페이지에서 테이블 검증 후 돌아오세요",
    )
    st.stop()

# 3. 자동 재연결
creds = st.session_state.get("hana_creds")
if creds:
    try:
        conn.ensure_connected(creds)
    except Exception as e:
        st.error(f"❌ DB 재연결 실패: {e}")
        st.stop()
elif not conn.is_connected():
    st.error("❌ DB 연결이 없습니다. 1번 페이지에서 먼저 연결하세요.")
    st.stop()
```

#### ETL 예외 처리

```python
try:
    with st.spinner("데이터 추출 중..."):
        records = extractor.extract_prescriptions(...)
except Exception as e:
    st.error("❌ 데이터 추출 실패")
    st.exception(e)
    st.info("💡 오류가 지속되면 1번 페이지에서 테이블 검증을 다시 실행하세요.")
    st.stop()
```

---

## 4. 데이터 흐름

```
[Page 1]
  connect() 성공
    → st.session_state["hana_conn"] = HANAConnection (연결됨)
    → st.session_state["hana_creds"] = {host, port, user, password}
  wizard 완료
    → hana_config.json {validated: true, tables: {...}, columns: {...}}

[Page 3]
  get_connection(st.session_state)
    → session_state["hana_conn"] 반환
  ensure_connected(session_state["hana_creds"])
    → 끊겼으면 재연결
  load_config() → validated: true 확인
  HANAExtractor(conn, cfg["tables"], cfg["columns"])
    → extract_prescriptions()
```

---

## 5. 오류 처리 정책

| 오류 상황 | 처리 방법 |
|-----------|-----------|
| 연결 끊김 (페이지 전환 후) | `ensure_connected()` 자동 재연결 |
| 재연결 실패 | `st.error` + `st.stop()` |
| validated=False | `st.warning` + Page 1 링크 + `st.stop()` |
| SQL 오류 (테이블/컬럼 불일치) | `st.exception()` 전체 스택트레이스 |
| 식별자 검증 실패 | wizard Step 3에서 즉시 인라인 오류 |

---

## 6. 테스트 전략

### 6.1 db.py 단위 테스트

```python
# session_state 격리 검증
def test_get_connection_creates_per_session():
    s1, s2 = {}, {}
    c1 = get_connection(s1)
    c2 = get_connection(s2)
    assert c1 is not c2          # 세션 격리
    assert get_connection(s1) is c1  # 같은 세션은 동일 객체

def test_get_connection_none_returns_fallback():
    assert get_connection(None) is get_connection(None)  # 동일 _fallback_conn

def test_ensure_connected_reconnects_on_disconnect():
    conn = HANAConnection()
    creds = {"host": "h", "port": 30015, "user": "u", "password": "p"}
    with patch.object(conn, "is_connected", return_value=False):
        with patch.object(conn, "connect") as mock_connect:
            conn.ensure_connected(creds)
            mock_connect.assert_called_once_with(host="h", port=30015, user="u", password="p")

def test_ensure_connected_skips_if_already_connected():
    conn = HANAConnection()
    with patch.object(conn, "is_connected", return_value=True):
        with patch.object(conn, "connect") as mock_connect:
            conn.ensure_connected({"host": "h", "port": 30015, "user": "u", "password": "p"})
            mock_connect.assert_not_called()
```

### 6.2 Page 1 Wizard 헬퍼 단위 테스트

Page 1의 wizard 로직은 `hana_app/core/table_validator.py` 헬퍼 모듈로 추출해 테스트:

```python
# hana_app/core/table_validator.py (신규)
def check_column_mapping(actual_cols: list[str], expected_map: dict) -> dict:
    """논리명 → 실제 컬럼명 매핑 검증. 불일치 항목 반환."""

# 테스트
def test_check_column_mapping_detects_missing():
    """기대 컬럼이 실제 DB에 없으면 missing 목록에 포함."""
    actual = ["INDI_DSCM_NO", "CMN_KEY"]
    expected = {"patient_id": "INDI_DSCM_NO", "bill_no": "MISSING_COL"}
    result = check_column_mapping(actual, expected)
    assert "bill_no" in result["missing"]

def test_check_column_mapping_all_match():
    actual = ["INDI_DSCM_NO", "CMN_KEY"]
    expected = {"patient_id": "INDI_DSCM_NO", "bill_no": "CMN_KEY"}
    result = check_column_mapping(actual, expected)
    assert result["missing"] == []

def test_wizard_rejects_unsafe_identifier():
    """SQL 인젝션 패턴 컬럼명 → ValueError."""
    with pytest.raises(ValueError):
        _assert_safe_identifier("col'; DROP TABLE--", "column")
```

### 6.3 Page 3 가드 테스트

Page 3 가드 로직도 `hana_app/core/page_guards.py` 헬퍼로 추출해 테스트:

```python
def test_page3_guard_raises_when_not_validated():
    """validated=False → StopException (st.stop() 내부 예외) 발생."""

def test_page3_guard_passes_when_validated():
    """validated=True + 연결됨 → 정상 통과."""
```

---

## 7. 구현 순서

| Task | 파일 | 내용 |
|------|------|------|
| 1 | `hana_app/core/db.py` | `get_connection(session_state)` + `ensure_connected()` + 테스트 4건 |
| 2 | `hana_app/core/config.py` | `validated` 플래그 추가 + 테스트 |
| 3 | `hana_app/core/table_validator.py` (신규) | `check_column_mapping()` 헬퍼 + 테스트 |
| 4 | `hana_app/pages/1_🔌_연결_및_테이블설정.py` | 연결 코드 교체 + creds 저장 + wizard 4단계 |
| 5 | `hana_app/pages/3_🤖_모델_학습.py` | 가드 로직 교체 + ETL 예외 표시 |
| 6 | 통합 검증 | 전체 테스트 스위트 통과 확인 |

---

## 8. 제약 사항

- **Python 3.12 / Windows** 환경 타겟
- `hdbcli`는 폐쇄망 설치 전제 — import는 `connect()` 내부에서만 (`from hdbcli import dbapi`)
- 비밀번호는 `session_state`에만 보관, JSON 저장 금지 (기존 Keychain 정책 유지)
- `hana_etl.py` / `ml_runner.py` 수정 없음 — config를 소비만 하는 구조 유지
