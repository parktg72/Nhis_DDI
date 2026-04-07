# Stage G: 분석 견고성 강화 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stage F 리뷰(Codex + Gemini)에서 지적된 4개 잔존 리스크 제거:
1. `df_prepared` 직접 주입 시 `_load_data()` 가드 우회 가능
2. `dropna()` 후 행 수 미달로 lifelines 불명확 오류 가능
3. 이벤트 수(Events) 기반 EPV 체크 미반영
4. `format_error_for_user()` 에서 `InsufficientDataError` 전용 분기 없음

**Architecture:**
- Task 1: `format_error_for_user` — `InsufficientDataError` 전용 분기 추가 (단순 1줄)
- Task 2: `_prepare()` 이후 `dropna()` 실행 전 최소 행 수 재확인 헬퍼 `_check_min_rows()` 추가
- Task 3: `run_cox()` 내 `cph.fit()` 직전 이벤트 수 EPV 체크 추가
- Task 4: `MIN_VALID_ROWS` 설정값 양의 정수 검증 추가

**Tech Stack:** Python 3.12, DuckDB, pytest, lifelines

---

## File Map

| File | Change |
|---|---|
| `utils.py:137-140` | `InsufficientDataError` 전용 분기 `EmptyDataError` 앞에 추가 |
| `statistical_analysis.py` | `_check_min_rows()` 헬퍼 추가, `run_cox()` EPV 체크, config 검증 |
| `tests/test_stage_g.py` | 신규 테스트 파일 |

---

### Task 1: format_error_for_user InsufficientDataError 전용 분기

**Files:**
- Modify: `utils.py:137`

**Context:** 현재 `InsufficientDataError(ValueError)` 는 `isinstance(exc, ValueError)` 분기에서 `"입력값 오류: ..."` 로 처리됨. 사용자에게 "코호트 크기 확인" 이라는 구체적 안내가 없어 혼란 가능.

- [ ] **Step 1: Write failing test**

`tests/test_stage_g.py` 신규 파일 생성:

```python
"""
tests/test_stage_g.py - Stage G 분석 견고성 강화 테스트
"""

import pytest
from utils import format_error_for_user, InsufficientDataError


def test_format_error_for_user_insufficient_data_error():
    """InsufficientDataError 가 사용자 친화적 메시지로 변환되어야 한다."""
    exc = InsufficientDataError(valid_rows=10, min_rows=30)
    msg = format_error_for_user(exc)
    assert "10" in msg or "30" in msg or "최소" in msg, \
        f"InsufficientDataError 전용 메시지 없음: {msg!r}"
    # ValueError 일반 분기("입력값 오류:")로 떨어지면 안 됨
    assert "입력값 오류" not in msg, \
        f"InsufficientDataError 가 일반 ValueError 로 처리됨: {msg!r}"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Volumes/model/yod_diabetes_app
python3 -m pytest tests/test_stage_g.py::test_format_error_for_user_insufficient_data_error -v
```

Expected: FAIL — "입력값 오류" 로 처리됨

- [ ] **Step 3: Add InsufficientDataError branch in utils.py**

`utils.py:137` 의 `EmptyDataError` 체크 직전에 추가:

```python
    if isinstance(exc, InsufficientDataError):
        return (
            f"유효 데이터 부족: {exc.valid_rows:,}건 (최소 {exc.min_rows:,}건 필요). "
            "코호트 크기를 확인하거나 MIN_VALID_ROWS 설정을 조정하세요."
        )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python3 -m pytest tests/test_stage_g.py::test_format_error_for_user_insufficient_data_error -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add utils.py tests/test_stage_g.py
git commit -m "fix: format_error_for_user InsufficientDataError 전용 분기 추가"
```

---

### Task 2: _check_min_rows() 헬퍼 — dropna 후 행 수 재확인

**Files:**
- Modify: `statistical_analysis.py`

**Context:** `_load_data()` 에서 30건 통과 후 `_prepare()` → `dropna()` 로 30건 미만이 될 수 있음. `run_cox()`, `run_subgroup()` 등 분석 함수 내 `df_model = df_prepared[cols].dropna()` 직후에도 행 수 체크가 필요. 공통 헬퍼로 추출.

- [ ] **Step 1: Write failing test**

`tests/test_stage_g.py` 에 추가:

```python
import duckdb
import pandas as pd
from unittest.mock import patch
from statistical_analysis import StatisticalAnalyzer, SamplingInfo
from utils import InsufficientDataError


def _make_analyzer_with_conn(conn):
    class MockStorage:
        def get_row_count(self, t):
            return conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    class MockDM:
        storage = MockStorage()
        def query(self, sql):
            return conn.execute(sql).df()
        def execute(self, sql):
            conn.execute(sql)
    analyzer = StatisticalAnalyzer.__new__(StatisticalAnalyzer)
    analyzer.dm = MockDM()
    analyzer.results = {}
    analyzer._cached_df = None
    analyzer._sampling_info = SamplingInfo(applied=False, total_rows=0, sampled_rows=0)
    return analyzer


def test_check_min_rows_raises_on_small_df():
    """_check_min_rows() 가 기준 미달 DataFrame 에서 InsufficientDataError 를 발생시킨다."""
    analyzer = StatisticalAnalyzer.__new__(StatisticalAnalyzer)
    analyzer.results = {}
    small_df = pd.DataFrame({'a': range(5)})
    with pytest.raises(InsufficientDataError):
        analyzer._check_min_rows(small_df, context="테스트")


def test_check_min_rows_passes_on_sufficient_df():
    """_check_min_rows() 가 기준 이상 DataFrame 에서 예외 없이 반환한다."""
    analyzer = StatisticalAnalyzer.__new__(StatisticalAnalyzer)
    analyzer.results = {}
    ok_df = pd.DataFrame({'a': range(30)})
    analyzer._check_min_rows(ok_df, context="테스트")  # 예외 없음
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/test_stage_g.py::test_check_min_rows_raises_on_small_df tests/test_stage_g.py::test_check_min_rows_passes_on_sufficient_df -v
```

Expected: FAIL — `_check_min_rows` 미정의

- [ ] **Step 3: Add _check_min_rows() helper to StatisticalAnalyzer**

`statistical_analysis.py` 의 `_release_cache()` 메서드 직전에 추가:

```python
    def _check_min_rows(self, df: pd.DataFrame, context: str = "") -> None:
        """dropna 등 필터 후 행 수가 MIN_VALID_ROWS 미만이면 InsufficientDataError 발생.

        run_cox, run_subgroup 등 분석 함수에서 cph.fit() 직전에 호출.
        """
        min_valid = int(STUDY_SETTINGS.get('MIN_VALID_ROWS', 30))
        if len(df) < min_valid:
            logger.warning("%s: dropna 후 행 수 %d < min_valid %d — InsufficientDataError",
                           context, len(df), min_valid)
            raise InsufficientDataError(valid_rows=len(df), min_rows=min_valid)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python3 -m pytest tests/test_stage_g.py::test_check_min_rows_raises_on_small_df tests/test_stage_g.py::test_check_min_rows_passes_on_sufficient_df -v
```

Expected: PASS

- [ ] **Step 5: Apply _check_min_rows in run_cox() after dropna**

`run_cox()` 에서 `df_model = df_prepared[cols].dropna()` 직후:

```python
            df_model = df_prepared[cols].dropna()
            self._check_min_rows(df_model, context=f"run_cox {mname}")
```

- [ ] **Step 6: Run full test suite**

```bash
python3 -m pytest tests/ -q --tb=short 2>&1 | tail -8
```

Expected: 기존 5건 실패 유지, 신규 실패 없음

- [ ] **Step 7: Commit**

```bash
git add statistical_analysis.py tests/test_stage_g.py
git commit -m "feat: _check_min_rows() 헬퍼 추가 — dropna 후 행 수 재확인"
```

---

### Task 3: run_cox() EPV(이벤트 수) 체크

**Files:**
- Modify: `statistical_analysis.py`

**Context:** Gemini 제안 — 행이 30건이어도 치매 이벤트가 1건이면 Cox 모델 불안정. `MIN_EVENTS = 10` (EPV 기준)을 `config.py`에 추가하고 `cph.fit()` 직전에 체크.

- [ ] **Step 1: config.py에 MIN_EVENTS 추가**

`config.py` 의 `MIN_VALID_ROWS` 아래에 추가:

```python
    'MIN_EVENTS': 10,             # Cox 분석 최소 이벤트 수 (EPV heuristic)
```

- [ ] **Step 2: Write failing test**

`tests/test_stage_g.py` 에 추가:

```python
def test_run_cox_raises_on_insufficient_events():
    """run_cox() 에서 이벤트 수가 MIN_EVENTS 미만이면 InsufficientDataError 발생."""
    conn = duckdb.connect(':memory:')
    # 30건이지만 치매 이벤트 0건
    conn.execute("""
        CREATE TABLE final_analysis AS
        SELECT 'T2DM_OHA' AS exposure_group,
               365 AS follow_up_days,
               1.0 AS follow_up_years,
               0 AS dementia_event
        FROM range(30)
    """)
    analyzer = _make_analyzer_with_conn(conn)
    with patch('statistical_analysis.mem_manager') as mock_mm:
        mock_mm.get_safe_analysis_rows.return_value = 200
        mock_mm.optimize_dtypes.side_effect = lambda df: df
        with pytest.raises(InsufficientDataError):
            analyzer.run_cox()
```

- [ ] **Step 3: Run test to verify it fails**

```bash
python3 -m pytest tests/test_stage_g.py::test_run_cox_raises_on_insufficient_events -v
```

Expected: FAIL (현재 lifelines 오류 또는 예외 없음)

- [ ] **Step 4: Add EPV check in run_cox()**

`run_cox()` 에서 `df_prepared = self._prepare(raw)` 직후:

```python
        min_events = int(STUDY_SETTINGS.get('MIN_EVENTS', 10))
        event_count = int(df_prepared[outcome].sum())
        if event_count < min_events:
            logger.warning("run_cox: 이벤트 수 %d < min_events %d — InsufficientDataError",
                           event_count, min_events)
            raise InsufficientDataError(valid_rows=event_count, min_rows=min_events)
```

- [ ] **Step 5: Run test to verify it passes**

```bash
python3 -m pytest tests/test_stage_g.py::test_run_cox_raises_on_insufficient_events -v
```

Expected: PASS

- [ ] **Step 6: Run full test suite**

```bash
python3 -m pytest tests/ -q --tb=short 2>&1 | tail -8
```

Expected: 기존 5건 실패 유지, 신규 실패 없음

- [ ] **Step 7: Commit**

```bash
git add config.py statistical_analysis.py tests/test_stage_g.py
git commit -m "feat: run_cox() EPV 이벤트 수 하한 체크 추가 (MIN_EVENTS=10)"
```

---

### Task 4: MIN_VALID_ROWS 설정값 양의 정수 검증

**Files:**
- Modify: `statistical_analysis.py`

**Context:** Codex 제안 — `MIN_VALID_ROWS = 0` 또는 음수 입력 시 하한 체크가 무력화됨. `_load_data()` 상단 `min_valid` 읽기 직후 검증.

- [ ] **Step 1: Write failing test**

`tests/test_stage_g.py` 에 추가:

```python
def test_load_data_raises_on_invalid_min_valid_rows():
    """MIN_VALID_ROWS 가 0 이하이면 ValueError 가 발생해야 한다."""
    conn = duckdb.connect(':memory:')
    conn.execute("""
        CREATE TABLE final_analysis AS
        SELECT 'T2DM_OHA' AS exposure_group, 1 AS follow_up_days, 1.0 AS follow_up_years, 0 AS dementia_event
        FROM range(50)
    """)
    analyzer = _make_analyzer_with_conn(conn)
    with patch('statistical_analysis.mem_manager') as mock_mm:
        mock_mm.get_safe_analysis_rows.return_value = 200
        mock_mm.optimize_dtypes.side_effect = lambda df: df
        with patch('statistical_analysis.STUDY_SETTINGS', {'MIN_VALID_ROWS': 0, 'SAMPLING_SEED': 42}):
            with pytest.raises(ValueError, match="MIN_VALID_ROWS"):
                analyzer._load_data()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python3 -m pytest tests/test_stage_g.py::test_load_data_raises_on_invalid_min_valid_rows -v
```

Expected: FAIL

- [ ] **Step 3: Add validation in _load_data()**

`_load_data()` 에서 `min_valid` 읽기 직후:

```python
        min_valid = int(STUDY_SETTINGS.get('MIN_VALID_ROWS', 30))
        if min_valid <= 0:
            raise ValueError(f"MIN_VALID_ROWS 는 양의 정수여야 합니다: {min_valid}")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python3 -m pytest tests/test_stage_g.py -v --tb=short
```

Expected: all PASS

- [ ] **Step 5: Run full test suite**

```bash
python3 -m pytest tests/ -q --tb=short 2>&1 | tail -8
```

Expected: 기존 5건 실패 유지, 신규 실패 없음

- [ ] **Step 6: Commit**

```bash
git add statistical_analysis.py tests/test_stage_g.py
git commit -m "feat: MIN_VALID_ROWS 양의 정수 검증 추가"
```

---

## Self-Review

**Spec coverage:**

| 리뷰 지적 | Task | 상태 |
|---|---|---|
| `format_error_for_user` InsufficientDataError 전용 분기 | Task 1 | ✓ |
| `dropna()` 후 행 수 미달 가드 | Task 2 | ✓ |
| EPV 이벤트 수 체크 | Task 3 | ✓ |
| `MIN_VALID_ROWS` 설정값 양수 검증 | Task 4 | ✓ |

**Placeholder scan:** 없음. 모든 코드 블록 완전함.

**Type consistency:**
- `_check_min_rows(df, context)` — `df: pd.DataFrame`, `context: str`
- `InsufficientDataError(valid_rows=int, min_rows=int)` — Task 2, 3 모두 일치
- `MIN_EVENTS` 키 — Task 3 config.py 추가와 run_cox() 접근 일치

**의존성:**
- Task 1, 4 독립적
- Task 2 완료 후 Task 3 (`_check_min_rows` 활용 가능)
- 권장 순서: Task 1 → Task 4 → Task 2 → Task 3
