# Stage F: 최소 유효 행 수 하한 검증 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Gemini Stage E 리뷰 제안 — 유효 행이 1~수십 건일 때 `EmptyDataError` 는 피하지만 Cox 분석에서 `LinAlgError`/`ConvergenceError` 가 발생하는 문제 방지. 통계적으로 의미 있는 최소 행 수 하한을 설정하고 조기 실패시킨다.

**Architecture:**
- `_load_data()` 에서 유효 행 수가 최소 기준 미달 시 새 예외 `InsufficientDataError` 발생
- 최소 기준: `config.py`의 `STUDY_SETTINGS`에 `MIN_VALID_ROWS` 설정값 추가 (기본 30 — Cox 회귀의 실용적 최솟값, EPV ≥ 10 기준)
- 커스텀 예외 `InsufficientDataError`를 `utils.py`에 정의 (도메인 특화)
- 기존 `EmptyDataError` (0건) 와 `InsufficientDataError` (1~29건) 계층 분리

**Tech Stack:** Python 3.12, DuckDB, pytest

---

## File Map

| File | Change |
|---|---|
| `utils.py` | `InsufficientDataError` 커스텀 예외 추가 |
| `config.py` | `STUDY_SETTINGS['MIN_VALID_ROWS'] = 30` 추가 |
| `statistical_analysis.py:_load_data()` | 유효 행 수 하한 체크 추가 (양 경로) |
| `tests/test_stage_f.py` | 신규 테스트 파일 |

---

### Task 1: InsufficientDataError 커스텀 예외 및 설정값 추가

**Files:**
- Modify: `utils.py`
- Modify: `config.py`

- [ ] **Step 1: utils.py 에 커스텀 예외 추가**

`utils.py`에서 기존 예외 클래스 정의 위치 확인:

```bash
grep -n "class.*Error\|class.*Exception" /Volumes/model/yod_diabetes_app/utils.py
```

기존 예외 클래스 뒤에 추가:

```python
class InsufficientDataError(ValueError):
    """분석에 필요한 최소 유효 행 수를 충족하지 못할 때 발생.

    Cox 회귀에서 EPV(Events Per Variable) ≥ 10 을 만족하려면
    최소 수십 건의 유효 행이 필요하다.
    """
    def __init__(self, valid_rows: int, min_rows: int):
        super().__init__(
            f"유효 행 수({valid_rows:,}건)가 최소 분석 기준({min_rows:,}건)에 미달합니다. "
            "코호트 크기를 확인하거나 MIN_VALID_ROWS 설정을 낮추세요."
        )
        self.valid_rows = valid_rows
        self.min_rows = min_rows
```

- [ ] **Step 2: config.py 에 MIN_VALID_ROWS 추가**

`config.py`의 `STUDY_SETTINGS` 딕셔너리에 추가:

```bash
grep -n "SAMPLING_SEED\|STUDY_SETTINGS" /Volumes/model/yod_diabetes_app/config.py | head -10
```

`SAMPLING_SEED` 아래에 추가:

```python
    'MIN_VALID_ROWS': 30,         # Cox 분석 최소 유효 행 수 (EPV ≥ 10 기준)
```

- [ ] **Step 3: Write failing tests**

`tests/test_stage_f.py` 신규 파일 생성:

```python
"""
tests/test_stage_f.py - Stage F 최소 유효 행 수 하한 검증 테스트
"""

import pytest
import duckdb
import pandas as pd
from unittest.mock import patch
from statistical_analysis import StatisticalAnalyzer, SamplingInfo
from utils import InsufficientDataError

# 테스트 데이터 상수
_MIN_VALID_ROWS = 30   # config.py STUDY_SETTINGS['MIN_VALID_ROWS'] 와 동일


def _make_analyzer_with_conn(conn):
    """테스트용 StatisticalAnalyzer — 실제 DuckDB 연결 사용."""
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


def test_nonsampling_path_below_min_rows_raises_insufficient_data_error():
    """비샘플링 경로에서 유효 행이 MIN_VALID_ROWS 미만이면 InsufficientDataError 발생."""
    conn = duckdb.connect(':memory:')
    # 유효 행 10건 < MIN_VALID_ROWS(30) → InsufficientDataError
    conn.execute("""
        CREATE TABLE final_analysis AS
        SELECT 'T2DM_OHA' AS exposure_group, 1 AS follow_up_days, 1.0 AS follow_up_years, 0 AS dementia_event
        FROM range(10)
    """)

    analyzer = _make_analyzer_with_conn(conn)

    with patch('statistical_analysis.mem_manager') as mock_mm:
        mock_mm.get_safe_analysis_rows.return_value = 200  # total(10) <= 200 → 비샘플링
        mock_mm.optimize_dtypes.side_effect = lambda df: df
        with pytest.raises(InsufficientDataError) as exc_info:
            analyzer._load_data()

    assert exc_info.value.valid_rows == 10
    assert exc_info.value.min_rows == _MIN_VALID_ROWS


def test_nonsampling_path_exactly_min_rows_succeeds():
    """비샘플링 경로에서 유효 행이 정확히 MIN_VALID_ROWS 이면 성공해야 한다."""
    conn = duckdb.connect(':memory:')
    conn.execute(f"""
        CREATE TABLE final_analysis AS
        SELECT 'T2DM_OHA' AS exposure_group, 1 AS follow_up_days, 1.0 AS follow_up_years, 0 AS dementia_event
        FROM range({_MIN_VALID_ROWS})
    """)

    analyzer = _make_analyzer_with_conn(conn)

    with patch('statistical_analysis.mem_manager') as mock_mm:
        mock_mm.get_safe_analysis_rows.return_value = 200
        mock_mm.optimize_dtypes.side_effect = lambda df: df
        df, info = analyzer._load_data()  # 예외 없이 성공

    assert len(df) == _MIN_VALID_ROWS


def test_sampling_path_below_min_rows_raises_insufficient_data_error():
    """샘플링 경로에서 유효 행 합계가 MIN_VALID_ROWS 미만이면 InsufficientDataError 발생."""
    conn = duckdb.connect(':memory:')
    # total(500) > max_rows(50) → 샘플링 분기, 하지만 유효 행 10건 < 30
    conn.execute("""
        CREATE TABLE final_analysis AS
        SELECT 'T2DM_OHA' AS exposure_group, 1 AS follow_up_days, 1.0 AS follow_up_years, 0 AS dementia_event
        FROM range(10)
        UNION ALL
        SELECT 'T2DM_OHA', 0, 0.0, 0 FROM range(490)
    """)

    analyzer = _make_analyzer_with_conn(conn)

    with patch('statistical_analysis.mem_manager') as mock_mm:
        mock_mm.get_safe_analysis_rows.return_value = 50  # total(500) > 50 → 샘플링
        mock_mm.optimize_dtypes.side_effect = lambda df: df
        with pytest.raises(InsufficientDataError):
            analyzer._load_data()
```

- [ ] **Step 4: Run tests to verify they fail**

```bash
cd /Volumes/model/yod_diabetes_app
python3 -m pytest tests/test_stage_f.py -v --tb=short
```

Expected: FAIL — `InsufficientDataError` 미정의

- [ ] **Step 5: Commit**

```bash
git add utils.py config.py tests/test_stage_f.py
git commit -m "test: Stage F 최소 유효 행 수 하한 실패 테스트 + 설정값/예외 추가"
```

---

### Task 2: _load_data() 에 최소 행 수 하한 체크 추가

**Files:**
- Modify: `statistical_analysis.py`

**Context:** `_load_data()` 양 경로에서 `EmptyDataError` 체크 직후 `InsufficientDataError` 체크 추가. `valid_total` (샘플링 경로) 또는 `len(self._cached_df)` (비샘플링 경로) 를 `MIN_VALID_ROWS` 와 비교.

- [ ] **Step 1: statistical_analysis.py import 확인**

```bash
grep -n "^from utils\|^import utils\|InsufficientDataError" /Volumes/model/yod_diabetes_app/statistical_analysis.py
```

Expected: `from utils import ...` 라인 확인 → `InsufficientDataError` 추가 필요 여부 판단

- [ ] **Step 2: import 에 InsufficientDataError 추가**

`statistical_analysis.py` 상단 utils import 라인에 `InsufficientDataError` 추가:

```python
from utils import setup_logging, format_error_for_user, InsufficientDataError
```

(실제 import 라인에 맞게 조정)

- [ ] **Step 3: 샘플링 경로에 하한 체크 추가**

`valid_total == 0` 가드 직후에 추가:

```python
            min_valid = int(STUDY_SETTINGS.get('MIN_VALID_ROWS', 30))
            if valid_total == 0:
                logger.error("샘플링 분기: total=%d, valid_total=0 — EmptyDataError", total)
                raise pd.errors.EmptyDataError(self._MSG_NO_VALID_ROWS)
            if valid_total < min_valid:
                logger.error("샘플링 분기: valid_total=%d < min_valid=%d — InsufficientDataError",
                             valid_total, min_valid)
                raise InsufficientDataError(valid_rows=valid_total, min_rows=min_valid)
```

- [ ] **Step 4: 비샘플링 경로에 하한 체크 추가**

`self._cached_df.empty` 가드 직후에 추가:

```python
            if self._cached_df.empty:
                logger.error("비샘플링 분기: total=%d, valid_rows=0 — EmptyDataError", total)
                raise pd.errors.EmptyDataError(self._MSG_NO_VALID_ROWS)
            min_valid = int(STUDY_SETTINGS.get('MIN_VALID_ROWS', 30))
            valid_rows = len(self._cached_df)
            if valid_rows < min_valid:
                logger.error("비샘플링 분기: valid_rows=%d < min_valid=%d — InsufficientDataError",
                             valid_rows, min_valid)
                raise InsufficientDataError(valid_rows=valid_rows, min_rows=min_valid)
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
python3 -m pytest tests/test_stage_f.py -v --tb=short
```

Expected: 3 tests PASS

- [ ] **Step 6: Run full test suite**

```bash
python3 -m pytest tests/ -q --tb=short 2>&1 | tail -8
```

Expected: 기존 5건 실패 유지, 신규 실패 없음

- [ ] **Step 7: Commit**

```bash
git add statistical_analysis.py
git commit -m "feat: _load_data() 최소 유효 행 수 하한 체크 추가 (MIN_VALID_ROWS=30)"
```

---

## Self-Review

**Spec coverage:**

| Gemini 제안 | Task | 상태 |
|---|---|---|
| 최소 유효 행 수 하한 검증 | Task 1 + Task 2 | ✓ |
| 도메인 특화 커스텀 예외 | Task 1 (`InsufficientDataError`) | ✓ |

**Placeholder scan:** 없음. 모든 코드 블록 완전함.

**Type consistency:** `InsufficientDataError.__init__` 의 `valid_rows`, `min_rows` 속성이 테스트 `exc_info.value.valid_rows` 와 일치.

**의존성:** Task 1 완료 후 Task 2 진행 (import 필요).

**도메인 주의:** `MIN_VALID_ROWS = 30` 은 Cox 회귀 EPV ≥ 10 기준의 보수적 추정값. 실제 연구 프로토콜에 따라 조정 가능하며, config.py 에서 변경할 수 있도록 설계됨.
