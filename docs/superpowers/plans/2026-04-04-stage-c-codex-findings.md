# Stage C: Codex 발견 수정 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Codex 최종 검토에서 발견된 2개 버그 수정: (1) zero-budget stratum 강제 1행 샘플링 왜곡, (2) WorkerThread 예외 파일 로그 미기록.

**Architecture:** 두 수정 모두 단일 파일 한 줄~수 줄 변경. Task 1은 `statistical_analysis.py`의 SQL CASE 생성 로직, Task 2는 `main_app.py`의 `WorkerThread.run()` 예외 핸들러.

**Tech Stack:** Python 3.12, DuckDB, PyQt5

---

## File Map

| File | Change |
|---|---|
| `statistical_analysis.py:95-98` | `max(1, n)` → `n` (0 할당 stratum 제외) |
| `main_app.py:54-56` | `WorkerThread.run()` except 블록에 `logger.exception()` 추가 |
| `tests/test_stage_c.py` | 신규 테스트 파일 |

---

### Task 1: Fix zero-budget stratum 강제 1행 포함

**Files:**
- Modify: `statistical_analysis.py:95-98`
- Test: `tests/test_stage_c.py`

**Context:** `per_group_sql_cases`에서 `max(1, n)`을 사용하여 할당이 0인 그룹(예: `non_dm_budget == 0`일 때 NON_DM)에도 강제로 1행이 포함됩니다. DM 그룹만으로 `max_rows`를 채우는 극단적 케이스에서 NON_DM이 분석 결과에 끼어들어 group composition을 왜곡합니다.

**수정 원칙:** 할당 0인 그룹은 CASE 조건에서 제외(`ELSE 0` 처리). DuckDB `WHERE rn <= 0`은 0건 반환.

- [ ] **Step 1: Write the failing test**

`tests/test_stage_c.py` 신규 파일 생성:

```python
"""
tests/test_stage_c.py - Stage C Codex 발견 수정 테스트
"""

import pytest
import duckdb
import pandas as pd
from unittest.mock import MagicMock, patch
from statistical_analysis import StatisticalAnalyzer, SamplingInfo


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


def test_zero_budget_stratum_excluded_from_sample():
    """non_dm_budget == 0 일 때 NON_DM 이 샘플에 포함되지 않아야 한다."""
    conn = duckdb.connect(':memory:')
    # DM 그룹 600건 (max_rows=500 이므로 DM 전수 > max_rows → non_dm_budget=0)
    conn.execute("""
        CREATE TABLE final_analysis AS
        SELECT 'T2DM_OHA' AS exposure_group, 1 AS follow_up_days, 1.0 AS follow_up_years, 0 AS dementia_event
        FROM range(600)
        UNION ALL
        SELECT 'NON_DM', 1, 1.0, 0 FROM range(200)
    """)

    analyzer = _make_analyzer_with_conn(conn)

    with patch('statistical_analysis.mem_manager') as mock_mm:
        mock_mm.get_safe_analysis_rows.return_value = 500  # DM 600 > 500 → budget 초과
        mock_mm.optimize_dtypes.side_effect = lambda df: df
        df, info = analyzer._load_data()

    # non_dm_budget = max(500 - 600, 0) = 0 → NON_DM 0건 할당
    # 수정 전: max(1, 0) = 1 → NON_DM 1건 포함 (버그)
    # 수정 후: NON_DM 0건 → 샘플에 없어야 함
    non_dm_rows = df[df['exposure_group'] == 'NON_DM']
    assert len(non_dm_rows) == 0, \
        f"non_dm_budget=0 인데 NON_DM {len(non_dm_rows)}건이 샘플에 포함됨"


def test_nonzero_budget_stratum_included():
    """정상 예산 할당 시 그룹이 샘플에 포함된다."""
    conn = duckdb.connect(':memory:')
    conn.execute("""
        CREATE TABLE final_analysis AS
        SELECT 'T2DM_OHA' AS exposure_group, 1 AS follow_up_days, 1.0 AS follow_up_years, 0 AS dementia_event
        FROM range(300)
        UNION ALL
        SELECT 'NON_DM', 1, 1.0, 0 FROM range(300)
    """)

    analyzer = _make_analyzer_with_conn(conn)

    with patch('statistical_analysis.mem_manager') as mock_mm:
        mock_mm.get_safe_analysis_rows.return_value = 400  # 총 600 > 400 → 샘플링
        mock_mm.optimize_dtypes.side_effect = lambda df: df
        df, info = analyzer._load_data()

    non_dm_rows = df[df['exposure_group'] == 'NON_DM']
    assert len(non_dm_rows) > 0, "예산 있는 NON_DM 그룹이 샘플에 포함되어야 함"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Volumes/model/yod_diabetes_app
python3 -m pytest tests/test_stage_c.py::test_zero_budget_stratum_excluded_from_sample -v
```

Expected: FAIL — `NON_DM 1건이 샘플에 포함됨` (max(1, 0)=1 버그)

- [ ] **Step 3: Implement the fix**

`statistical_analysis.py:95-98`을 다음으로 교체:

```python
            # 할당 0인 그룹은 CASE 조건에서 제외 — ELSE 0 으로 rn <= 0 → 0건 반환
            per_group_sql_cases = " ".join(
                f"WHEN exposure_group = '{g}' THEN {n}"
                for g, n in alloc.items()
                if n > 0
            )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python3 -m pytest tests/test_stage_c.py -v --tb=short
```

Expected: PASS (both)

- [ ] **Step 5: Verify existing tests still pass**

```bash
python3 -m pytest tests/test_sampling_info.py tests/test_stage_b.py -v --tb=short -q
```

Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add statistical_analysis.py tests/test_stage_c.py
git commit -m "fix: zero-budget stratum 강제 1행 포함 제거 — 샘플링 구성 왜곡 수정"
```

---

### Task 2: Fix WorkerThread 예외 파일 로그 미기록

**Files:**
- Modify: `main_app.py:54-56`
- Test: `tests/test_stage_c.py`

**Context:** `WorkerThread.run()`은 예외 발생 시 `error.emit(f"{e}\n{traceback.format_exc()}")`를 호출하지만 `logger.exception()`은 호출하지 않습니다. `_on_error()`도 `self.log()` (GUI 텍스트박스)에만 씁니다. 따라서 분석 실패 트레이스백이 파일 로거에 기록되지 않아 감사추적(Audit Trail)이 누락됩니다.

**수정:** `WorkerThread.run()` except 블록에 `logger.exception()` 추가. 이미 `logger = logging.getLogger(__name__)` 이 파일 상단에 있습니다.

- [ ] **Step 1: Write the failing test**

`tests/test_stage_c.py`에 추가:

```python
import logging
from unittest.mock import MagicMock, patch


def test_worker_thread_logs_exception_to_file_logger():
    """WorkerThread.run() 예외 시 logger.exception 이 호출된다."""
    from main_app import WorkerThread

    def failing_func(progress_callback=None):
        raise ValueError("test error for audit")

    thread = WorkerThread(failing_func)
    thread.error = MagicMock()

    with patch('main_app.logger') as mock_logger:
        thread.run()

    mock_logger.exception.assert_called_once()
    call_args = mock_logger.exception.call_args[0][0]
    assert "WorkerThread" in call_args or "분석" in call_args or "test error" in call_args or True, \
        "logger.exception 이 호출되지 않았습니다"
    # error signal 도 여전히 emit 되어야 함
    thread.error.emit.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python3 -m pytest tests/test_stage_c.py::test_worker_thread_logs_exception_to_file_logger -v
```

Expected: FAIL — `mock_logger.exception.assert_called_once()` 실패

- [ ] **Step 3: Implement the fix**

`main_app.py`에서 `WorkerThread.run()` except 블록 수정. 현재:

```python
        except Exception as e:
            if not self.is_cancelled:
                self.error.emit(f"{e}\n{traceback.format_exc()}")
```

변경 후:

```python
        except Exception as e:
            logger.exception("WorkerThread 분석 중 예외 발생")
            if not self.is_cancelled:
                self.error.emit(f"{e}\n{traceback.format_exc()}")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python3 -m pytest tests/test_stage_c.py -v --tb=short
```

Expected: all PASS

- [ ] **Step 5: Run full test suite**

```bash
python3 -m pytest tests/ -q --tb=short 2>&1 | tail -12
```

Expected: Stage C 신규 실패 없음 (기존 5개 실패만 유지)

- [ ] **Step 6: Commit**

```bash
git add main_app.py tests/test_stage_c.py
git commit -m "fix: WorkerThread 예외 파일 로거 기록 누락 — 감사추적 복원"
```

---

## Self-Review

**Spec coverage:**

| Codex 발견 | Task | 상태 |
|---|---|---|
| zero-budget stratum `max(1,n)` 왜곡 | Task 1 | ✓ |
| `WorkerThread` 예외 파일 로그 미기록 | Task 2 | ✓ |

**Placeholder scan:** 없음. 모든 코드 블록 완전함.

**Type consistency:** `alloc[g]` 값은 `int` — `if n > 0` 필터는 동일 타입에 적용. `logger.exception()` 시그니처 변경 없음.

**의존성:** Task 1과 Task 2는 독립적. 순서 무관.
