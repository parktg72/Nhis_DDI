# Stage D: 테스트 정밀도 향상 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stage C 리뷰(Codex + Gemini)에서 지적된 테스트 정밀도 결함 3건 수정 — 과다 샘플링 미검증, 로거 메시지 미고정, dm_total=0 방어 로직 미검증.

**Architecture:** `tests/test_stage_c.py` 기존 테스트 강화 + 신규 테스트 1건. 프로덕션 코드 변경 없음.

**Tech Stack:** Python 3.12, DuckDB, pytest, unittest.mock

---

## File Map

| File | Change |
|---|---|
| `tests/test_stage_c.py:60-79` | `test_nonzero_budget_stratum_included` — `> 0` → `== 100` 정확한 건수 assert |
| `tests/test_stage_c.py:95` | `assert_called_once()` → `assert_called_once_with("WorkerThread 분석 중 예외 발생")` |
| `tests/test_stage_c.py` | 신규: `test_dm_total_zero_raises_error` (dm_total=0 방어 로직) |

---

### Task 1: test_nonzero_budget_stratum_included 정밀화

**Files:**
- Modify: `tests/test_stage_c.py:60-79`

**Context:** 현재 `len(non_dm_rows) > 0` 만 확인해 "과다 샘플링이 없는지"를 보장하지 않는다. 데이터셋(DM 300, NON_DM 300, limit 400) 기준으로 `non_dm_budget = 400 - 300 = 100` 이므로 NON_DM은 정확히 100건이어야 한다.

- [ ] **Step 1: Run existing test to confirm it passes**

```bash
cd /Volumes/model/yod_diabetes_app
python3 -m pytest tests/test_stage_c.py::test_nonzero_budget_stratum_included -v
```

Expected: PASS (현재 `> 0` 조건 통과)

- [ ] **Step 2: Strengthen the assertion**

`tests/test_stage_c.py:78-79` 를 다음으로 교체:

```python
    non_dm_rows = df[df['exposure_group'] == 'NON_DM']
    # non_dm_budget = max_rows(400) - dm_total(300) = 100
    assert len(non_dm_rows) == 100, \
        f"NON_DM 샘플 건수가 예산(100)과 다름: {len(non_dm_rows)}건"
    dm_rows = df[df['exposure_group'] == 'T2DM_OHA']
    assert len(dm_rows) == 300, \
        f"DM 그룹 전수 포함되어야 하나 {len(dm_rows)}건"
```

- [ ] **Step 3: Run test to verify it passes**

```bash
python3 -m pytest tests/test_stage_c.py::test_nonzero_budget_stratum_included -v
```

Expected: PASS — 정확한 건수 검증 통과

- [ ] **Step 4: Commit**

```bash
git add tests/test_stage_c.py
git commit -m "test: 샘플링 예산 정확한 건수 assert 강화 (non_dm == 100, dm == 300)"
```

---

### Task 2: WorkerThread 로거 메시지 고정

**Files:**
- Modify: `tests/test_stage_c.py:95`

**Context:** `assert_called_once()` 는 호출 유무만 확인. 메시지까지 고정하면 감사 로그 문구 변경도 즉시 감지.

- [ ] **Step 1: Run existing test to confirm it passes**

```bash
python3 -m pytest tests/test_stage_c.py::test_worker_thread_logs_exception_to_file_logger -v
```

Expected: PASS

- [ ] **Step 2: Strengthen the assertion**

`tests/test_stage_c.py:95` 를 다음으로 교체:

```python
    mock_logger.exception.assert_called_once_with("WorkerThread 분석 중 예외 발생")
```

- [ ] **Step 3: Run test to verify it passes**

```bash
python3 -m pytest tests/test_stage_c.py::test_worker_thread_logs_exception_to_file_logger -v
```

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_stage_c.py
git commit -m "test: WorkerThread 로거 메시지 고정 assert 강화"
```

---

### Task 3: dm_total=0 방어 로직 테스트 신규 추가

**Files:**
- Modify: `tests/test_stage_c.py` (신규 테스트 추가)

**Context:** `statistical_analysis.py:84` 에 `dm_total == 0` 일 때 예외를 발생시키는 방어 로직이 있다. 해당 로직이 실제로 동작하는지 검증하는 테스트가 없다. 추가할 예외 타입은 `pd.errors.EmptyDataError`.

- [ ] **Step 1: Locate the guard clause**

```bash
grep -n "dm_total" /Volumes/model/yod_diabetes_app/statistical_analysis.py
```

Expected: `dm_total == 0` 조건 및 예외 발생 라인 확인

- [ ] **Step 2: Write the failing test (add to test_stage_c.py)**

`tests/test_stage_c.py` 파일 끝에 추가:

```python

def test_dm_total_zero_raises_empty_data_error():
    """DM 그룹이 전혀 없으면 EmptyDataError 가 발생해야 한다."""
    conn = duckdb.connect(':memory:')
    conn.execute("""
        CREATE TABLE final_analysis AS
        SELECT 'NON_DM' AS exposure_group, 1 AS follow_up_days, 1.0 AS follow_up_years, 0 AS dementia_event
        FROM range(100)
    """)

    analyzer = _make_analyzer_with_conn(conn)

    with patch('statistical_analysis.mem_manager') as mock_mm:
        mock_mm.get_safe_analysis_rows.return_value = 50
        mock_mm.optimize_dtypes.side_effect = lambda df: df
        with pytest.raises(pd.errors.EmptyDataError):
            analyzer._load_data()
```

- [ ] **Step 3: Run test to verify it passes**

```bash
python3 -m pytest tests/test_stage_c.py::test_dm_total_zero_raises_empty_data_error -v
```

Expected: PASS

- [ ] **Step 4: Run full Stage C test suite**

```bash
python3 -m pytest tests/test_stage_c.py -v --tb=short
```

Expected: 4 tests PASS

- [ ] **Step 5: Run full test suite**

```bash
python3 -m pytest tests/ -q --tb=short 2>&1 | tail -8
```

Expected: 기존 5건 실패 유지, 신규 실패 없음

- [ ] **Step 6: Commit**

```bash
git add tests/test_stage_c.py
git commit -m "test: dm_total=0 방어 로직 EmptyDataError 테스트 추가"
```

---

## Self-Review

**Spec coverage:**

| Codex/Gemini 지적 | Task | 상태 |
|---|---|---|
| `NON_DM > 0` → `== 100` 정확한 건수 assert | Task 1 | ✓ |
| `assert_called_once()` → 메시지 고정 | Task 2 | ✓ |
| dm_total=0 방어 로직 테스트 | Task 3 | ✓ |

**Placeholder scan:** 없음. 모든 코드 블록 완전함.

**Type consistency:** `pd.errors.EmptyDataError` — `statistical_analysis.py` 의 실제 예외 타입 확인 필요 (Task 3 Step 1).

**의존성:** Task 1, 2, 3 독립적. 순서 무관.
