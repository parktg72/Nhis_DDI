# Stage E: 비샘플링 경로 유효 행 0건 동작 명시 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Codex Stage D 리뷰 최우선 제안 — `total <= max_rows`(비샘플링) 경로에서 유효 행(`follow_up_days > 0`)이 0건일 때 동작을 명시하고 테스트로 고정한다.

**Architecture:** 현재 `_load_data()`의 `valid_total == 0` 가드는 샘플링 분기(`total > max_rows`) 내부에만 존재한다. 비샘플링 경로에서 `follow_up_days > 0` 행이 0건이면 `EmptyDataError` 없이 빈 DataFrame이 반환된다. 이 동작이 의도라면 테스트로 명시하고, 버그라면 가드를 비샘플링 경로로도 확장한다.

**의사결정 기준:**
- 빈 DataFrame을 상위 분석 로직이 graceful하게 처리할 수 있으면 → 현재 동작(빈 DF 반환) 유지 + 테스트 명시
- 상위 로직이 빈 DataFrame에서 KeyError/ZeroDivisionError 등 예측 불가 오류를 낼 가능성이 있으면 → 가드 확장 후 테스트

**Tech Stack:** Python 3.12, DuckDB, pytest

---

## File Map

| File | Change |
|---|---|
| `statistical_analysis.py:60-130` | (조건부) 비샘플링 경로에 `valid_total == 0` 가드 추가 |
| `tests/test_stage_c.py` | 비샘플링 경로 유효 행 0건 동작 테스트 추가 |

---

### Task 1: 비샘플링 경로 동작 조사

**Files:**
- Read: `statistical_analysis.py:60-160` (비샘플링 경로 전체)

**Context:** 비샘플링 경로(`total <= max_rows`)에서 `follow_up_days > 0` 필터가 어디서 적용되는지, 빈 DataFrame이 반환될 때 상위 분석 로직(`run_analysis`, `calculate_*` 등)이 어떻게 반응하는지 확인한다.

- [ ] **Step 1: 비샘플링 경로 코드 확인**

```bash
cd /Volumes/model/yod_diabetes_app
grep -n "follow_up_days" statistical_analysis.py
```

Expected: 비샘플링 경로에서 `follow_up_days > 0` 필터가 있는지 확인

- [ ] **Step 2: 비샘플링 경로 실제 동작 확인**

```python
# 비샘플링 경로 흐름:
# total <= max_rows → self.dm.query("SELECT * FROM final_analysis WHERE ...") 직접 반환
# → follow_up_days > 0 필터가 적용되는지 확인
```

```bash
grep -n -A5 "total > max_rows" statistical_analysis.py
```

- [ ] **Step 3: 상위 로직에서 빈 DF 처리 확인**

```bash
grep -n "def run_analysis\|def calculate_\|def _calculate" statistical_analysis.py | head -20
```

Expected: 빈 DataFrame 입력 시 KeyError/ZeroDivisionError 발생 여부 판단

---

### Task 2: 비샘플링 경로 가드 추가 (조사 결과에 따라)

**Files:**
- Modify: `statistical_analysis.py` (비샘플링 경로에 가드 추가 — Task 1 결과에 따라 조건부)

**Context:** Task 1 조사 결과 빈 DataFrame이 상위 로직에서 오류를 유발한다고 판단되면 가드를 추가한다. 그렇지 않으면 이 Task는 건너뛰고 Task 3(테스트 명시)만 수행한다.

**가드 추가 위치:** `statistical_analysis.py`에서 `total <= max_rows` 분기, `self.dm.query(...)` 직전

**추가할 코드 (조건부):**

```python
        else:
            # 비샘플링 경로: 전체 데이터 직접 로드
            # follow_up_days > 0 유효 행 검증 — 빈 DataFrame 하류 오류 방지
            valid_count = self.dm.storage.get_row_count('final_analysis')  # 또는 별도 쿼리
            # (실제 컬럼/쿼리는 Task 1 조사 결과로 확정)
```

- [ ] **Step 1: Write the failing test first**

`tests/test_stage_c.py` 끝에 추가:

```python

def test_nonsampling_path_no_valid_rows_raises_error():
    """비샘플링 경로(total <= max_rows)에서 유효 행이 없을 때 EmptyDataError 발생."""
    conn = duckdb.connect(':memory:')
    # total(50) <= max_rows(200) → 비샘플링 경로
    # follow_up_days=0 → 유효 행 0건
    conn.execute("""
        CREATE TABLE final_analysis AS
        SELECT 'T2DM_OHA' AS exposure_group, 0 AS follow_up_days, 0.0 AS follow_up_years, 0 AS dementia_event
        FROM range(50)
    """)

    analyzer = _make_analyzer_with_conn(conn)

    with patch('statistical_analysis.mem_manager') as mock_mm:
        mock_mm.get_safe_analysis_rows.return_value = 200  # total(50) <= 200 → 비샘플링
        mock_mm.optimize_dtypes.side_effect = lambda df: df
        with pytest.raises(pd.errors.EmptyDataError):
            analyzer._load_data()
```

- [ ] **Step 2: Run test to verify it fails (현재 동작 확인)**

```bash
python3 -m pytest tests/test_stage_c.py::test_nonsampling_path_no_valid_rows_raises_error -v
```

Expected: FAIL (현재 가드 없음) 또는 PASS (이미 필터 적용)

- [ ] **Step 3: 조사 결과에 따른 처리**

  **Case A — 테스트 FAIL (가드 없음, 빈 DF가 문제):**
  `statistical_analysis.py` 비샘플링 분기에 가드 추가:
  ```python
          else:
              self._cached_df = self.dm.query(
                  "SELECT * FROM final_analysis WHERE follow_up_days > 0"
              )
              if len(self._cached_df) == 0:
                  raise pd.errors.EmptyDataError(
                      "추적 가능한 행(follow_up_days > 0)이 없습니다. "
                      "코호트 구성 단계를 확인하세요."
                  )
  ```

  **Case B — 테스트 PASS (이미 필터 존재) 또는 빈 DF가 허용:**
  테스트 docstring을 현재 동작 명시로 변경:
  ```python
  def test_nonsampling_path_all_zero_followup_returns_empty_df():
      """비샘플링 경로(total <= max_rows)에서 유효 행이 없으면 빈 DataFrame 반환 (의도된 동작)."""
      ...
      df, info = analyzer._load_data()
      assert len(df) == 0
      assert info.applied is False
  ```

- [ ] **Step 4: Run test to verify it passes**

```bash
python3 -m pytest tests/test_stage_c.py -v --tb=short
```

Expected: 5 tests PASS

- [ ] **Step 5: Run full test suite**

```bash
python3 -m pytest tests/ -q --tb=short 2>&1 | tail -8
```

Expected: 기존 5건 실패 유지, 신규 실패 없음

- [ ] **Step 6: Commit**

```bash
git add statistical_analysis.py tests/test_stage_c.py  # 또는 tests만
git commit -m "test: 비샘플링 경로 유효 행 0건 동작 명시 (Stage E)"
```

---

## Self-Review

**Spec coverage:**

| Codex 제안 | Task | 상태 |
|---|---|---|
| 비샘플링 경로 유효 행 0건 동작 명시 | Task 1 + Task 2 | ✓ |

**Placeholder scan:** Task 2 Step 3은 조사 결과에 따른 두 갈래(Case A/B)를 모두 명시했으므로 플레이스홀더 아님.

**의존성:** Task 1 조사 완료 후 Task 2 진행.
