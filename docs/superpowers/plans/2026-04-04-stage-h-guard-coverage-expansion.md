# Stage H: 가드 적용 범위 확대 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stage G 리뷰(Codex + Gemini)에서 이관된 3개 항목 구현 — `InsufficientDataError` 메시지 rows/events 컨텍스트 분기, 하드코딩 `< 100` 임계값 제거, `_check_min_rows()` 적용 범위 확대.

**Architecture:**
- Task 1: `InsufficientDataError`에 `kind` 파라미터 추가(`"rows"` vs `"events"`), `format_error_for_user()`에서 분기하여 올바른 설정 키 안내
- Task 2: `run_subgroup()` / `run_competing_risks()` 의 하드코딩 `100` 을 `STUDY_SETTINGS.get('MIN_VALID_ROWS', 30)` 으로 교체
- Task 3: `run_psm()` 의 `df_ps` dropna 이후 `_check_min_rows()` 적용

**Tech Stack:** Python 3.12, DuckDB, pytest, lifelines

---

## File Map

| File | Change |
|---|---|
| `utils.py:106-141` | `InsufficientDataError.__init__` `kind` 파라미터 추가; `format_error_for_user` 분기 |
| `statistical_analysis.py` | `run_subgroup` `run_competing_risks` 하드코딩 제거; `run_psm` `_check_min_rows` 추가; EPV raise 에 `kind='events'` 추가 |
| `tests/test_stage_h.py` | 신규 테스트 파일 |

---

### Task 1: InsufficientDataError kind 파라미터 + format_error_for_user 분기

**Files:**
- Modify: `utils.py:106-141`
- Test: `tests/test_stage_h.py`

**Context:**
현재 `InsufficientDataError(valid_rows=event_count, min_rows=min_events)` 로 EPV 실패를 raise할 때도 `format_error_for_user`가 "MIN_VALID_ROWS 설정을 조정하세요" 로 안내한다. 사용자는 `MIN_EVENTS` 를 바꿔야 하는데 잘못된 설정 키를 안내받는다.

수정 방향:
- `InsufficientDataError.__init__`에 `kind: str = "rows"` 파라미터 추가 (기존 호출 하위호환)
- `format_error_for_user`에서 `exc.kind == "events"` 이면 `MIN_EVENTS` 안내, 나머지는 `MIN_VALID_ROWS` 안내
- `run_cox()`의 EPV raise에 `kind='events'` 전달

- [ ] **Step 1: Write failing tests**

`tests/test_stage_h.py` 신규 파일 생성:

```python
"""
tests/test_stage_h.py - Stage H 가드 적용 범위 확대 테스트
"""

import pytest
import pandas as pd
from unittest.mock import patch
from utils import format_error_for_user, InsufficientDataError


def test_insufficient_data_error_default_kind_is_rows():
    """kind 미지정 시 기본값 'rows' 여야 한다 (하위호환)."""
    exc = InsufficientDataError(valid_rows=5, min_rows=30)
    assert exc.kind == "rows"


def test_insufficient_data_error_kind_events():
    """kind='events' 로 생성 가능해야 한다."""
    exc = InsufficientDataError(valid_rows=3, min_rows=10, kind="events")
    assert exc.kind == "events"


def test_format_error_rows_kind_mentions_min_valid_rows():
    """rows 종류 에러는 MIN_VALID_ROWS 설정을 안내해야 한다."""
    exc = InsufficientDataError(valid_rows=5, min_rows=30, kind="rows")
    msg = format_error_for_user(exc)
    assert "MIN_VALID_ROWS" in msg, f"MIN_VALID_ROWS 언급 없음: {msg!r}"
    assert "MIN_EVENTS" not in msg, f"잘못된 설정 키 MIN_EVENTS 언급: {msg!r}"


def test_format_error_events_kind_mentions_min_events():
    """events 종류 에러는 MIN_EVENTS 설정을 안내해야 한다."""
    exc = InsufficientDataError(valid_rows=3, min_rows=10, kind="events")
    msg = format_error_for_user(exc)
    assert "MIN_EVENTS" in msg, f"MIN_EVENTS 언급 없음: {msg!r}"
    assert "MIN_VALID_ROWS" not in msg, f"잘못된 설정 키 MIN_VALID_ROWS 언급: {msg!r}"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Volumes/model/yod_diabetes_app
python3 -m pytest tests/test_stage_h.py -v 2>&1 | tail -15
```

Expected: FAIL — `InsufficientDataError.__init__` 에 `kind` 파라미터 없음

- [ ] **Step 3: Add kind parameter to InsufficientDataError and update format_error_for_user**

`utils.py:106-141` 을 다음으로 교체:

```python
class InsufficientDataError(ValueError):
    """분석에 필요한 최소 유효 행/이벤트 수를 충족하지 못할 때 발생.

    kind='rows'  — 행 수 부족 (MIN_VALID_ROWS 관련)
    kind='events' — 이벤트 수 부족 (MIN_EVENTS/EPV 관련)
    """
    def __init__(self, valid_rows: int, min_rows: int, kind: str = "rows"):
        self.valid_rows = valid_rows
        self.min_rows = min_rows
        self.kind = kind
        if kind == "events":
            super().__init__(
                f"이벤트 수({valid_rows:,}건)가 EPV 최소 기준({min_rows:,}건)에 미달합니다. "
                "코호트 크기를 확인하거나 MIN_EVENTS 설정을 조정하세요."
            )
        else:
            super().__init__(
                f"유효 행 수({valid_rows:,}건)가 최소 분석 기준({min_rows:,}건)에 미달합니다. "
                "코호트 크기를 확인하거나 MIN_VALID_ROWS 설정을 낮추세요."
            )
```

그리고 `format_error_for_user` 의 `InsufficientDataError` 분기(`utils.py:137-141`)를:

```python
    if isinstance(exc, InsufficientDataError):
        if exc.kind == "events":
            return (
                f"이벤트(결과 발생) 수 부족: {exc.valid_rows:,}건 (EPV 최소 {exc.min_rows:,}건 필요). "
                "코호트 크기를 확인하거나 MIN_EVENTS 설정을 조정하세요."
            )
        return (
            f"유효 데이터 부족: {exc.valid_rows:,}건 (최소 {exc.min_rows:,}건 필요). "
            "코호트 크기를 확인하거나 MIN_VALID_ROWS 설정을 조정하세요."
        )
```

- [ ] **Step 4: Add kind='events' to run_cox() EPV raise**

`statistical_analysis.py` 에서 EPV 실패 raise 위치(`run_cox` 내 `raise InsufficientDataError(valid_rows=event_count, min_rows=min_events)` 줄):

```python
            raise InsufficientDataError(valid_rows=event_count, min_rows=min_events, kind='events')
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
python3 -m pytest tests/test_stage_h.py -v 2>&1 | tail -10
```

Expected: 4 tests PASS

- [ ] **Step 6: Run full test suite**

```bash
python3 -m pytest tests/ -q --tb=short 2>&1 | tail -8
```

Expected: 기존 5건 실패 유지, 신규 실패 없음

- [ ] **Step 7: Commit**

```bash
git add utils.py statistical_analysis.py tests/test_stage_h.py
git commit -m "feat: InsufficientDataError kind 파라미터 추가 — rows/events 메시지 분기"
```

---

### Task 2: 하드코딩 100 임계값 → STUDY_SETTINGS['MIN_VALID_ROWS']

**Files:**
- Modify: `statistical_analysis.py:535, 645, 735`
- Test: `tests/test_stage_h.py`

**Context:**
`run_subgroup` (line 535), `run_competing_risks` (lines 645, 735) 에 `< 100`, `>= 100` 하드코딩이 있다.
이 값들이 config의 `MIN_VALID_ROWS=30` 과 불일치하면 분석 결과가 달라질 수 있다.
동일한 `STUDY_SETTINGS.get('MIN_VALID_ROWS', 30)` 로 교체하되, 두 메서드 모두 `continue` (예외 아님) 동작을 유지한다.

- [ ] **Step 1: Write failing tests**

`tests/test_stage_h.py` 에 추가:

```python
from statistical_analysis import StatisticalAnalyzer, SamplingInfo
import duckdb


def _make_analyzer_with_df(df):
    """미리 준비된 df 를 _cached_df 로 주입한 분석기."""
    analyzer = StatisticalAnalyzer.__new__(StatisticalAnalyzer)
    analyzer.results = {}
    analyzer._cached_df = df
    analyzer._sampling_info = SamplingInfo(applied=False, total_rows=len(df), sampled_rows=len(df))
    return analyzer


def test_run_subgroup_respects_min_valid_rows_from_config():
    """run_subgroup 이 하드코딩 100 대신 MIN_VALID_ROWS 를 사용한다.

    MIN_VALID_ROWS=30 으로 패치하면 30건 서브그룹이 실행되어야 한다.
    하드코딩 100 이라면 30건은 skip 되어 결과가 비어있다.
    """
    # 50명, 남성 30명 / 여성 20명, 치매 이벤트 5건
    n = 50
    df = pd.DataFrame({
        'follow_up_years': [1.0] * n,
        'dementia_event': [1] * 5 + [0] * (n - 5),
        'exposure_group': ['T2DM_OHA'] * n,
        'is_t1dm': [0] * n,
        'is_t2dm_oha': [1] * n,
        'is_t2dm_insulin': [0] * n,
        'is_t2dm_nomed': [0] * n,
        'male': [1] * 30 + [0] * 20,
        'age_at_index': [60.0] * n,
        'cci_score': [1] * n,
        'age_group': ['55-64'] * n,
    })
    analyzer = _make_analyzer_with_df(df)
    # MIN_VALID_ROWS=30 이면 남성(30건) 서브그룹이 실행되어야 함
    with patch('statistical_analysis.STUDY_SETTINGS', {'MIN_VALID_ROWS': 30, 'MIN_EVENTS': 10, 'SAMPLING_SEED': 42}):
        result = analyzer.run_subgroup(df_prepared=df)
    # 하드코딩 100 이라면 결과가 비어있음, MIN_VALID_ROWS=30 이면 sex_male 이 포함
    assert 'sex_male' in result or len(result) > 0, \
        "MIN_VALID_ROWS=30 인데도 sex_male(30건) 서브그룹이 skip 됨 — 하드코딩 100 사용 중"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python3 -m pytest tests/test_stage_h.py::test_run_subgroup_respects_min_valid_rows_from_config -v 2>&1 | tail -10
```

Expected: FAIL — 결과가 비어있거나 sex_male 없음

- [ ] **Step 3: Replace hardcoded 100 in run_subgroup (line 535)**

`statistical_analysis.py:535` 에서:

```python
                if len(dm) < 100 or dm['dementia_event'].sum() < 5:
                    continue
```

를:

```python
                _min_sg = int(STUDY_SETTINGS.get('MIN_VALID_ROWS', 30))
                if len(dm) < _min_sg or dm['dementia_event'].sum() < 5:
                    continue
```

로 교체.

- [ ] **Step 4: Replace hardcoded 100 in run_competing_risks (lines 645, 735)**

`statistical_analysis.py:645`:

```python
            if len(df_cr) < 100:
                continue
```

를:

```python
            _min_cr = int(STUDY_SETTINGS.get('MIN_VALID_ROWS', 30))
            if len(df_cr) < _min_cr:
                continue
```

로 교체.

`statistical_analysis.py:735`:

```python
                if len(df_fit) >= 100 and df_fit[outcome].sum() >= 5:
```

를:

```python
                if len(df_fit) >= int(STUDY_SETTINGS.get('MIN_VALID_ROWS', 30)) and df_fit[outcome].sum() >= 5:
```

로 교체.

- [ ] **Step 5: Run tests to verify they pass**

```bash
python3 -m pytest tests/test_stage_h.py::test_run_subgroup_respects_min_valid_rows_from_config -v 2>&1 | tail -8
```

Expected: PASS

- [ ] **Step 6: Run full test suite**

```bash
python3 -m pytest tests/ -q --tb=short 2>&1 | tail -8
```

Expected: 기존 5건 실패 유지, 신규 실패 없음

- [ ] **Step 7: Commit**

```bash
git add statistical_analysis.py tests/test_stage_h.py
git commit -m "fix: run_subgroup/run_competing_risks 하드코딩 100 → STUDY_SETTINGS MIN_VALID_ROWS"
```

---

### Task 3: run_psm() dropna 후 _check_min_rows() 적용

**Files:**
- Modify: `statistical_analysis.py:327`
- Test: `tests/test_stage_h.py`

**Context:**
`run_psm()` 의 `df_ps = df_dm[ps_vars + ['is_t1dm']].dropna()` (line 327) 이후 최소 행 수 체크가 없다.
작은 코호트에서 sklearn 로지스틱 회귀가 불명확한 오류를 내기 전에 `_check_min_rows()` 로 조기 실패시킨다.
단, PSM 전체를 중단하지 않고 `skipped` 결과를 반환하는 기존 패턴(`n_treated < 2` 체크, line 332-337)에 맞춰 `InsufficientDataError` 를 잡아서 skip 처리한다.

- [ ] **Step 1: Write failing test**

`tests/test_stage_h.py` 에 추가:

```python
def test_run_psm_skips_on_too_few_rows():
    """run_psm() 이 MIN_VALID_ROWS 미만의 df_ps 에서 skip 결과를 반환한다."""
    n = 10  # MIN_VALID_ROWS(30) 미만
    df = pd.DataFrame({
        'follow_up_years': [1.0] * n,
        'dementia_event': [0] * n,
        'ad_event': [0] * n,
        'vad_event': [0] * n,
        'exposure_group': (['T1DM'] * 5) + (['T2DM_OHA'] * 5),
        'is_t1dm': [1] * 5 + [0] * 5,
        'is_t2dm_oha': [0] * 5 + [1] * 5,
        'is_t2dm_insulin': [0] * n,
        'is_t2dm_nomed': [0] * n,
        'male': [1] * n,
        'age_at_index': [60.0] * n,
        'income_q': [3.0] * n,
        'comor_hypertension': [0] * n,
        'comor_dyslipidemia': [0] * n,
        'dm_duration_years': [5.0] * n,
    })
    analyzer = StatisticalAnalyzer.__new__(StatisticalAnalyzer)
    analyzer.results = {}
    with patch('statistical_analysis.STUDY_SETTINGS', {'MIN_VALID_ROWS': 30, 'MIN_EVENTS': 10, 'SAMPLING_SEED': 42}):
        result = analyzer.run_psm(df_prepared=df)
    assert result.get('skipped') is True, \
        f"MIN_VALID_ROWS=30 인데 행 수 {n}건으로 PSM skip 안 됨: {result}"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python3 -m pytest tests/test_stage_h.py::test_run_psm_skips_on_too_few_rows -v 2>&1 | tail -10
```

Expected: FAIL (sklearn 오류 또는 다른 예외 발생, skipped 반환 안 함)

- [ ] **Step 3: Add _check_min_rows call in run_psm() after dropna**

`statistical_analysis.py` 에서 `df_ps = df_dm[ps_vars + ['is_t1dm']].dropna()` 줄(line ~327) 직후:

```python
        df_ps = df_dm[ps_vars + ['is_t1dm']].dropna()

        try:
            self._check_min_rows(df_ps, context="run_psm")
        except InsufficientDataError as e:
            msg = f"PSM 스킵: {e}"
            logger.warning(msg)
            if cb: cb(msg)
            self.results['psm'] = {'skipped': True, 'reason': msg}
            return self.results['psm']
```

(기존 `# PSM 실행 가능 여부 검증:` 주석 앞에 삽입)

- [ ] **Step 4: Run tests to verify they pass**

```bash
python3 -m pytest tests/test_stage_h.py -v --tb=short 2>&1 | tail -12
```

Expected: all PASS (Task 1 + 2 + 3 테스트 모두)

- [ ] **Step 5: Run full test suite**

```bash
python3 -m pytest tests/ -q --tb=short 2>&1 | tail -8
```

Expected: 기존 5건 실패 유지, 신규 실패 없음

- [ ] **Step 6: Commit**

```bash
git add statistical_analysis.py tests/test_stage_h.py
git commit -m "feat: run_psm() dropna 후 _check_min_rows() 적용 — 조기 skip 처리"
```

---

## Self-Review

**Spec coverage:**

| Stage G 이관 항목 | Task | 상태 |
|---|---|---|
| InsufficientDataError rows/events 메시지 분기 | Task 1 | ✓ |
| 하드코딩 `< 100` → `MIN_VALID_ROWS` | Task 2 | ✓ |
| `run_psm` `_check_min_rows` 적용 | Task 3 | ✓ |

**Placeholder scan:** 없음. 모든 코드 블록 완전함.

**Type consistency:**
- `InsufficientDataError(valid_rows, min_rows, kind='rows')` — Task 1 정의와 Task 3 `_check_min_rows` 내부 raise 일치 (`_check_min_rows` 는 `kind` 미지정 → 기본값 `"rows"` 유지)
- `STUDY_SETTINGS.get('MIN_VALID_ROWS', 30)` — Task 2 세 곳 동일 패턴

**의존성:**
- Task 1 완료 후 Task 2, 3 순서 무관
- 권장 순서: Task 1 → Task 2 → Task 3
