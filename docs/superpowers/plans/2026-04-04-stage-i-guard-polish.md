# Stage I: 가드 마무리 폴리시 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stage H 리뷰(Codex + Gemini + 셀프)에서 Stage I 이관된 3개 항목 구현:
1. `run_subgroup` / `run_competing_risks` 의 이벤트 플로어 `5` → `config.py` `MIN_SUBGROUP_EVENTS` 설정값
2. `run_psm` skip reason에서 `str(e)` 대신 `format_error_for_user(e)` 사용
3. `run_competing_risks` `outcome='dementia_event'` 경로 테스트 커버리지 추가 (need_cols 중복 dedup 검증)

**Architecture:**
- Task 1: `config.py`에 `MIN_SUBGROUP_EVENTS: 5` 추가, `run_subgroup`/`run_competing_risks` 하드코딩 `5` 교체
- Task 2: `run_psm` skip reason을 `format_error_for_user(e)` 로 교체
- Task 3: `test_stage_i.py`에 `outcome='dementia_event'` 경로 테스트 추가

**Tech Stack:** Python 3.12, DuckDB, pytest, lifelines

---

## File Map

| File | Change |
|---|---|
| `config.py` | `STUDY_SETTINGS['MIN_SUBGROUP_EVENTS'] = 5` 추가 |
| `statistical_analysis.py:544` | `sum() < 5` → `sum() < min_sg_events` |
| `statistical_analysis.py:746` | `sum() >= 5` → `sum() >= min_cr_events` |
| `statistical_analysis.py:332` | `f"PSM 스킵: {e}"` → `format_error_for_user(e)` |
| `tests/test_stage_i.py` | 신규 테스트 파일 |

---

### Task 1: MIN_SUBGROUP_EVENTS 설정값 — 이벤트 플로어 5 교체

**Files:**
- Modify: `config.py`
- Modify: `statistical_analysis.py:544, 746`
- Test: `tests/test_stage_i.py`

**Context:**
`run_subgroup` (line 544) 과 `run_competing_risks` Fine-Gray (line 746) 에 `< 5` / `>= 5` 이벤트 플로어가 하드코딩되어 있다.
`MIN_EVENTS=10` (EPV, Cox 전체) 와는 별도의 서브그룹/Fine-Gray 용 임계값이므로 `MIN_SUBGROUP_EVENTS=5` 로 분리한다.

- [ ] **Step 1: config.py에 MIN_SUBGROUP_EVENTS 추가**

`config.py` 의 `MIN_EVENTS` 아래에 추가:

현재:
```python
    'MIN_EVENTS': 10,             # Cox 분석 최소 이벤트 수 (EPV heuristic)
```

추가 후:
```python
    'MIN_EVENTS': 10,             # Cox 분석 최소 이벤트 수 (EPV heuristic)
    'MIN_SUBGROUP_EVENTS': 5,     # 서브그룹/Fine-Gray 분석 최소 이벤트 수
```

- [ ] **Step 2: Write failing tests**

`tests/test_stage_i.py` 신규 파일 생성:

```python
"""
tests/test_stage_i.py - Stage I 가드 마무리 폴리시 테스트
"""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from statistical_analysis import StatisticalAnalyzer, SamplingInfo


def _make_analyzer_with_df(df):
    analyzer = StatisticalAnalyzer.__new__(StatisticalAnalyzer)
    analyzer.results = {}
    analyzer._cached_df = df
    analyzer._sampling_info = SamplingInfo(applied=False, total_rows=len(df), sampled_rows=len(df))
    return analyzer


def test_run_subgroup_respects_min_subgroup_events():
    """run_subgroup 이 하드코딩 5 대신 MIN_SUBGROUP_EVENTS 를 사용한다.

    MIN_SUBGROUP_EVENTS=10 으로 패치하면 이벤트 5건인 서브그룹이 skip 되어야 한다.
    """
    n = 50
    df = pd.DataFrame({
        'follow_up_years': [1.0] * n,
        'dementia_event': [1] * 5 + [0] * (n - 5),  # 이벤트 정확히 5건
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
    # MIN_SUBGROUP_EVENTS=10 이면 이벤트 5건 서브그룹은 skip 되어야 함
    with patch('statistical_analysis.STUDY_SETTINGS',
               {'MIN_VALID_ROWS': 30, 'MIN_EVENTS': 10, 'MIN_SUBGROUP_EVENTS': 10, 'SAMPLING_SEED': 42}):
        with patch('lifelines.CoxPHFitter') as mock_cph:
            mock_cph.return_value.fit.return_value = None
            mock_cph.return_value.summary = pd.DataFrame()
            mock_cph.return_value.concordance_index_ = 0.5
            result = analyzer.run_subgroup(df_prepared=df)
    # 이벤트 5건 < MIN_SUBGROUP_EVENTS(10) → 모든 서브그룹 skip
    assert len(result) == 0, \
        f"MIN_SUBGROUP_EVENTS=10 인데 이벤트 5건 서브그룹이 skip 안 됨: {list(result.keys())}"
```

- [ ] **Step 3: Run test to verify it fails**

```bash
cd /Volumes/model/yod_diabetes_app
python3 -m pytest tests/test_stage_i.py::test_run_subgroup_respects_min_subgroup_events -v 2>&1 | tail -10
```

Expected: FAIL (이벤트 5건이 skip 안 됨 — 하드코딩 5 사용 중)

- [ ] **Step 4: Replace hardcoded 5 in run_subgroup (line 544)**

`statistical_analysis.py` 에서 `_min_sg` 가 정의된 직후 `_min_sg_events` 도 추가:

현재 (line 533-544):
```python
        sg_results = {}
        _min_sg = int(STUDY_SETTINGS.get('MIN_VALID_ROWS', 30))
        for name, mask in subgroups.items():
            ...
                if len(dm) < _min_sg or dm['dementia_event'].sum() < 5:
                    continue
```

수정 후:
```python
        sg_results = {}
        _min_sg = int(STUDY_SETTINGS.get('MIN_VALID_ROWS', 30))
        _min_sg_events = int(STUDY_SETTINGS.get('MIN_SUBGROUP_EVENTS', 5))
        for name, mask in subgroups.items():
            ...
                if len(dm) < _min_sg or dm['dementia_event'].sum() < _min_sg_events:
                    continue
```

- [ ] **Step 5: Replace hardcoded 5 in run_competing_risks (line 746)**

`statistical_analysis.py` 에서 `_min_cr` 정의 직후 `_min_cr_events` 추가:

현재 (line 642-643):
```python
        results = {}
        _min_cr = int(STUDY_SETTINGS.get('MIN_VALID_ROWS', 30))
```

수정 후:
```python
        results = {}
        _min_cr = int(STUDY_SETTINGS.get('MIN_VALID_ROWS', 30))
        _min_cr_events = int(STUDY_SETTINGS.get('MIN_SUBGROUP_EVENTS', 5))
```

그리고 line 746:
```python
                if len(df_fit) >= _min_cr and df_fit[outcome].sum() >= _min_cr_events:
```

- [ ] **Step 6: Run test to verify it passes**

```bash
python3 -m pytest tests/test_stage_i.py::test_run_subgroup_respects_min_subgroup_events -v 2>&1 | tail -8
```

Expected: PASS

- [ ] **Step 7: Run full test suite**

```bash
python3 -m pytest tests/ -q --tb=short 2>&1 | tail -8
```

Expected: 기존 5건 실패 유지, 신규 실패 없음

- [ ] **Step 8: Commit**

```bash
git add config.py statistical_analysis.py tests/test_stage_i.py
git commit -m "feat: MIN_SUBGROUP_EVENTS 설정값 추가 — run_subgroup/run_competing_risks 이벤트 플로어 교체"
```

---

### Task 2: run_psm skip reason → format_error_for_user

**Files:**
- Modify: `statistical_analysis.py:332`
- Test: `tests/test_stage_i.py`

**Context:**
현재 `run_psm` 의 `except InsufficientDataError as e:` 블록에서 `msg = f"PSM 스킵: {e}"` 로 `str(e)` 를 사용한다.
`format_error_for_user(e)` 가 사용자 친화적 메시지를 제공하는 공식 경로이므로 일관성을 위해 교체한다.
단, `run_selected` 의 로그나 UI 상에서 "PSM 스킵: " 접두사가 있어야 식별이 가능하므로 접두사는 유지한다.

- [ ] **Step 1: Write failing test**

`tests/test_stage_i.py` 에 추가:

```python
from utils import format_error_for_user, InsufficientDataError


def test_run_psm_skip_reason_uses_format_error_for_user():
    """run_psm skip reason 이 format_error_for_user 메시지를 포함해야 한다.

    MIN_VALID_ROWS=30 으로 패치, 10건 df → skip 됨.
    reason 에 'MIN_VALID_ROWS' 가 포함되어야 한다 (format_error_for_user 경유).
    현재 str(e) 사용 시 reason 에는 원본 예외 메시지만 포함됨.
    """
    n = 10
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
    with patch('statistical_analysis.STUDY_SETTINGS',
               {'MIN_VALID_ROWS': 30, 'MIN_EVENTS': 10, 'MIN_SUBGROUP_EVENTS': 5, 'SAMPLING_SEED': 42}):
        result = analyzer.run_psm(df_prepared=df)
    assert result.get('skipped') is True
    # format_error_for_user 경유 시 'MIN_VALID_ROWS' 가 포함됨
    reason = result.get('reason', '')
    assert 'MIN_VALID_ROWS' in reason, \
        f"skip reason 에 MIN_VALID_ROWS 없음 — format_error_for_user 미사용: {reason!r}"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python3 -m pytest tests/test_stage_i.py::test_run_psm_skip_reason_uses_format_error_for_user -v 2>&1 | tail -10
```

Expected: FAIL (`MIN_VALID_ROWS` 가 reason 에 없음 — 현재 `str(e)` 사용)

- [ ] **Step 3: Update run_psm() skip reason**

`statistical_analysis.py:332` 에서:

현재:
```python
        except InsufficientDataError as e:
            msg = f"PSM 스킵: {e}"
            if cb: cb(msg)
```

수정 후:
```python
        except InsufficientDataError as e:
            msg = f"PSM 스킵: {format_error_for_user(e)}"
            if cb: cb(msg)
```

`format_error_for_user` 는 이미 `from utils import setup_logging, format_error_for_user, InsufficientDataError` 로 import 되어 있음.

- [ ] **Step 4: Run test to verify it passes**

```bash
python3 -m pytest tests/test_stage_i.py::test_run_psm_skip_reason_uses_format_error_for_user -v 2>&1 | tail -8
```

Expected: PASS

- [ ] **Step 5: Run full test suite**

```bash
python3 -m pytest tests/ -q --tb=short 2>&1 | tail -8
```

Expected: 기존 5건 실패 유지, 신규 실패 없음

- [ ] **Step 6: Commit**

```bash
git add statistical_analysis.py tests/test_stage_i.py
git commit -m "fix: run_psm skip reason 에 format_error_for_user 사용"
```

---

### Task 3: run_competing_risks outcome='dementia_event' 경로 테스트

**Files:**
- Test: `tests/test_stage_i.py`

**Context:**
Stage H 즉시 반영으로 `need_cols` 중복 컬럼 버그가 수정되었다 (`dict.fromkeys` dedup).
그러나 `test_stage_h.py` 의 `test_run_competing_risks_respects_min_valid_rows_from_config` 는 `dementia_event` 컬럼을 드롭하여 이 경로를 우회했다.
`outcome='dementia_event'` 일 때 중복 없이 실행되는지 직접 검증하는 테스트가 필요하다.

- [ ] **Step 1: Write failing test (should PASS after dedup fix)**

`tests/test_stage_i.py` 에 추가:

```python
def test_run_competing_risks_dementia_event_no_duplicate_column_error():
    """outcome='dementia_event' 일 때 need_cols 중복으로 인한 오류 없이 실행돼야 한다.

    Stage H 즉시 반영: dict.fromkeys 로 need_cols 중복 제거.
    이 테스트는 회귀 방지 — 중복 dedup 이 제거되면 여기서 실패한다.
    """
    n = 35
    df = pd.DataFrame({
        'follow_up_years': [1.0] * n,
        'dementia_event': [1] * 5 + [0] * (n - 5),
        'competing_death_event': [0] * n,
        'is_t1dm': [0] * n,
        'is_t2dm_oha': [1] * n,
        'is_t2dm_insulin': [0] * n,
        'is_t2dm_nomed': [0] * n,
        'age_at_index': [60.0] * n,
        'male': [1] * n,
    })
    analyzer = _make_analyzer_with_df(df)
    with patch('statistical_analysis.STUDY_SETTINGS',
               {'MIN_VALID_ROWS': 30, 'MIN_EVENTS': 10, 'MIN_SUBGROUP_EVENTS': 5, 'SAMPLING_SEED': 42}):
        with patch('statistical_analysis.is_gpu_enabled', return_value=False):
            with patch('statistical_analysis.compute_cif_gpu', return_value=None):
                # 예외 없이 실행되어야 함 (중복 컬럼 IndexError 방지)
                result = analyzer.run_competing_risks(df_prepared=df)
    # dementia_event 키가 결과에 있어야 함
    assert 'dementia_event' in result, \
        f"outcome='dementia_event' 결과 없음: {list(result.keys())}"
```

- [ ] **Step 2: Run test to verify it passes (dedup fix already applied)**

```bash
python3 -m pytest tests/test_stage_i.py::test_run_competing_risks_dementia_event_no_duplicate_column_error -v 2>&1 | tail -10
```

Expected: PASS (Stage H 즉시 반영으로 이미 수정됨)

- [ ] **Step 3: Run full test suite**

```bash
python3 -m pytest tests/ -q --tb=short 2>&1 | tail -8
```

Expected: 기존 5건 실패 유지, 신규 실패 없음

- [ ] **Step 4: Commit**

```bash
git add tests/test_stage_i.py
git commit -m "test: run_competing_risks dementia_event 경로 회귀 테스트 추가"
```

---

## Self-Review

**Spec coverage:**

| Stage H 이관 항목 | Task | 상태 |
|---|---|---|
| 이벤트 플로어 `5` → `MIN_SUBGROUP_EVENTS` | Task 1 | ✓ |
| `run_psm` skip reason `format_error_for_user` | Task 2 | ✓ |
| `dementia_event` 경로 테스트 커버 | Task 3 | ✓ |

**Placeholder scan:** 없음. 모든 코드 블록 완전함.

**Type consistency:**
- `STUDY_SETTINGS.get('MIN_SUBGROUP_EVENTS', 5)` — Task 1 config 추가와 run_subgroup/competing_risks 접근 일치
- `format_error_for_user` — 이미 `from utils import` 에 포함되어 있어 추가 import 불필요

**의존성:**
- Task 1 완료 후 Task 2, 3 병렬 가능
- Task 3 은 Stage H 즉시 반영(dedup fix)이 전제 — 이미 main에 커밋됨
- 권장 순서: Task 1 → Task 2 → Task 3
