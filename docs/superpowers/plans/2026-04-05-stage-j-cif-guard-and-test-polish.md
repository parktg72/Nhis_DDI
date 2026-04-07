# Stage J: CIF 이벤트 가드 & 테스트 폴리시 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stage I 리뷰(Codex + Gemini + 셀프)에서 도출된 3개 항목 구현: CIF per-group 이벤트 수 가드 추가, PSM 테스트 단언 강화, `_min_sg` 변수명 명확화.

**Architecture:** 모든 변경은 `statistical_analysis.py`(가드 로직) + `tests/test_stage_i.py`(기존 테스트 강화) + `tests/test_stage_j.py`(신규 테스트)에 국한된다. config.py·utils.py 변경 없음.

**Tech Stack:** Python 3.12, pytest, pandas, lifelines (CoxPHFitter, KaplanMeierFitter), unittest.mock

---

## 파일 구조

| 파일 | 변경 내용 |
|------|-----------|
| `statistical_analysis.py` | `run_competing_risks` CIF per-group 루프(line ~684, ~702) 이벤트 수 guard 추가; `run_subgroup` `_min_sg` → `_min_sg_rows` 변수명 |
| `tests/test_stage_i.py` | `test_run_psm_skip_reason_uses_format_error_for_user` 단언 강화 — `MIN_VALID_ROWS` 포함 여부 추가 |
| `tests/test_stage_j.py` | 신규: CIF per-group 이벤트 가드 2개 테스트 |

---

## Background: 현재 코드 상태

`statistical_analysis.py` `run_competing_risks` 안의 CIF per-group 루프 (line ~677–712):

```python
# 노출군별 CIF (line ~677-695)
for group_col, group_name in [...]:
    mask = df_cr[group_col].values == 1
    if mask.sum() < _min_cr:          # ← 행 수만 확인, 이벤트 수 미확인 (버그)
        continue
    ...CIF 계산...

# NON_DM CIF (line ~698-712)
if non_dm_mask.sum() >= _min_cr:      # ← 행 수만 확인, 이벤트 수 미확인 (버그)
    ...CIF 계산...
```

수정 목표: 두 곳 모두 `(event_type[mask] == 1).sum() < _min_cr_events` 가드 추가.

`run_subgroup` (line ~533):
```python
_min_sg = int(STUDY_SETTINGS.get('MIN_VALID_ROWS', 30))      # ← 명칭 모호
_min_sg_events = int(STUDY_SETTINGS.get('MIN_SUBGROUP_EVENTS', 5))
```

---

### Task 1: CIF per-group 이벤트 수 가드 (TDD)

**Files:**
- Create: `tests/test_stage_j.py`
- Modify: `statistical_analysis.py` (line ~684, ~702)

- [ ] **Step 1: `tests/test_stage_j.py` 신규 파일 생성 후 failing test 2개 작성**

```python
"""
tests/test_stage_j.py - Stage J: CIF per-group 이벤트 가드 + MIN_SUBGROUP_EVENTS 테스트
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch
from statistical_analysis import StatisticalAnalyzer, SamplingInfo


def _make_analyzer_with_df(df):
    analyzer = StatisticalAnalyzer.__new__(StatisticalAnalyzer)
    analyzer.results = {}
    analyzer._cached_df = df
    analyzer._sampling_info = SamplingInfo(applied=False, total_rows=len(df), sampled_rows=len(df))
    return analyzer


def test_cif_skips_group_with_zero_events():
    """CIF per-group 루프가 이벤트 0건 그룹을 skip 해야 한다.

    T1DM: 15행, 0 이벤트 → MIN_SUBGROUP_EVENTS=3 → CIF skip 되어야 함
    T2DM_OHA: 25행, 5 이벤트 → CIF 포함되어야 함
    현재 코드는 행 수만 확인하므로 T1DM 도 포함됨 — 이 테스트는 수정 전 FAIL.
    """
    n = 40
    df = pd.DataFrame({
        'follow_up_years': [1.0] * n,
        'dementia_event': [0] * 15 + [1] * 5 + [0] * 20,  # T1DM=0건, T2DM_OHA=5건
        'competing_death_event': [0] * n,
        'is_t1dm': [1] * 15 + [0] * 25,
        'is_t2dm_oha': [0] * 15 + [1] * 25,
        'is_t2dm_insulin': [0] * n,
        'is_t2dm_nomed': [0] * n,
        'age_at_index': [60.0] * n,
        'male': [1] * n,
    })
    analyzer = _make_analyzer_with_df(df)
    with patch('statistical_analysis.STUDY_SETTINGS',
               {'MIN_VALID_ROWS': 10, 'MIN_EVENTS': 3, 'MIN_SUBGROUP_EVENTS': 3, 'SAMPLING_SEED': 42}):
        with patch('gpu_accelerator.is_gpu_enabled', return_value=False):
            result = analyzer.run_competing_risks(df_prepared=df)
    cif = result.get('dementia_event', {}).get('cif_by_group', {})
    assert 'T1DM' not in cif, \
        f"이벤트 0건 T1DM 이 CIF 에 포함됨 — 이벤트 수 가드 미적용: {list(cif.keys())}"
    assert 'T2DM_OHA' in cif, \
        f"이벤트 5건 T2DM_OHA 가 CIF 에서 누락됨: {list(cif.keys())}"


def test_cif_respects_min_subgroup_events_threshold():
    """MIN_SUBGROUP_EVENTS 를 임계값 위아래로 패치해 CIF 포함/skip 전환을 검증한다.

    T2DM_OHA: 25행, 4 이벤트
    MIN_SUBGROUP_EVENTS=3 → 포함 (4 >= 3)
    MIN_SUBGROUP_EVENTS=5 → skip  (4 < 5)
    """
    n = 40
    df = pd.DataFrame({
        'follow_up_years': [1.0] * n,
        'dementia_event': [0] * 15 + [1] * 4 + [0] * 21,  # T1DM=0건, T2DM_OHA=4건
        'competing_death_event': [0] * n,
        'is_t1dm': [1] * 15 + [0] * 25,
        'is_t2dm_oha': [0] * 15 + [1] * 25,
        'is_t2dm_insulin': [0] * n,
        'is_t2dm_nomed': [0] * n,
        'age_at_index': [60.0] * n,
        'male': [1] * n,
    })
    analyzer = _make_analyzer_with_df(df)

    # MIN_SUBGROUP_EVENTS=3 → T2DM_OHA (4건) 포함
    with patch('statistical_analysis.STUDY_SETTINGS',
               {'MIN_VALID_ROWS': 10, 'MIN_EVENTS': 3, 'MIN_SUBGROUP_EVENTS': 3, 'SAMPLING_SEED': 42}):
        with patch('gpu_accelerator.is_gpu_enabled', return_value=False):
            result_runs = analyzer.run_competing_risks(df_prepared=df)

    # MIN_SUBGROUP_EVENTS=5 → T2DM_OHA (4건) skip
    with patch('statistical_analysis.STUDY_SETTINGS',
               {'MIN_VALID_ROWS': 10, 'MIN_EVENTS': 3, 'MIN_SUBGROUP_EVENTS': 5, 'SAMPLING_SEED': 42}):
        with patch('gpu_accelerator.is_gpu_enabled', return_value=False):
            result_skips = analyzer.run_competing_risks(df_prepared=df)

    cif_runs = result_runs.get('dementia_event', {}).get('cif_by_group', {})
    cif_skips = result_skips.get('dementia_event', {}).get('cif_by_group', {})
    assert 'T2DM_OHA' in cif_runs, \
        f"MIN_SUBGROUP_EVENTS=3 인데 이벤트 4건 T2DM_OHA 가 CIF 에서 누락됨: {list(cif_runs.keys())}"
    assert 'T2DM_OHA' not in cif_skips, \
        f"MIN_SUBGROUP_EVENTS=5 인데 이벤트 4건 T2DM_OHA 가 CIF 에 포함됨: {list(cif_skips.keys())}"
```

- [ ] **Step 2: 테스트가 FAIL 하는지 확인 (현재 코드에서 행 수만 체크하므로 두 테스트 모두 실패해야 함)**

```bash
cd /Volumes/model/yod_diabetes_app
pytest tests/test_stage_j.py -v 2>&1 | tail -20
```

Expected: 2 FAILED (T1DM 이 cif 에 포함, T2DM_OHA 가 skip 되지 않음)

- [ ] **Step 3: `statistical_analysis.py` CIF per-group 루프에 이벤트 수 가드 추가**

`statistical_analysis.py` line ~681-685 구간을 다음과 같이 수정:

```python
# 수정 전 (line ~681-685):
                mask = df_cr[group_col].values == 1
                if mask.sum() < _min_cr:
                    continue

# 수정 후:
                mask = df_cr[group_col].values == 1
                if mask.sum() < _min_cr or (event_type[mask] == 1).sum() < _min_cr_events:
                    continue
```

`statistical_analysis.py` line ~700-702 구간을 다음과 같이 수정:

```python
# 수정 전 (line ~700-702):
            if non_dm_mask.sum() >= _min_cr:

# 수정 후:
            if (non_dm_mask.sum() >= _min_cr and
                    (event_type[non_dm_mask.values] == 1).sum() >= _min_cr_events):
```

- [ ] **Step 4: 테스트가 PASS 하는지 확인**

```bash
pytest tests/test_stage_j.py -v 2>&1 | tail -15
```

Expected: 2 PASSED

- [ ] **Step 5: 전체 테스트 스위트 실행 (기존 테스트 회귀 없는지 확인)**

```bash
pytest tests/ -x -q 2>&1 | tail -15
```

Expected: 기존 통과 테스트 유지, 신규 2개 추가

- [ ] **Step 6: 커밋**

```bash
git add tests/test_stage_j.py statistical_analysis.py
git commit -m "fix: CIF per-group 이벤트 수 가드 추가 (Stage J Task 1)"
```

---

### Task 2: PSM 테스트 단언 강화 + `_min_sg` 변수명 명확화

**Files:**
- Modify: `tests/test_stage_i.py` (line ~52-54)
- Modify: `statistical_analysis.py` (line ~533, ~545)

- [ ] **Step 1: `tests/test_stage_i.py` PSM 테스트 단언 강화**

`tests/test_stage_i.py` 파일에서 `test_run_psm_skip_reason_uses_format_error_for_user` 함수의 단언 부분을 수정한다.

현재 (line ~52-54):
```python
    reason = result.get('reason', '')
    # format_error_for_user 경유 시 "유효 데이터 부족:" 형식, str(e) 경유 시 "유효 행 수(" 형식
    assert '유효 데이터 부족' in reason, \
        f"skip reason 에 '유효 데이터 부족' 없음 — format_error_for_user 미사용(str(e) 사용 중): {reason!r}"
```

수정 후:
```python
    reason = result.get('reason', '')
    # format_error_for_user 경유 시 "유효 데이터 부족: ... MIN_VALID_ROWS ..." 형식
    # str(e) 경유 시 "유효 행 수(..." 형식 — MIN_VALID_ROWS 언급 없음
    assert '유효 데이터 부족' in reason, \
        f"skip reason 에 '유효 데이터 부족' 없음 — format_error_for_user 미사용(str(e) 사용 중): {reason!r}"
    assert 'MIN_VALID_ROWS' in reason, \
        f"skip reason 에 'MIN_VALID_ROWS' 없음 — format_error_for_user 미경유(str(e) 사용 중): {reason!r}"
```

- [ ] **Step 2: 테스트가 여전히 PASS 하는지 확인 (강화된 단언이 현재 구현을 통과해야 함)**

```bash
pytest tests/test_stage_i.py::test_run_psm_skip_reason_uses_format_error_for_user -v
```

Expected: PASSED (format_error_for_user 가 이미 MIN_VALID_ROWS 를 포함하므로)

- [ ] **Step 3: `statistical_analysis.py` `_min_sg` → `_min_sg_rows` 변수명 변경**

`statistical_analysis.py` 에서 `run_subgroup` 함수 내 `_min_sg` 변수명을 `_min_sg_rows` 로 변경한다.

현재:
```python
        _min_sg = int(STUDY_SETTINGS.get('MIN_VALID_ROWS', 30))
        _min_sg_events = int(STUDY_SETTINGS.get('MIN_SUBGROUP_EVENTS', 5))
        for name, mask in subgroups.items():
            ...
            if len(dm) < _min_sg or dm['dementia_event'].sum() < _min_sg_events:
```

수정 후:
```python
        _min_sg_rows = int(STUDY_SETTINGS.get('MIN_VALID_ROWS', 30))
        _min_sg_events = int(STUDY_SETTINGS.get('MIN_SUBGROUP_EVENTS', 5))
        for name, mask in subgroups.items():
            ...
            if len(dm) < _min_sg_rows or dm['dementia_event'].sum() < _min_sg_events:
```

- [ ] **Step 4: 관련 테스트 PASS 확인**

```bash
pytest tests/test_stage_h.py::test_run_subgroup_respects_min_valid_rows_from_config \
       tests/test_stage_i.py::test_run_subgroup_respects_min_subgroup_events -v
```

Expected: 2 PASSED

- [ ] **Step 5: 전체 테스트 스위트 실행**

```bash
pytest tests/ -x -q 2>&1 | tail -15
```

Expected: 기존 통과 테스트 유지

- [ ] **Step 6: 커밋**

```bash
git add tests/test_stage_i.py statistical_analysis.py
git commit -m "refactor: PSM 테스트 단언 강화 + _min_sg → _min_sg_rows 변수명 (Stage J Task 2)"
```

---

## 완료 기준

- `tests/test_stage_j.py` 2개 테스트 PASS
- `tests/test_stage_i.py::test_run_psm_skip_reason_uses_format_error_for_user` `MIN_VALID_ROWS` 단언 포함 후 PASS
- `_min_sg` → `_min_sg_rows` 전체 반영 (2개 라인)
- `pytest tests/ -q` 신규 실패 없음
