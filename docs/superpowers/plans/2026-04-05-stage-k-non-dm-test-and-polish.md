# Stage K: NON_DM CIF 테스트 + 코드 폴리시 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stage J 리뷰(Codex + Gemini + 셀프)에서 도출된 3개 항목 구현: NON_DM CIF 브랜치 테스트 추가, 오해 유발 주석 수정, `df_cr.loc[mask]` → `df_cr[mask]` 표현 명확화.

**Architecture:** 모든 변경은 `tests/test_stage_j.py`(신규 테스트 2개 추가) + `statistical_analysis.py`(line 686, 704 표현 명확화) 에 국한된다. 기능 동작 변경 없음.

**Tech Stack:** Python 3.12, pytest, pandas, unittest.mock

---

## 파일 구조

| 파일 | 변경 내용 |
|------|-----------|
| `tests/test_stage_j.py` | NON_DM CIF 이벤트 가드 테스트 2개 추가; 기존 주석(line 25) 수정 |
| `statistical_analysis.py` | `df_cr.loc[mask, T]` → `df_cr[mask][T]` (line 686, 704) |

---

## Background: 현재 코드 상태

`statistical_analysis.py` `run_competing_risks()` 내 CIF 섹션:

```python
# per-group 루프 (line ~683-695)
mask = df_cr[group_col].values == 1          # numpy bool array
if mask.sum() < _min_cr or (event_type[mask] == 1).sum() < _min_cr_events:
    continue
times_g = df_cr.loc[mask, T].values.astype(float)   # ← WARN: loc + numpy bool
events_g = event_type[mask]

# NON_DM 블록 (line ~697-712)
non_dm_mask = (...)                           # pandas bool Series
if (non_dm_mask.sum() >= _min_cr and
        (event_type[non_dm_mask.values] == 1).sum() >= _min_cr_events):
    times_g = df_cr.loc[non_dm_mask, T].values.astype(float)   # ← WARN: loc + Series
    events_g = event_type[non_dm_mask.values]
```

- `df_cr.loc[mask, T]`는 numpy bool → loc 혼용 (기능상 정확하나 `df_cr[mask][T]`가 의도를 더 명확히 표현)
- NON_DM 가드는 Stage J에서 추가됐으나 `tests/test_stage_j.py`에 커버리지 없음

---

### Task 1: NON_DM CIF 이벤트 가드 테스트 추가

**Files:**
- Modify: `tests/test_stage_j.py` (파일 끝에 2개 테스트 추가)

- [ ] **Step 1: 기존 테스트 현황 확인**

```bash
cd /path/to/worktree
/usr/bin/env python3 -m pytest tests/test_stage_j.py -v
```

Expected: 2 PASSED (기존 per-group 테스트)

- [ ] **Step 2: `tests/test_stage_j.py` 끝에 2개 테스트 추가**

파일 끝(line 90 다음)에 다음을 추가한다:

```python

def test_cif_non_dm_skips_group_with_zero_events():
    """NON_DM CIF 블록이 이벤트 0건일 때 skip 해야 한다.

    NON_DM: 15행, 0 이벤트 → MIN_SUBGROUP_EVENTS=3 → CIF skip 되어야 함
    T2DM_OHA: 35행, 5 이벤트 → CIF 포함되어야 함
    Stage J 에서 추가된 NON_DM 이벤트 수 가드의 회귀 방지 테스트.
    """
    n = 50
    # 행 0-14: is_t1dm=0, is_t2dm_oha=0, ... → NON_DM (15행, 0 이벤트)
    # 행 15-49: is_t2dm_oha=1 → T2DM_OHA (35행, 5 이벤트)
    df = pd.DataFrame({
        'follow_up_years': [1.0] * n,
        'dementia_event': [0] * 15 + [1] * 5 + [0] * 30,  # NON_DM=0건, T2DM_OHA=5건
        'competing_death_event': [0] * n,
        'is_t1dm': [0] * n,
        'is_t2dm_oha': [0] * 15 + [1] * 35,
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
    assert 'NON_DM' not in cif, \
        f"이벤트 0건 NON_DM 이 CIF 에 포함됨 — NON_DM 이벤트 수 가드 미적용: {list(cif.keys())}"
    assert 'T2DM_OHA' in cif, \
        f"이벤트 5건 T2DM_OHA 가 CIF 에서 누락됨: {list(cif.keys())}"


def test_cif_non_dm_respects_min_subgroup_events_threshold():
    """NON_DM CIF 블록이 MIN_SUBGROUP_EVENTS 임계값을 정확히 적용한다.

    NON_DM: 15행, 4 이벤트
    MIN_SUBGROUP_EVENTS=3 → 포함 (4 >= 3)
    MIN_SUBGROUP_EVENTS=5 → skip  (4 < 5)
    """
    n = 50
    # 행 0-14: NON_DM (15행) — 이 중 4건 이벤트
    # 행 15-49: T2DM_OHA (35행, 5 이벤트)
    df = pd.DataFrame({
        'follow_up_years': [1.0] * n,
        'dementia_event': [1] * 4 + [0] * 11 + [1] * 5 + [0] * 30,  # NON_DM=4건, T2DM_OHA=5건
        'competing_death_event': [0] * n,
        'is_t1dm': [0] * n,
        'is_t2dm_oha': [0] * 15 + [1] * 35,
        'is_t2dm_insulin': [0] * n,
        'is_t2dm_nomed': [0] * n,
        'age_at_index': [60.0] * n,
        'male': [1] * n,
    })
    analyzer = _make_analyzer_with_df(df)

    # MIN_SUBGROUP_EVENTS=3 → NON_DM (4건) 포함
    with patch('statistical_analysis.STUDY_SETTINGS',
               {'MIN_VALID_ROWS': 10, 'MIN_EVENTS': 3, 'MIN_SUBGROUP_EVENTS': 3, 'SAMPLING_SEED': 42}):
        with patch('gpu_accelerator.is_gpu_enabled', return_value=False):
            result_runs = analyzer.run_competing_risks(df_prepared=df)

    # MIN_SUBGROUP_EVENTS=5 → NON_DM (4건) skip
    with patch('statistical_analysis.STUDY_SETTINGS',
               {'MIN_VALID_ROWS': 10, 'MIN_EVENTS': 3, 'MIN_SUBGROUP_EVENTS': 5, 'SAMPLING_SEED': 42}):
        with patch('gpu_accelerator.is_gpu_enabled', return_value=False):
            result_skips = analyzer.run_competing_risks(df_prepared=df)

    cif_runs = result_runs.get('dementia_event', {}).get('cif_by_group', {})
    cif_skips = result_skips.get('dementia_event', {}).get('cif_by_group', {})
    assert 'NON_DM' in cif_runs, \
        f"MIN_SUBGROUP_EVENTS=3 인데 이벤트 4건 NON_DM 이 CIF 에서 누락됨: {list(cif_runs.keys())}"
    assert 'NON_DM' not in cif_skips, \
        f"MIN_SUBGROUP_EVENTS=5 인데 이벤트 4건 NON_DM 이 CIF 에 포함됨: {list(cif_skips.keys())}"
```

- [ ] **Step 3: 새 테스트가 PASS 하는지 확인 (NON_DM 가드는 이미 Stage J에서 구현됨)**

```bash
/usr/bin/env python3 -m pytest tests/test_stage_j.py -v
```

Expected: 4 PASSED (기존 2개 + 신규 2개)

- [ ] **Step 4: 커밋**

```bash
git add tests/test_stage_j.py
git commit -m "test: NON_DM CIF 이벤트 가드 커버리지 추가 (Stage K Task 1)"
```

---

### Task 2: 주석 수정 + `df_cr.loc[mask]` → `df_cr[mask]` 표현 명확화

**Files:**
- Modify: `tests/test_stage_j.py` (line 25 주석)
- Modify: `statistical_analysis.py` (line 686, 704)

- [ ] **Step 1: `tests/test_stage_j.py` line 25 주석 수정**

현재:
```python
    현재 코드는 행 수만 확인하므로 T1DM 도 포함됨 — 이 테스트는 수정 전 FAIL.
```

수정 후:
```python
    Stage J 이전에는 행 수만 확인해 T1DM 이 포함됐으나, 이벤트 수 가드 추가(Stage J)로 수정됨.
```

- [ ] **Step 2: `statistical_analysis.py` line 686 수정 (`df_cr.loc[mask, T]` → `df_cr[mask][T]`)**

현재 (line ~686):
```python
                times_g = df_cr.loc[mask, T].values.astype(float)
```

수정 후:
```python
                times_g = df_cr[mask][T].values.astype(float)
```

- [ ] **Step 3: `statistical_analysis.py` line 704 수정 (`df_cr.loc[non_dm_mask, T]` → `df_cr[non_dm_mask][T]`)**

현재 (line ~704):
```python
                times_g = df_cr.loc[non_dm_mask, T].values.astype(float)
```

수정 후:
```python
                times_g = df_cr[non_dm_mask][T].values.astype(float)
```

- [ ] **Step 4: 전체 테스트 스위트 실행 — 기능 동작 불변 확인**

```bash
/usr/bin/env python3 -m pytest tests/ -q 2>&1 | tail -10
```

Expected: 175+ passed (기존 173 + 신규 2개), 5 pre-existing failures 유지

- [ ] **Step 5: 커밋**

```bash
git add tests/test_stage_j.py statistical_analysis.py
git commit -m "refactor: 주석 정확화 + df_cr[mask] 인덱싱 표현 명확화 (Stage K Task 2)"
```

---

## 완료 기준

- `tests/test_stage_j.py` 총 4개 테스트 PASS (기존 2개 + 신규 2개)
- `statistical_analysis.py` line 686, 704 `loc` 제거
- `tests/test_stage_j.py:25` 주석 업데이트
- `pytest tests/ -q` 신규 실패 없음
