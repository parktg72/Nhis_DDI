# Stage M: CIF 테스트 완전성 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stage L 코드 리뷰어 제안 3개 구현 — `n_competing` 단언 강화, `vad_event` 경쟁위험 테스트, `ad_event` threshold boundary 테스트.

**Architecture:** `tests/test_stage_j.py` 만 수정. 구현 코드 변경 없음. Task 1은 기존 테스트 강화, Task 2·3은 신규 테스트 추가.

**Tech Stack:** Python 3.12, pytest, pandas, unittest.mock

---

## Background

`run_competing_risks()` 결과 구조 (`statistical_analysis.py:760-769`):
```python
results[outcome] = {
    'cif_by_group': cif_by_group,
    'fine_gray_summary': fg_summary,
    'n_event':     int((event_type == 1).sum()),   # 관심사건 총 수
    'n_competing': int((event_type == 2).sum()),   # 경쟁위험 총 수 (전체 코호트)
    'n_censored':  int((event_type == 0).sum()),   # 검열 총 수
}
```

Stage L에서 추가된 `test_cif_ad_event_other_dementia_classified_as_competing_risk` 테스트는 `n_competing` 단언이 없어 `other_dementia` 분류 건수를 직접 검증하지 않음.

---

## 파일 구조

| 파일 | 변경 내용 |
|------|-----------|
| `tests/test_stage_j.py` | Task 1: 기존 테스트 `n_competing` 단언 추가; Task 2: `vad_event` 경쟁위험 테스트 신규; Task 3: `ad_event` threshold boundary 테스트 신규 |

---

### Task 1: `test_cif_ad_event_other_dementia_classified_as_competing_risk` 에 `n_competing` 단언 추가

**Files:**
- Modify: `tests/test_stage_j.py` (line ~229-246)

현재 테스트 끝 부분:
```python
    cif_competing = cif['T2DM_OHA'].get('cif_competing', [])
    assert any(v > 0 for v in cif_competing), \
        (f"ad_event CIF T2DM_OHA 의 cif_competing 이 모두 0 — "
         f"other_dementia 경쟁위험 미분류 의심: {cif_competing}")
```

- [ ] **Step 1: 현재 테스트 끝 부분 확인**

```bash
/usr/bin/env python3 -m pytest tests/test_stage_j.py::test_cif_ad_event_other_dementia_classified_as_competing_risk -v
```

Expected: PASSED

- [ ] **Step 2: `n_competing` 단언 추가**

`tests/test_stage_j.py` 에서 `test_cif_ad_event_other_dementia_classified_as_competing_risk` 함수 내 기존 단언 블록을 찾아 아래와 같이 수정한다.

현재 (파일 끝부분):
```python
    cif_competing = cif['T2DM_OHA'].get('cif_competing', [])
    assert any(v > 0 for v in cif_competing), \
        (f"ad_event CIF T2DM_OHA 의 cif_competing 이 모두 0 — "
         f"other_dementia 경쟁위험 미분류 의심: {cif_competing}")
```

수정 후 (단언 2개 추가):
```python
    # n_competing 는 전체 코호트의 경쟁위험 건수 (other_dementia 3건)
    n_competing = result.get('ad_event', {}).get('n_competing')
    assert n_competing == 3, \
        f"other_dementia 경쟁위험 분류 건수가 3 이어야 함: {n_competing}"
    cif_competing = cif['T2DM_OHA'].get('cif_competing', [])
    assert any(v > 0 for v in cif_competing), \
        (f"ad_event CIF T2DM_OHA 의 cif_competing 이 모두 0 — "
         f"other_dementia 경쟁위험 미분류 의심: {cif_competing}")
```

- [ ] **Step 3: 테스트 PASS 확인**

```bash
/usr/bin/env python3 -m pytest tests/test_stage_j.py::test_cif_ad_event_other_dementia_classified_as_competing_risk -v
```

Expected: PASSED

- [ ] **Step 4: 커밋**

```bash
git add tests/test_stage_j.py
git commit -m "test: n_competing 단언 추가 — other_dementia 분류 건수 명시적 검증 (Stage M Task 1)"
```

---

### Task 2: `vad_event` 경쟁위험 분류 테스트 추가

**Files:**
- Modify: `tests/test_stage_j.py` (파일 끝에 1개 테스트 추가)

- [ ] **Step 1: 테스트 파일 끝 확인**

```bash
/usr/bin/env python3 -m pytest tests/test_stage_j.py -v 2>&1 | tail -10
```

Expected: 6 PASSED

- [ ] **Step 2: `vad_event` 테스트 추가**

`tests/test_stage_j.py` 파일 끝에 다음을 추가한다:

```python

def test_cif_vad_event_other_dementia_classified_as_competing_risk():
    """vad_event 경로에서 non-VaD 치매(dementia=1, vad=0)가 경쟁위험으로 분류된다.

    T2DM_OHA: 25행
      - 4건: vad_event=1 (관심사건, event_type=1)
      - 3건: dementia_event=1 AND vad_event=0 (other_dementia 경쟁위험, event_type=2)
      - 나머지: 검열 (event_type=0)
    n_competing == 3, cif_competing > 0 이어야 한다.
    """
    n = 40
    # 행 0-14: T1DM (15행, 이벤트 없음 — skip)
    # 행 15-39: T2DM_OHA (25행)
    #   행 15-18: vad_event=1 (4건)
    #   행 19-21: dementia_event=1, vad_event=0 (other_dementia 3건)
    #   행 22-39: 이벤트 없음
    vad_events  = [0] * 15 + [1] * 4 + [0] * 21
    dem_events  = [0] * 15 + [0] * 4 + [1] * 3 + [0] * 18  # non-VaD 치매 3건
    df = pd.DataFrame({
        'follow_up_years': [1.0] * n,
        'vad_event':         vad_events,
        'dementia_event':    dem_events,
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
    vad_result = result.get('vad_event', {})
    cif = vad_result.get('cif_by_group', {})
    assert 'T2DM_OHA' in cif, \
        f"T2DM_OHA 가 vad_event CIF 에서 누락됨: {list(cif.keys())}"
    n_competing = vad_result.get('n_competing')
    assert n_competing == 3, \
        f"vad_event other_dementia 경쟁위험 분류 건수가 3 이어야 함: {n_competing}"
    cif_competing = cif['T2DM_OHA'].get('cif_competing', [])
    assert any(v > 0 for v in cif_competing), \
        (f"vad_event CIF T2DM_OHA 의 cif_competing 이 모두 0 — "
         f"other_dementia 경쟁위험 미분류 의심: {cif_competing}")
```

- [ ] **Step 3: 테스트 PASS 확인**

```bash
/usr/bin/env python3 -m pytest tests/test_stage_j.py::test_cif_vad_event_other_dementia_classified_as_competing_risk -v
```

Expected: PASSED

- [ ] **Step 4: 커밋**

```bash
git add tests/test_stage_j.py
git commit -m "test: vad_event other_dementia 경쟁위험 분류 테스트 추가 (Stage M Task 2)"
```

---

### Task 3: `ad_event` threshold boundary 테스트 추가

**Files:**
- Modify: `tests/test_stage_j.py` (파일 끝에 1개 테스트 추가)

- [ ] **Step 1: 테스트 파일 끝 확인**

```bash
/usr/bin/env python3 -m pytest tests/test_stage_j.py -v 2>&1 | tail -10
```

Expected: 7 PASSED

- [ ] **Step 2: `ad_event` threshold boundary 테스트 추가**

`tests/test_stage_j.py` 파일 끝에 다음을 추가한다:

```python

def test_cif_ad_event_respects_min_subgroup_events_threshold():
    """ad_event 경로에서 MIN_SUBGROUP_EVENTS threshold crossing 을 검증한다.

    T2DM_OHA: 25행, 4 AD 이벤트
    MIN_SUBGROUP_EVENTS=3 → CIF 포함 (4 >= 3)
    MIN_SUBGROUP_EVENTS=5 → CIF skip  (4 < 5)
    """
    n = 40
    # 행 0-14: T1DM (15행, 이벤트 없음)
    # 행 15-39: T2DM_OHA (25행, AD 이벤트 4건)
    df = pd.DataFrame({
        'follow_up_years': [1.0] * n,
        'ad_event': [0] * 15 + [1] * 4 + [0] * 21,  # T1DM=0건, T2DM_OHA=4건
        'dementia_event': [0] * n,
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

    cif_runs = result_runs.get('ad_event', {}).get('cif_by_group', {})
    cif_skips = result_skips.get('ad_event', {}).get('cif_by_group', {})
    assert 'T2DM_OHA' in cif_runs, \
        f"MIN_SUBGROUP_EVENTS=3 인데 AD 이벤트 4건 T2DM_OHA 가 ad_event CIF 에서 누락됨: {list(cif_runs.keys())}"
    assert 'T2DM_OHA' not in cif_skips, \
        f"MIN_SUBGROUP_EVENTS=5 인데 AD 이벤트 4건 T2DM_OHA 가 ad_event CIF 에 포함됨: {list(cif_skips.keys())}"
```

- [ ] **Step 3: 테스트 PASS 확인**

```bash
/usr/bin/env python3 -m pytest tests/test_stage_j.py::test_cif_ad_event_respects_min_subgroup_events_threshold -v
```

Expected: PASSED

- [ ] **Step 4: 전체 스위트 확인**

```bash
/usr/bin/env python3 -m pytest tests/ -q 2>&1 | tail -8
```

Expected: 180 passed, 5 pre-existing failures 유지

- [ ] **Step 5: 커밋**

```bash
git add tests/test_stage_j.py
git commit -m "test: ad_event threshold boundary 테스트 추가 (Stage M Task 3)"
```

---

## 완료 기준

- `tests/test_stage_j.py` 총 8개 테스트 PASS (기존 6개 + 신규 2개 + 강화 1개)
- `pytest tests/ -q` 180 passed, 5 pre-existing failures 유지
- `n_competing` 단언으로 other_dementia 분류 건수 명시적 검증
- `vad_event` 경쟁위험 경로 커버됨
- `ad_event` threshold boundary 검증됨
