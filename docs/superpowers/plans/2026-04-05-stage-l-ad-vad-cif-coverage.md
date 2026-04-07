# Stage L: ad_event/vad_event CIF 가드 경로 테스트 커버리지 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stage K 리뷰(Codex INFO #7)에서 지적된 `ad_event`/`vad_event` CIF 경로 미커버리지를 해소 — 이벤트 가드 동작 및 `other_dementia` 경쟁위험 분류 경로를 검증하는 테스트 추가.

**Architecture:** `tests/test_stage_j.py`에 2개 테스트를 추가한다. 구현 코드 변경 없음 — 순수 테스트 커버리지 확장.

**Tech Stack:** Python 3.12, pytest, pandas, unittest.mock

---

## Background: `ad_event`/`vad_event` 경로의 차별점

`run_competing_risks()` 의 `for outcome in ['dementia_event', 'ad_event', 'vad_event']` 루프에서 `ad_event`/`vad_event` 는 추가 경쟁위험 분류 로직이 있다:

```python
# statistical_analysis.py:668-672
if outcome in ('ad_event', 'vad_event') and 'dementia_event' in df_cr.columns:
    # 비대상 치매 유형도 경쟁위험으로 분류
    other_dementia = ((df_cr['dementia_event'].values == 1) &
                      (df_cr[outcome].values == 0))
    competing_mask = competing_mask | other_dementia
```

현재 `tests/test_stage_j.py` 의 4개 테스트는 모두 `dementia_event` 경로만 검증. `ad_event` 경로의 CIF 이벤트 가드와 `other_dementia` 경쟁위험 분류가 테스트되지 않음.

---

## 파일 구조

| 파일 | 변경 내용 |
|------|-----------|
| `tests/test_stage_j.py` | `ad_event` CIF 가드 테스트 2개 추가 (파일 끝에 append) |

---

### Task 1: `ad_event` CIF 이벤트 가드 + `other_dementia` 경쟁위험 분류 테스트

**Files:**
- Modify: `tests/test_stage_j.py` (파일 끝에 2개 테스트 추가)

- [ ] **Step 1: 현재 테스트 현황 확인**

```bash
/usr/bin/env python3 -m pytest tests/test_stage_j.py -v 2>&1 | tail -10
```

Expected: 4 passed (기존 dementia_event 경로 테스트)

- [ ] **Step 2: `tests/test_stage_j.py` 끝에 2개 테스트 추가**

파일 끝(현재 마지막 줄 다음)에 다음을 추가한다:

```python

def test_cif_ad_event_skips_group_with_insufficient_events():
    """ad_event 경로에서도 CIF per-group 이벤트 가드가 동일하게 적용된다.

    T1DM: 15행, 0 AD 이벤트 → MIN_SUBGROUP_EVENTS=3 → CIF skip 되어야 함
    T2DM_OHA: 25행, 5 AD 이벤트 → CIF 포함되어야 함
    dementia_event 는 모두 0 → other_dementia 경쟁위험 없음 (순수 이벤트 가드 테스트).
    """
    n = 40
    # 행 0-14: T1DM (15행, AD 이벤트 0건)
    # 행 15-39: T2DM_OHA (25행, AD 이벤트 5건)
    df = pd.DataFrame({
        'follow_up_years': [1.0] * n,
        'ad_event': [0] * 15 + [1] * 5 + [0] * 20,  # T1DM=0건, T2DM_OHA=5건
        'dementia_event': [0] * n,                   # non-AD 치매 없음 → other_dementia=0
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
    cif = result.get('ad_event', {}).get('cif_by_group', {})
    assert 'T1DM' not in cif, \
        f"AD 이벤트 0건 T1DM 이 ad_event CIF 에 포함됨 — 이벤트 가드 미적용: {list(cif.keys())}"
    assert 'T2DM_OHA' in cif, \
        f"AD 이벤트 5건 T2DM_OHA 가 ad_event CIF 에서 누락됨: {list(cif.keys())}"


def test_cif_ad_event_other_dementia_classified_as_competing_risk():
    """ad_event 경로에서 non-AD 치매(dementia=1, ad=0)가 경쟁위험(event_type=2)으로 분류된다.

    T2DM_OHA: 25행
      - 5건: ad_event=1 (관심사건, event_type=1)
      - 3건: dementia_event=1 AND ad_event=0 (other_dementia 경쟁위험, event_type=2)
      - 나머지: 검열 (event_type=0)
    경쟁위험 분류가 올바르면 CIF 결과에 'cif_competing' 값이 양수여야 한다.
    """
    n = 40
    # 행 0-14: T1DM (15행, 이벤트 없음 — MIN_SUBGROUP_EVENTS=3 이므로 skip)
    # 행 15-39: T2DM_OHA (25행)
    #   행 15-19: ad_event=1 (5건)
    #   행 20-22: dementia_event=1, ad_event=0 (other_dementia 3건)
    #   행 23-39: 이벤트 없음
    ad_events   = [0] * 15 + [1] * 5 + [0] * 20
    dem_events  = [0] * 15 + [0] * 5 + [1] * 3 + [0] * 17  # non-AD 치매 3건
    df = pd.DataFrame({
        'follow_up_years': [1.0] * n,
        'ad_event':         ad_events,
        'dementia_event':   dem_events,
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
    ad_result = result.get('ad_event', {})
    cif = ad_result.get('cif_by_group', {})
    assert 'T2DM_OHA' in cif, \
        f"T2DM_OHA 가 ad_event CIF 에서 누락됨: {list(cif.keys())}"
    # other_dementia 경쟁위험이 분류됐으면 cif_competing 에 양수 값이 있어야 함
    cif_competing = cif['T2DM_OHA'].get('cif_competing', [])
    assert any(v > 0 for v in cif_competing), \
        (f"ad_event CIF T2DM_OHA 의 cif_competing 이 모두 0 — "
         f"other_dementia 경쟁위험 미분류 의심: {cif_competing}")
```

- [ ] **Step 3: 새 테스트 2개가 PASS 하는지 확인**

```bash
/usr/bin/env python3 -m pytest tests/test_stage_j.py -v 2>&1 | tail -12
```

Expected: 6 passed (기존 4개 + 신규 2개)

- [ ] **Step 4: 전체 스위트 실행 — 회귀 없는지 확인**

```bash
/usr/bin/env python3 -m pytest tests/ -q 2>&1 | tail -8
```

Expected: 177+ passed, 5 pre-existing failures 유지

- [ ] **Step 5: 커밋**

```bash
git add tests/test_stage_j.py
git commit -m "test: ad_event CIF 가드 및 other_dementia 경쟁위험 분류 커버리지 추가 (Stage L)"
```

---

## 완료 기준

- `tests/test_stage_j.py` 총 6개 테스트 PASS (기존 4개 + 신규 2개)
- `pytest tests/ -q` 신규 실패 없음
- `ad_event` 경로의 이벤트 가드 동작과 `other_dementia` 경쟁위험 분류가 모두 검증됨
