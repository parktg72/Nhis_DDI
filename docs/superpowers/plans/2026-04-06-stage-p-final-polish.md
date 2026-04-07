# Stage P: Final Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stage O 최종 리뷰에서 제안된 2개 개선사항 구현 — `pooled_sd` NaN 가드 추가, `test_stage_o.py` W5/I1 행동 커버리지 테스트 추가.

**Architecture:** `statistical_analysis.py` 1줄 수정, `tests/test_stage_o.py` 테스트 2개 추가. 구현 코드 변경은 최소.

**Tech Stack:** Python 3.12, pytest, unittest.mock, numpy

---

## Background

Stage O 최종 리뷰(reviewer) 제안:

1. **Important**: `statistical_analysis.py:375` — `pooled_sd == 0` 가드가 NaN 을 커버하지 않음. `pd.Series.var()` 는 단일 요소 시리즈에서 `NaN` 반환 → `pooled_sd = NaN` → `caliper = NaN` → 모든 매칭 조건 `d <= NaN` 이 `False` → 조용히 전체 매칭 거부. 경고 없이 넘어감.
   - Fix: `if pooled_sd == 0 or np.isnan(pooled_sd):`

2. **Suggestion**: `tests/test_stage_o.py` 에 W5(pooled_sd 경고 로그), I1(hdbcli ImportError 메시지) 행동 커버리지 테스트 추가.

현재 테스트 상태: **193 passed, 0 failed**

---

## 파일 구조

| 파일 | 변경 내용 |
|------|-----------|
| `statistical_analysis.py:375` | `pooled_sd == 0` → `pooled_sd == 0 or np.isnan(pooled_sd)` |
| `tests/test_stage_o.py` | 테스트 2개 추가 (W5 경고 로그, I1 ImportError 메시지) |

---

### Task 1: `pooled_sd` NaN 가드 + 경고 로그 커버리지 테스트

**Files:**
- Modify: `statistical_analysis.py:375`
- Modify: `tests/test_stage_o.py` (끝에 추가)

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_stage_o.py` 끝에 추가:

```python

def test_psm_warns_when_pooled_sd_is_nan(caplog):
    """pooled_sd 가 NaN 이면 (단일 요소 treated/control) 경고 로그가 발생한다."""
    import logging
    import pandas as pd
    import numpy as np
    from statistical_analysis import StatisticalAnalyzer
    from unittest.mock import patch, MagicMock

    # treated=1명, control=1명 → var() = NaN → pooled_sd = NaN
    n = 2
    df = pd.DataFrame({
        'exposure_group': ['T1DM', 'T2DM_OHA'],
        'is_t1dm':        [1, 0],
        'is_t2dm_oha':    [0, 1],
        'is_t2dm_insulin':[0, 0],
        'is_t2dm_nomed':  [0, 0],
        'age_at_index':   [50.0, 55.0],
        'male':           [1, 1],
        'income_q':       [5, 5],
        'comor_hypertension':  [0, 0],
        'comor_dyslipidemia':  [0, 0],
        'dm_duration_years':   [3.0, 3.0],
        'follow_up_years':     [1.0, 1.0],
        'dementia_event':      [1, 0],
        'ad_event':            [0, 0],
        'vad_event':           [0, 0],
    })
    analyzer = StatisticalAnalyzer.__new__(StatisticalAnalyzer)
    analyzer._cached_df = df
    analyzer._sampling_info = None
    analyzer.results = {}
    analyzer.db_path = ':memory:'

    mock_lr = MagicMock()
    mock_lr.fit = MagicMock()
    # PS: treated=0.9, control=0.1 → logit(0.9)≈2.2, logit(0.1)≈-2.2
    # 각각 단일 값 → var() = NaN
    mock_lr.predict_proba = MagicMock(return_value=np.array([[0.1, 0.9], [0.9, 0.1]]))

    with patch('statistical_analysis.STUDY_SETTINGS',
               {'MIN_VALID_ROWS': 1, 'MIN_EVENTS': 1, 'SAMPLING_SEED': 42,
                'PSM_RATIO': 1, 'PSM_CALIPER': 0.2, 'PSM_SMD_THRESHOLD': 0.1,
                'PH_ALPHA': 0.05}):
        with patch('gpu_accelerator.is_gpu_enabled', return_value=False):
            with patch('gpu_accelerator.get_logistic_regression', return_value=mock_lr):
                with patch('gpu_accelerator.get_nearest_neighbors') as mock_nn_cls:
                    mock_nn_cls.return_value.fit = MagicMock()
                    mock_nn_cls.return_value.kneighbors = MagicMock(
                        return_value=(np.array([[1.0]]), np.array([[0]]))
                    )
                    with caplog.at_level(logging.WARNING, logger='statistical_analysis'):
                        analyzer.run_psm(df_prepared=df)

    assert any('pooled_sd' in msg for msg in caplog.messages), \
        f"pooled_sd NaN/0 경고가 로그에 없음. 로그: {caplog.messages}"
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
/usr/bin/env python3 -m pytest tests/test_stage_o.py::test_psm_warns_when_pooled_sd_is_nan -v 2>&1 | tail -15
```

Expected: FAILED (현재 `pooled_sd == 0` 만 체크 — NaN 은 통과)

- [ ] **Step 3: `statistical_analysis.py:375` 수정**

현재:
```python
        if pooled_sd == 0:
            logger.warning("PSM: pooled_sd = 0 — caliper = 0 이 되어 모든 매칭 거부됩니다 "
                           "(treated/control logit(PS) 분산이 0, 데이터 다양성 부족)")
```

수정 후:
```python
        if pooled_sd == 0 or np.isnan(pooled_sd):
            logger.warning("PSM: pooled_sd = 0 또는 NaN — caliper 가 무효화되어 모든 매칭 거부됩니다 "
                           "(treated/control logit(PS) 분산 부족, 데이터 다양성 확인 필요)")
```

- [ ] **Step 4: 테스트 PASS 확인**

```bash
/usr/bin/env python3 -m pytest tests/test_stage_o.py::test_psm_warns_when_pooled_sd_is_nan -v
```

Expected: PASSED

- [ ] **Step 5: 전체 스위트 회귀 없음**

```bash
/usr/bin/env python3 -m pytest tests/ -q 2>&1 | tail -5
```

Expected: 0 failed, 194 passed

- [ ] **Step 6: 커밋**

```bash
git add statistical_analysis.py tests/test_stage_o.py
git commit -m "fix: pooled_sd NaN 가드 추가 + 경고 로그 커버리지 테스트 (Stage P)"
```

---

### Task 2: hdbcli ImportError 메시지 행동 커버리지 테스트

**Files:**
- Modify: `tests/test_stage_o.py` (끝에 추가)

`db_connector.py:293-295` — hdbcli ImportError 메시지가 `requirements-hana.txt` 를 안내하는지 행동으로 검증.

- [ ] **Step 1: 테스트 작성**

`tests/test_stage_o.py` 끝에 추가:

```python

def test_hana_connect_importerror_mentions_requirements_hana(monkeypatch):
    """hdbcli 미설치 시 ImportError 메시지에 requirements-hana.txt 가 포함된다."""
    import builtins
    import importlib
    import db_connector

    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == 'hdbcli':
            raise ImportError("No module named 'hdbcli'")
        return original_import(name, *args, **kwargs)

    storage = db_connector.HANAStorage(
        host='localhost', port=39015, user='test', password='test'
    )

    monkeypatch.setattr(builtins, '__import__', mock_import)
    with pytest.raises(ImportError) as exc_info:
        storage.connect()

    assert 'requirements-hana.txt' in str(exc_info.value), \
        f"ImportError 메시지에 requirements-hana.txt 가 없음: {exc_info.value}"
```

- [ ] **Step 2: 테스트 실행**

```bash
/usr/bin/env python3 -m pytest tests/test_stage_o.py::test_hana_connect_importerror_mentions_requirements_hana -v
```

Expected: PASSED (I1 수정이 Stage O에서 이미 완료됨)

- [ ] **Step 3: 전체 스위트 최종 확인**

```bash
/usr/bin/env python3 -m pytest tests/ -q 2>&1 | tail -5
```

Expected: 0 failed, 195 passed

- [ ] **Step 4: 커밋**

```bash
git add tests/test_stage_o.py
git commit -m "test: hdbcli ImportError 메시지 requirements-hana.txt 안내 커버리지 (Stage P)"
```

---

## 완료 기준

- `pytest tests/ -q` → 0 failed, 195 passed
- `statistical_analysis.py:375`: `pooled_sd == 0 or np.isnan(pooled_sd)` 조건
- `test_psm_warns_when_pooled_sd_is_nan`: NaN 케이스에서 경고 로그 검증
- `test_hana_connect_importerror_mentions_requirements_hana`: hdbcli 메시지 행동 검증
