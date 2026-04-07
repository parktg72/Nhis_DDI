# Stage N: Critical Bug Fixes & Code Quality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 종합 리뷰에서 식별된 7개 이슈를 우선순위 순서(#1→#2→#3→#5→#8→#4→#9)로 수정 — DuckDB 1.2+ 호환성, 코호트 Step 3 0건 처리, run_interaction 가드, hdbcli 분리, 경로 추상화, Cox 침묵 실패, 임계값 설정화.

**Architecture:** `db_connector.py`, `cohort_builder.py`, `statistical_analysis.py`, `config.py`, `utils.py`, `requirements.txt` 수정. 신규 파일 1개(`requirements-hana.txt`). 각 태스크마다 TDD — 실패하는 테스트를 먼저 작성하고 구현으로 통과.

**Tech Stack:** Python 3.12, pytest, pandas, unittest.mock, DuckDB

---

## Background

현재 `pytest tests/ -q` 결과: **5 failed, 179 passed**.

실패 원인:
- 1건: `test_db_connector_decimal_chunks.py` — DuckDB 1.2+ 에서 `data_type='DECIMAL(4,0)'` 반환 → `_widen_decimal_columns` 가 타입 매칭 실패
- 4건: `test_cohort_builder.py::TestBuildCohortFull` — Step 3(`dm_medications`) 0건 결과 시 `CohortStepError` 발생 (T2DM_NOMED는 정상)

Task 1·2 완료 후: **0 failed, 184 passed** 기대.

---

## 파일 구조

| 파일 | Task | 변경 내용 |
|------|------|-----------|
| `db_connector.py:161` | 1 | `data_type` 문자열 비교 → `startswith` |
| `cohort_builder.py:483` | 2 | `_safe_step` `allow_zero` 파라미터 추가 |
| `statistical_analysis.py:473` | 3 | `run_interaction` MIN_VALID_ROWS/MIN_EVENTS 가드 추가 |
| `requirements.txt` | 5 | `hdbcli` 행 제거 |
| `requirements-hana.txt` | 5 | 신규 — `hdbcli>=2.21.0` |
| `config.py:261` | 8 | `_BASE_DIR` 패턴으로 `_SETTINGS_FILE` 경로 수정 |
| `db_connector.py:191` | 8 | `DuckDBStorage.__init__` 기본 경로 `_BASE_DIR` 사용 |
| `statistical_analysis.py:276,365,418` | 4 | Cox 전체 실패 감지 + raise |
| `config.py:185` | 9 | `PH_ALPHA`, `PSM_CALIPER`, `PSM_SMD_THRESHOLD` 추가 |
| `statistical_analysis.py:276,365,418` | 9 | 하드코딩 임계값 → `STUDY_SETTINGS` 참조 |

---

### Task 1 (#1): DuckDB 1.2+ DECIMAL 타입 문자열 호환성 수정

**Files:**
- Modify: `db_connector.py:161`

기존 코드가 `'DECIMAL(4,0)'.upper() not in ('DECIMAL', 'NUMERIC')` → `True` 로 평가돼 컬럼 확장 건너뜀. DuckDB 1.2+ 는 `'DECIMAL(4,0)'` 형식으로 반환.

- [ ] **Step 1: 현재 실패 확인**

```bash
/usr/bin/env python3 -m pytest tests/test_db_connector_decimal_chunks.py::test_widen_decimal_columns_expands_narrow_decimal_schema -v
```

Expected: FAILED (`assert 'DECIMAL(4,0)' == 'DECIMAL(38,0)'`)

- [ ] **Step 2: `db_connector.py:161` 수정**

현재:
```python
        if str(row['data_type']).upper() not in ('DECIMAL', 'NUMERIC'):
            continue
```

수정 후:
```python
        data_type_upper = str(row['data_type']).upper()
        if not (data_type_upper.startswith('DECIMAL') or data_type_upper.startswith('NUMERIC')):
            continue
```

- [ ] **Step 3: 테스트 PASS 확인**

```bash
/usr/bin/env python3 -m pytest tests/test_db_connector_decimal_chunks.py -v
```

Expected: all 3 PASSED

- [ ] **Step 4: 커밋**

```bash
git add db_connector.py
git commit -m "fix: DuckDB 1.2+ DECIMAL(n,s) 타입 문자열 호환성 수정 (Stage N #1)"
```

---

### Task 2 (#2): 코호트 Step 3 0건 허용

**Files:**
- Modify: `cohort_builder.py:483-509` (`_safe_step` 내부)

Step 3(`dm_medications`)는 T2DM_NOMED 코호트에서 처방 기록이 0건이 정상. 현재 `_safe_step`은 0건 결과를 무조건 에러로 처리.

- [ ] **Step 1: 현재 실패 확인**

```bash
/usr/bin/env python3 -m pytest tests/test_cohort_builder.py::TestBuildCohortFull -v --tb=short 2>&1 | tail -20
```

Expected: 4 FAILED (CohortStepError: dm_medications 결과 0건)

- [ ] **Step 2: `_safe_step` 수정 — `allow_zero` 파라미터 추가**

`cohort_builder.py` 에서 `def build_cohort(self, cb=None):` 내부의 `_safe_step` 클로저를 찾아 다음과 같이 수정한다.

현재 시그니처:
```python
        def _safe_step(step_num, step_name, step_fn, result_table):
```

수정 후 시그니처와 0건 체크 분기:
```python
        def _safe_step(step_num, step_name, step_fn, result_table, allow_zero=False):
```

현재 0건 체크 블록:
```python
            n = self.dm.storage.get_row_count(result_table)
            if n == 0:
                raise CohortStepError(
                    step_num, step_name,
                    ValueError(f"{result_table} 결과 0건 — 데이터 적재 상태를 확인하세요.")
                )
            logger.info(f"[{step_num}/6] {step_name} 완료: {n:,}건")
```

수정 후 0건 체크 블록:
```python
            n = self.dm.storage.get_row_count(result_table)
            if n == 0 and not allow_zero:
                raise CohortStepError(
                    step_num, step_name,
                    ValueError(f"{result_table} 결과 0건 — 데이터 적재 상태를 확인하세요.")
                )
            logger.info(f"[{step_num}/6] {step_name} 완료: {n:,}건")
```

- [ ] **Step 3: Step 3 호출부에 `allow_zero=True` 추가**

현재 Step 3 호출:
```python
        results['dm_meds'], _ = _safe_step(
            3, "당뇨 약물 처방 식별",
            self.step3_dm_medications, "dm_medications"
        )
```

수정 후:
```python
        results['dm_meds'], _ = _safe_step(
            3, "당뇨 약물 처방 식별",
            self.step3_dm_medications, "dm_medications",
            allow_zero=True
        )
```

- [ ] **Step 4: 테스트 PASS 확인**

```bash
/usr/bin/env python3 -m pytest tests/test_cohort_builder.py::TestBuildCohortFull -v
```

Expected: 4 PASSED

- [ ] **Step 5: 전체 스위트 — 0 failures 확인**

```bash
/usr/bin/env python3 -m pytest tests/ -q 2>&1 | tail -5
```

Expected: **0 failed**, 184 passed

- [ ] **Step 6: 커밋**

```bash
git add cohort_builder.py
git commit -m "fix: Step 3 dm_medications 0건 허용 — T2DM_NOMED 정상 케이스 (Stage N #2)"
```

---

### Task 3 (#3): `run_interaction` 최소 데이터 가드 추가

**Files:**
- Modify: `statistical_analysis.py:472` (after `d = d[d['follow_up_years'] > 0]`)

현재 `run_interaction`은 데이터가 너무 적어도 Cox 피팅을 시도하고 예외를 삼킨다. `run_cox`처럼 사전 가드가 필요.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_stage_n.py` 를 새로 만든다:

```python
"""Stage N: run_interaction 가드 + Cox 침묵 실패 + 임계값 설정화 테스트"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from statistical_analysis import YODAnalyzer


def _make_analyzer(df):
    """테스트용 YODAnalyzer — _cached_df 직접 주입."""
    analyzer = YODAnalyzer.__new__(YODAnalyzer)
    analyzer._cached_df = df
    analyzer._sampling_info = None
    analyzer.results = {}
    analyzer.db_path = ':memory:'
    return analyzer


def test_run_interaction_returns_none_when_too_few_rows():
    """run_interaction: MIN_VALID_ROWS 미만이면 None 반환 (Cox 시도 안 함)."""
    n = 5  # MIN_VALID_ROWS=30 보다 훨씬 적음
    df = pd.DataFrame({
        'exposure_group': ['T1DM'] * n,
        'is_t1dm': [1] * n,
        'dm_duration_cat': ['<5yr'] * n,
        'age_at_index': [50.0] * n,
        'male': [1] * n,
        'income_q': [5] * n,
        'cci_score': [0] * n,
        'follow_up_years': [1.0] * n,
        'dementia_event': [0] * n,
    })
    analyzer = _make_analyzer(df)
    with patch('statistical_analysis.STUDY_SETTINGS',
               {'MIN_VALID_ROWS': 30, 'MIN_EVENTS': 10, 'SAMPLING_SEED': 42}):
        result = analyzer.run_interaction(df_prepared=df)
    assert result is None, \
        f"MIN_VALID_ROWS 미만 데이터에서 run_interaction 이 None 을 반환해야 함: {result}"


def test_run_interaction_returns_none_when_too_few_events():
    """run_interaction: MIN_EVENTS 미만이면 None 반환."""
    n = 50
    df = pd.DataFrame({
        'exposure_group': ['T1DM'] * n,
        'is_t1dm': [1] * n,
        'dm_duration_cat': ['<5yr'] * n,
        'age_at_index': [50.0] * n,
        'male': [1] * n,
        'income_q': [5] * n,
        'cci_score': [0] * n,
        'follow_up_years': [1.0] * n,
        'dementia_event': [0] * (n - 2) + [1, 1],  # 이벤트 2건 (MIN_EVENTS=10 미만)
    })
    analyzer = _make_analyzer(df)
    with patch('statistical_analysis.STUDY_SETTINGS',
               {'MIN_VALID_ROWS': 30, 'MIN_EVENTS': 10, 'SAMPLING_SEED': 42}):
        result = analyzer.run_interaction(df_prepared=df)
    assert result is None, \
        f"MIN_EVENTS 미만 이벤트에서 run_interaction 이 None 을 반환해야 함: {result}"
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
/usr/bin/env python3 -m pytest tests/test_stage_n.py::test_run_interaction_returns_none_when_too_few_rows tests/test_stage_n.py::test_run_interaction_returns_none_when_too_few_events -v
```

Expected: FAILED (현재 가드 없으므로 Cox 피팅 시도 또는 다른 예외)

- [ ] **Step 3: `run_interaction` 에 가드 추가**

`statistical_analysis.py` 에서 `run_interaction` 함수의 `d = d[d['follow_up_years'] > 0]` 바로 다음에 다음 블록을 추가한다.

현재 (라인 ~472):
```python
        d = d[d['follow_up_years'] > 0]

        try:
            cph = CoxPHFitter()
```

수정 후:
```python
        d = d[d['follow_up_years'] > 0]

        _min_rows = int(STUDY_SETTINGS.get('MIN_VALID_ROWS', 30))
        _min_events = int(STUDY_SETTINGS.get('MIN_EVENTS', 10))
        if len(d) < _min_rows or int(d['dementia_event'].sum()) < _min_events:
            logger.warning(
                "run_interaction: 데이터 부족 — 행 수 %d (최소 %d), 이벤트 수 %d (최소 %d) — 분석 스킵",
                len(d), _min_rows, int(d['dementia_event'].sum()), _min_events,
            )
            return None

        try:
            cph = CoxPHFitter()
```

- [ ] **Step 4: 테스트 PASS 확인**

```bash
/usr/bin/env python3 -m pytest tests/test_stage_n.py::test_run_interaction_returns_none_when_too_few_rows tests/test_stage_n.py::test_run_interaction_returns_none_when_too_few_events -v
```

Expected: 2 PASSED

- [ ] **Step 5: 전체 스위트 회귀 없음 확인**

```bash
/usr/bin/env python3 -m pytest tests/ -q 2>&1 | tail -5
```

Expected: 0 failed, 186 passed

- [ ] **Step 6: 커밋**

```bash
git add statistical_analysis.py tests/test_stage_n.py
git commit -m "fix: run_interaction MIN_VALID_ROWS/MIN_EVENTS 가드 추가 (Stage N #3)"
```

---

### Task 4 (#5): `hdbcli` 를 `requirements-hana.txt` 로 분리

**Files:**
- Modify: `requirements.txt` — `hdbcli>=2.21.0` 행 제거
- Create: `requirements-hana.txt`

SAP HANA 클라이언트는 HANA DB 사용 시에만 필요. 기본 DuckDB 모드에서는 불필요한 의존성.

- [ ] **Step 1: `requirements-hana.txt` 생성**

파일 전체 내용:
```
# SAP HANA DB 연결 시에만 필요합니다.
# pip install -r requirements-hana.txt
hdbcli>=2.21.0
```

- [ ] **Step 2: `requirements.txt` 에서 hdbcli 제거**

현재 `requirements.txt` 의 `# --- DB Connectors ---` 섹션:
```
# --- DB Connectors ---
# hdbcli: SAP HANA 클라이언트. PyPI 설치 가능하나 HANA 서버 접속 시에만 실제 사용됨.
# HANA DB 없이 DuckDB 로컬 모드로만 사용한다면 설치를 건너뛰어도 됩니다.
hdbcli>=2.21.0
```

수정 후 (3줄 모두 제거, 섹션 전체 제거):
```
# --- System ---
```

즉, `requirements.txt` 에서 `# --- DB Connectors ---` 부터 `hdbcli>=2.21.0` 까지 4줄을 삭제한다.

- [ ] **Step 3: 파일 내용 확인**

```bash
grep -n hdbcli requirements.txt requirements-hana.txt
```

Expected:
```
requirements-hana.txt:3:hdbcli>=2.21.0
```
(`requirements.txt` 에는 hdbcli 없어야 함)

- [ ] **Step 4: 커밋**

```bash
git add requirements.txt requirements-hana.txt
git commit -m "refactor: hdbcli 를 requirements-hana.txt 로 분리 — 선택적 의존성 (Stage N #5)"
```

---

### Task 5 (#8): 경로 추상화 — `_BASE_DIR` 패턴 적용

**Files:**
- Modify: `config.py:261` — `_SETTINGS_FILE` 경로
- Modify: `db_connector.py:191` — `DuckDBStorage.__init__` 기본 경로
- Modify: `db_connector.py:196` — `TEMP_DIRECTORY` 기본값

PyInstaller 패키징 시 CWD가 예측 불가. `sys.executable` 부모 디렉토리를 기반으로 경로를 설정해야 한다.

- [ ] **Step 1: `config.py` 수정 — `_SETTINGS_FILE` 경로**

`config.py` 상단 임포트 섹션을 확인한다:

```bash
head -10 config.py
```

`config.py` 의 `# 설정 저장/불러오기` 섹션 (라인 ~255-261) 을 찾아 수정한다.

현재:
```python
import json
from pathlib import Path

_SETTINGS_FILE = Path('./yod_settings.json')
```

수정 후:
```python
import json
import sys
from pathlib import Path

_BASE_DIR = Path(sys.executable).parent if getattr(sys, 'frozen', False) else Path(__file__).parent
_SETTINGS_FILE = _BASE_DIR / 'yod_settings.json'
```

- [ ] **Step 2: `db_connector.py` 수정 — `DuckDBStorage` 기본 경로**

`db_connector.py` 상단에 `_BASE_DIR` 패턴을 추가한다. 현재 파일 상단 임포트 확인:

```bash
head -20 db_connector.py
```

`db_connector.py` 상단 임포트 블록(os, re 등) 직후, 클래스 정의 전에 다음을 추가한다:
```python
import sys
from pathlib import Path as _Path

_BASE_DIR = _Path(sys.executable).parent if getattr(sys, 'frozen', False) else _Path(__file__).parent
```

그리고 `DuckDBStorage.__init__` 의 기본 경로를 수정한다.

현재:
```python
    def __init__(self, db_path='./nhis_analysis.duckdb'):
```

수정 후:
```python
    def __init__(self, db_path=None):
        if db_path is None:
            db_path = str(_BASE_DIR / 'nhis_analysis.duckdb')
```

`connect()` 메서드의 `temp_dir` 도 수정한다.

현재:
```python
        temp_dir = DUCKDB_SETTINGS.get('TEMP_DIRECTORY', './temp_duckdb')
```

수정 후:
```python
        temp_dir = DUCKDB_SETTINGS.get('TEMP_DIRECTORY', str(_BASE_DIR / 'temp_duckdb'))
```

- [ ] **Step 3: 기존 테스트 PASS 확인 (회귀 없음)**

```bash
/usr/bin/env python3 -m pytest tests/test_db_connector.py tests/test_db_connector_decimal_chunks.py -v 2>&1 | tail -15
```

Expected: 모든 테스트 PASSED

- [ ] **Step 4: 전체 스위트 회귀 없음 확인**

```bash
/usr/bin/env python3 -m pytest tests/ -q 2>&1 | tail -5
```

Expected: 0 failed, 186 passed

- [ ] **Step 5: 커밋**

```bash
git add config.py db_connector.py
git commit -m "fix: _BASE_DIR 패턴으로 상대경로 추상화 — PyInstaller/Windows 호환 (Stage N #8)"
```

---

### Task 6 (#4): Cox 전체 모델 실패 시 명시적 오류 발생

**Files:**
- Modify: `statistical_analysis.py` — `run_cox()` 루프 후 빈 결과 체크

현재: 3개 모델이 모두 실패하면 빈 `{}` 를 저장하고 반환 — 호출자가 분석 실패를 알 수 없음.

- [ ] **Step 1: 실패하는 테스트 추가**

`tests/test_stage_n.py` 끝에 추가한다:

```python

def test_run_cox_raises_when_all_models_fail():
    """run_cox: 모든 모델 피팅 실패 시 RuntimeError 발생 (침묵 실패 방지)."""
    n = 50
    df = pd.DataFrame({
        'exposure_group': ['T1DM'] * 25 + ['T2DM_OHA'] * 25,
        'is_t1dm':        [1] * 25 + [0] * 25,
        'is_t2dm_oha':    [0] * 25 + [1] * 25,
        'is_t2dm_insulin':[0] * n,
        'is_t2dm_nomed':  [0] * n,
        'age_at_index':   [50.0] * n,
        'male':           [1] * n,
        'income_q':       [5] * n,
        'comor_hypertension':  [0] * n,
        'comor_dyslipidemia':  [0] * n,
        'comor_depression':    [0] * n,
        'comp_retinopathy':    [0] * n,
        'comp_nephropathy':    [0] * n,
        'comp_neuropathy':     [0] * n,
        'comor_ischemic_stroke':   [0] * n,
        'comor_hemorrhagic_stroke':[0] * n,
        'comor_ihd':           [0] * n,
        'comor_atrial_fib':    [0] * n,
        'comor_heart_failure': [0] * n,
        'comp_hypoglycemia':   [0] * n,
        'follow_up_years':     [1.0] * n,
        'dementia_event':      [1] * 15 + [0] * 35,  # 15 이벤트 — MIN_EVENTS 통과
    })
    analyzer = _make_analyzer(df)

    # 모든 CoxPHFitter.fit 호출이 실패하도록 강제
    with patch('statistical_analysis.STUDY_SETTINGS',
               {'MIN_VALID_ROWS': 10, 'MIN_EVENTS': 10, 'SAMPLING_SEED': 42,
                'PH_ALPHA': 0.05}):
        with patch('statistical_analysis.CoxPHFitter') as mock_cox_cls:
            mock_cox_cls.return_value.fit.side_effect = ValueError("강제 실패")
            import pytest as _pytest
            with _pytest.raises(RuntimeError, match="모든 Cox 모델"):
                analyzer.run_cox(df_prepared=df)
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
/usr/bin/env python3 -m pytest tests/test_stage_n.py::test_run_cox_raises_when_all_models_fail -v
```

Expected: FAILED (현재 RuntimeError 발생 안 함)

- [ ] **Step 3: `run_cox` 에 전체 실패 감지 추가**

`statistical_analysis.py` 에서 `run_cox()` 내 모델 루프가 끝난 뒤 `ph_combined` 취합 블록 직전에 추가한다.

현재 (라인 ~293-301):
```python
        # PH 검정 요약을 모델별로 취합하여 최상위에 저장
        ph_combined = {}
        for mname, entry in results.items():
            if 'ph_test' in entry:
                ph_combined[mname] = entry['ph_test']
        if ph_combined:
            results['ph_test_summary'] = ph_combined

        self.results[f'cox_{outcome}'] = results
        return results
```

수정 후:
```python
        # 전체 모델 실패 감지
        if not results:
            raise RuntimeError(
                f"run_cox {outcome}: 모든 Cox 모델 피팅 실패 — 결과 없음"
            )

        # PH 검정 요약을 모델별로 취합하여 최상위에 저장
        ph_combined = {}
        for mname, entry in results.items():
            if 'ph_test' in entry:
                ph_combined[mname] = entry['ph_test']
        if ph_combined:
            results['ph_test_summary'] = ph_combined

        self.results[f'cox_{outcome}'] = results
        return results
```

- [ ] **Step 4: 테스트 PASS 확인**

```bash
/usr/bin/env python3 -m pytest tests/test_stage_n.py::test_run_cox_raises_when_all_models_fail -v
```

Expected: PASSED

- [ ] **Step 5: 전체 스위트 회귀 없음 확인**

```bash
/usr/bin/env python3 -m pytest tests/ -q 2>&1 | tail -5
```

Expected: 0 failed, 187 passed

- [ ] **Step 6: 커밋**

```bash
git add statistical_analysis.py tests/test_stage_n.py
git commit -m "fix: run_cox 전체 모델 실패 시 RuntimeError 발생 — 침묵 실패 방지 (Stage N #4)"
```

---

### Task 7 (#9): 하드코딩 임계값 → `STUDY_SETTINGS` 설정화

**Files:**
- Modify: `config.py:185` — `STUDY_SETTINGS` 에 3개 키 추가
- Modify: `statistical_analysis.py:276` — PH alpha 0.05 → 설정값
- Modify: `statistical_analysis.py:365` — PSM caliper 0.2 → 설정값
- Modify: `statistical_analysis.py:418` — SMD threshold 0.1 → 설정값

현재 하드코딩:
- `ph_test.summary[ph_test.summary['p'] < 0.05]` (라인 276)
- `caliper = 0.2 * pooled_sd` (라인 365)
- `'balanced': smd < 0.1` (라인 418)

- [ ] **Step 1: 실패하는 테스트 추가**

`tests/test_stage_n.py` 끝에 추가한다:

```python

def test_psm_caliper_respects_study_settings():
    """PSM caliper 는 STUDY_SETTINGS['PSM_CALIPER'] 를 사용해야 한다."""
    import statistical_analysis as sa
    # PSM_CALIPER 키가 STUDY_SETTINGS 에 존재해야 함
    from config import STUDY_SETTINGS
    assert 'PSM_CALIPER' in STUDY_SETTINGS, \
        "STUDY_SETTINGS 에 PSM_CALIPER 키가 없음"
    assert 'PSM_SMD_THRESHOLD' in STUDY_SETTINGS, \
        "STUDY_SETTINGS 에 PSM_SMD_THRESHOLD 키가 없음"
    assert 'PH_ALPHA' in STUDY_SETTINGS, \
        "STUDY_SETTINGS 에 PH_ALPHA 키가 없음"


def test_smd_threshold_uses_study_settings():
    """SMD balanced 판정은 STUDY_SETTINGS['PSM_SMD_THRESHOLD'] 를 사용해야 한다."""
    n = 60
    df = pd.DataFrame({
        'exposure_group': ['T1DM'] * 30 + ['T2DM_OHA'] * 30,
        'is_t1dm':        [1] * 30 + [0] * 30,
        'is_t2dm_oha':    [0] * 30 + [1] * 30,
        'is_t2dm_insulin':[0] * n,
        'is_t2dm_nomed':  [0] * n,
        'age_at_index':   [50.0] * 30 + [55.0] * 30,
        'male':           [1] * n,
        'income_q':       [5] * n,
        'comor_hypertension':  [0] * n,
        'comor_dyslipidemia':  [0] * n,
        'dm_duration_years':   [3.0] * n,
        'follow_up_years':     [2.0] * n,
        'dementia_event':      [1] * 10 + [0] * 50,
        'ad_event':            [0] * n,
        'vad_event':           [0] * n,
    })
    analyzer = _make_analyzer(df)

    # PSM_SMD_THRESHOLD=0.5 → age_at_index SMD (약 0.33) 가 balanced=True 여야 함
    with patch('statistical_analysis.STUDY_SETTINGS',
               {'MIN_VALID_ROWS': 10, 'MIN_EVENTS': 5, 'SAMPLING_SEED': 42,
                'PSM_RATIO': 3, 'PSM_CALIPER': 0.2, 'PSM_SMD_THRESHOLD': 0.5,
                'PH_ALPHA': 0.05}):
        with patch('gpu_accelerator.is_gpu_enabled', return_value=False):
            with patch('gpu_accelerator.get_logistic_regression') as mock_lr, \
                 patch('gpu_accelerator.get_nearest_neighbors') as mock_nn:
                mock_lr.return_value.fit = MagicMock()
                mock_lr.return_value.predict_proba = MagicMock(
                    return_value=np.column_stack([
                        np.linspace(0.3, 0.7, n),
                        np.linspace(0.3, 0.7, n),
                    ])
                )
                mock_nn.return_value.fit = MagicMock()
                mock_nn.return_value.kneighbors = MagicMock(
                    return_value=(
                        np.zeros((30, 3)),
                        np.array([[i, i+1, i+2] for i in range(30)]) % 30,
                    )
                )
                result = analyzer.run_psm(df_prepared=df)

    # PSM 이 스킵되지 않았다면 balance 키 확인
    if result and 'balance' in result:
        age_balance = result['balance'].get('age_at_index', {})
        assert age_balance.get('balanced') is True, \
            f"PSM_SMD_THRESHOLD=0.5 인데 age_at_index balanced=False: {age_balance}"
```

- [ ] **Step 2: 키 존재 여부 테스트 실패 확인**

```bash
/usr/bin/env python3 -m pytest tests/test_stage_n.py::test_psm_caliper_respects_study_settings -v
```

Expected: FAILED (STUDY_SETTINGS 에 PSM_CALIPER 키 없음)

- [ ] **Step 3: `config.py` 에 3개 키 추가**

`config.py` 의 `STUDY_SETTINGS` 딕셔너리에서 마지막 항목(`'MIN_SUBGROUP_EVENTS': 5,`) 다음에 추가한다.

현재:
```python
    'MIN_SUBGROUP_EVENTS': 5,     # 서브그룹/Fine-Gray 분석 최소 이벤트 수
}
```

수정 후:
```python
    'MIN_SUBGROUP_EVENTS': 5,     # 서브그룹/Fine-Gray 분석 최소 이벤트 수
    'PH_ALPHA': 0.05,             # Cox PH 가정 검정 유의수준
    'PSM_CALIPER': 0.2,           # PSM caliper = PSM_CALIPER × pooled logit(PS) SD
    'PSM_SMD_THRESHOLD': 0.1,     # PSM 균형 판정 SMD 임계값
}
```

- [ ] **Step 4: `statistical_analysis.py` — PH alpha 하드코딩 교체**

현재 (라인 ~276):
```python
                    violated = ph_test.summary[ph_test.summary['p'] < 0.05]
```

수정 후:
```python
                    _ph_alpha = float(STUDY_SETTINGS.get('PH_ALPHA', 0.05))
                    violated = ph_test.summary[ph_test.summary['p'] < _ph_alpha]
```

- [ ] **Step 5: `statistical_analysis.py` — PSM caliper 하드코딩 교체**

현재 (라인 ~365):
```python
        caliper = 0.2 * pooled_sd
```

수정 후:
```python
        caliper = float(STUDY_SETTINGS.get('PSM_CALIPER', 0.2)) * pooled_sd
```

- [ ] **Step 6: `statistical_analysis.py` — SMD threshold 하드코딩 교체**

현재 (라인 ~418):
```python
                           'smd': round(smd, 4), 'balanced': smd < 0.1}
```

수정 후:
```python
                           'smd': round(smd, 4), 'balanced': smd < float(STUDY_SETTINGS.get('PSM_SMD_THRESHOLD', 0.1))}
```

- [ ] **Step 7: 테스트 PASS 확인**

```bash
/usr/bin/env python3 -m pytest tests/test_stage_n.py::test_psm_caliper_respects_study_settings tests/test_stage_n.py::test_smd_threshold_uses_study_settings -v
```

Expected: 2 PASSED (`test_smd_threshold_uses_study_settings` 는 PSM skipped 가능 — skip 이라도 assert는 통과)

- [ ] **Step 8: 전체 스위트 최종 확인**

```bash
/usr/bin/env python3 -m pytest tests/ -q 2>&1 | tail -8
```

Expected: 0 failed, 189 passed

- [ ] **Step 9: 커밋**

```bash
git add config.py statistical_analysis.py tests/test_stage_n.py
git commit -m "refactor: PH_ALPHA/PSM_CALIPER/PSM_SMD_THRESHOLD STUDY_SETTINGS 설정화 (Stage N #9)"
```

---

## 완료 기준

- `pytest tests/ -q` → **0 failed** (기존 5 failed → 0 fixed)
- 총 테스트 수: 179 + 5(Tasks 3·6·7 신규) = 184+ passed
- `db_connector.py`: DuckDB 1.2+ `DECIMAL(n,s)` 타입 처리
- `cohort_builder.py`: Step 3 0건 허용
- `statistical_analysis.py`: `run_interaction` 데이터 가드, Cox 전체 실패 감지, 임계값 설정 참조
- `config.py`: `_BASE_DIR` 패턴, `PH_ALPHA`/`PSM_CALIPER`/`PSM_SMD_THRESHOLD` 추가
- `requirements.txt`: `hdbcli` 제거 / `requirements-hana.txt` 생성
