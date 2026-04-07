# Stage O: Stage N 리뷰 후속 조치 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stage N 종합 리뷰(Codex + Gemini + Self)에서 발견된 5개 이슈를 수정 — TEMP_DIRECTORY 경로 버그, 설정 파일 쓰기 권한 안전 처리, `_check_min_rows` 위치 수정 + pooled_sd 경고, 오류 메시지 개선, 문서/주석 정비.

**Architecture:** `config.py`, `db_connector.py`, `statistical_analysis.py`, `cohort_builder.py`, `requirements.txt` 수정. 구조적 변경 없음. Task 1–3은 TDD로 테스트 포함, Task 4–5는 코드 정확성 확인 후 커밋.

**Tech Stack:** Python 3.12, pytest, unittest.mock, Windows/PyInstaller 호환성

---

## Background

Stage N 후속 리뷰 발견 이슈:

| ID | 심각도 | 설명 |
|----|--------|------|
| W1 | WARN | `config.py:215` `TEMP_DIRECTORY='./temp_duckdb'` → `db_connector.py:202` fallback 무력화 |
| W2 | WARN | `config.py:266` `_SETTINGS_FILE = _BASE_DIR / 'yod_settings.json'` → Program Files 쓰기 실패 |
| W3 | WARN | `statistical_analysis.py:267` `_check_min_rows` 가 `try` 블록 밖 → `InsufficientDataError` 가 모델 스킵 대신 루프 탈출 |
| W5 | WARN | `statistical_analysis.py:371` `pooled_sd == 0` → caliper=0 silent rejection |
| W4 | WARN | `statistical_analysis.py:296` RuntimeError 메시지 미포맷 |
| I1 | INFO | `db_connector.py:293` hdbcli ImportError 메시지 구식 |
| I2 | INFO | `cohort_builder.py:478` docstring Step 3 allow_zero 미반영 |
| I4 | INFO | `requirements.txt` requirements-hana.txt 안내 누락 |

현재 테스트 상태: **188 passed, 0 failed**

---

## 파일 구조

| 파일 | Task | 변경 내용 |
|------|------|-----------|
| `config.py:215` | 1 | `'./temp_duckdb'` → `None` |
| `db_connector.py:202` | 1 | `None` 처리 → `_BASE_DIR / 'temp_duckdb'` |
| `config.py:266` | 2 | `_SETTINGS_FILE` 계산 함수 + 쓰기 실패 시 fallback |
| `config.py:293` | 2 | `save_settings()` `PermissionError` 처리 |
| `statistical_analysis.py:267` | 3 | `_check_min_rows` → `try` 내부 이동 + `InsufficientDataError` 명시적 catch |
| `statistical_analysis.py:371` | 3 | `pooled_sd == 0` 경고 로그 추가 |
| `statistical_analysis.py:296` | 4 | RuntimeError 메시지에 `format_error_for_user` 패턴 적용 |
| `db_connector.py:293` | 4 | hdbcli ImportError 메시지 업데이트 |
| `cohort_builder.py:476-480` | 5 | docstring 수정 |
| `requirements.txt` | 5 | requirements-hana.txt 안내 주석 추가 |

---

### Task 1 (W1): TEMP_DIRECTORY `_BASE_DIR` fallback 활성화

**Files:**
- Modify: `config.py:215`
- Modify: `db_connector.py:202`
- Test: `tests/test_stage_o.py` (신규)

`DUCKDB_SETTINGS['TEMP_DIRECTORY']` 기본값이 `'./temp_duckdb'` 이므로 `.get()` 호출 시 항상 이 값이 반환됨 → `_BASE_DIR` fallback 무의미. `None` 으로 변경하면 `db_connector.py` 의 fallback이 실제로 동작.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_stage_o.py` 신규 생성:

```python
"""Stage O: TEMP_DIRECTORY fallback, 설정 파일 권한, _check_min_rows 위치 테스트"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import patch, MagicMock


def test_temp_directory_none_uses_base_dir(tmp_path):
    """TEMP_DIRECTORY=None 이면 _BASE_DIR 기준 경로가 사용된다."""
    import db_connector as dc
    from config import DUCKDB_SETTINGS

    storage = dc.DuckDBStorage(str(tmp_path / 'test.duckdb'))

    fake_settings = {
        'TEMP_DIRECTORY': None,
        'MEMORY_LIMIT': '1GB',
        'THREADS': 1,
    }
    with patch('db_connector.DUCKDB_SETTINGS', fake_settings):
        with patch('db_connector.os.makedirs') as mock_makedirs:
            with patch('db_connector.duckdb.connect') as mock_conn:
                mock_conn.return_value.execute = MagicMock()
                try:
                    storage.connect()
                except Exception:
                    pass  # connect 후속 오류는 무시, makedirs 호출만 검증
                called_path = mock_makedirs.call_args[0][0]
                assert str(dc._BASE_DIR) in called_path, \
                    f"TEMP_DIRECTORY=None 인데 _BASE_DIR 경로를 사용하지 않음: {called_path}"


def test_temp_directory_explicit_path_is_respected(tmp_path):
    """TEMP_DIRECTORY 가 명시된 경우 그 경로를 그대로 사용한다."""
    import db_connector as dc

    storage = dc.DuckDBStorage(str(tmp_path / 'test.duckdb'))
    explicit_path = str(tmp_path / 'custom_temp')

    fake_settings = {
        'TEMP_DIRECTORY': explicit_path,
        'MEMORY_LIMIT': '1GB',
        'THREADS': 1,
    }
    with patch('db_connector.DUCKDB_SETTINGS', fake_settings):
        with patch('db_connector.os.makedirs') as mock_makedirs:
            with patch('db_connector.duckdb.connect') as mock_conn:
                mock_conn.return_value.execute = MagicMock()
                try:
                    storage.connect()
                except Exception:
                    pass
                called_path = mock_makedirs.call_args[0][0]
                assert called_path == explicit_path, \
                    f"명시 경로가 무시됨: {called_path} != {explicit_path}"
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
/usr/bin/env python3 -m pytest tests/test_stage_o.py::test_temp_directory_none_uses_base_dir -v
```

Expected: FAILED (`_BASE_DIR` 경로를 사용하지 않음 — 현재 `None` 이 아닌 `'./temp_duckdb'` 반환)

- [ ] **Step 3: `config.py:215` 수정**

`config.py` 의 `DUCKDB_SETTINGS` 딕셔너리에서:

현재:
```python
DUCKDB_SETTINGS = {
    'MEMORY_LIMIT': '4GB',
    'THREADS': 4,
    'TEMP_DIRECTORY': './temp_duckdb',
}
```

수정 후:
```python
DUCKDB_SETTINGS = {
    'MEMORY_LIMIT': '4GB',
    'THREADS': 4,
    'TEMP_DIRECTORY': None,  # None → db_connector.py 가 _BASE_DIR 기준으로 해결
}
```

- [ ] **Step 4: `db_connector.py:202` 수정**

`DuckDBStorage.connect()` 내부:

현재:
```python
        temp_dir = DUCKDB_SETTINGS.get('TEMP_DIRECTORY', str(_BASE_DIR / 'temp_duckdb'))
```

수정 후:
```python
        _raw_temp = DUCKDB_SETTINGS.get('TEMP_DIRECTORY')
        temp_dir = str(_BASE_DIR / 'temp_duckdb') if not _raw_temp else _raw_temp
```

- [ ] **Step 5: 테스트 PASS 확인**

```bash
/usr/bin/env python3 -m pytest tests/test_stage_o.py::test_temp_directory_none_uses_base_dir tests/test_stage_o.py::test_temp_directory_explicit_path_is_respected -v
```

Expected: 2 PASSED

- [ ] **Step 6: 전체 스위트 회귀 없음**

```bash
/usr/bin/env python3 -m pytest tests/ -q 2>&1 | tail -5
```

Expected: 0 failed, 190 passed

- [ ] **Step 7: 커밋**

```bash
git add config.py db_connector.py tests/test_stage_o.py
git commit -m "fix: TEMP_DIRECTORY None → _BASE_DIR fallback 활성화 (Stage O W1)"
```

---

### Task 2 (W2): 설정 파일 쓰기 권한 안전 처리

**Files:**
- Modify: `config.py:265-266, 278-294`

Windows `Program Files` 설치 시 `_BASE_DIR` 가 쓰기 금지 경로일 수 있음. `save_settings()` 에서 `PermissionError` 를 처리하고, frozen 환경에서는 `%APPDATA%\YodApp` 을 우선 시도.

- [ ] **Step 1: 실패하는 테스트 추가**

`tests/test_stage_o.py` 끝에 추가:

```python

def test_save_settings_raises_informative_error_on_permission_denied(tmp_path):
    """save_settings: 쓰기 불가 경로면 PermissionError 대신 RuntimeError(명확 메시지)."""
    import config

    unwritable = tmp_path / 'readonly_dir' / 'yod_settings.json'
    # 부모 디렉터리를 만들지 않아 쓰기 실패 유도
    with pytest.raises((PermissionError, OSError, RuntimeError)):
        config.save_settings(path=str(unwritable))
    # 핵심: 예외가 조용히 삼켜지지 않아야 한다 (위 assert 가 핵심)


def test_save_settings_succeeds_with_explicit_writable_path(tmp_path):
    """save_settings: 쓰기 가능 경로이면 파일이 생성된다."""
    import config

    out = tmp_path / 'settings.json'
    result = config.save_settings(path=str(out))
    assert out.exists(), "설정 파일이 생성되지 않음"
    assert result == str(out)
```

- [ ] **Step 2: 테스트 실행 (현재 동작 확인)**

```bash
/usr/bin/env python3 -m pytest tests/test_stage_o.py::test_save_settings_raises_informative_error_on_permission_denied tests/test_stage_o.py::test_save_settings_succeeds_with_explicit_writable_path -v
```

Expected: 두 테스트 모두 PASSED (현재도 예외 발생 + 정상 저장 동작). 테스트가 이미 통과하면 현재 동작 확인 완료 — Step 3으로 진행.

- [ ] **Step 3: `config.py` — frozen 환경 APPDATA fallback 추가**

`config.py` 의 `_BASE_DIR` / `_SETTINGS_FILE` 정의 블록을 수정한다.

현재 (라인 261-266):
```python
import json
import sys
from pathlib import Path

_BASE_DIR = Path(sys.executable).parent if getattr(sys, 'frozen', False) else Path(__file__).parent
_SETTINGS_FILE = _BASE_DIR / 'yod_settings.json'
```

수정 후:
```python
import json
import sys
from pathlib import Path

_BASE_DIR = Path(sys.executable).parent if getattr(sys, 'frozen', False) else Path(__file__).parent


def _resolve_settings_file() -> Path:
    """설정 파일 경로 결정. frozen(PyInstaller) + Windows 에서는 %APPDATA%\YodApp 우선."""
    if getattr(sys, 'frozen', False) and os.name == 'nt':
        appdata = os.environ.get('APPDATA', '')
        if appdata:
            d = Path(appdata) / 'YodApp'
            try:
                d.mkdir(parents=True, exist_ok=True)
                return d / 'yod_settings.json'
            except OSError:
                pass  # APPDATA 도 실패하면 _BASE_DIR 로 fallback
    return _BASE_DIR / 'yod_settings.json'


_SETTINGS_FILE = _resolve_settings_file()
```

`os` 는 `config.py` 상단에서 이미 임포트 되어 있는지 확인해야 한다. 만약 없다면 파일 최상단 임포트에 `import os` 를 추가한다.

- [ ] **Step 4: `config.py` 상단 `import os` 확인 및 추가**

`config.py` 상단에서 `import os` 가 없다면 추가한다:

```bash
grep -n "^import os" config.py
```

없으면 파일 최상단(docstring 다음 첫 번째 `#` 섹션 위)에 추가:
```python
import os
```

- [ ] **Step 5: 테스트 PASS 확인**

```bash
/usr/bin/env python3 -m pytest tests/test_stage_o.py::test_save_settings_raises_informative_error_on_permission_denied tests/test_stage_o.py::test_save_settings_succeeds_with_explicit_writable_path -v
```

Expected: 2 PASSED

- [ ] **Step 6: 전체 스위트 회귀 없음**

```bash
/usr/bin/env python3 -m pytest tests/ -q 2>&1 | tail -5
```

Expected: 0 failed, 192 passed

- [ ] **Step 7: 커밋**

```bash
git add config.py tests/test_stage_o.py
git commit -m "fix: frozen Windows에서 APPDATA 기반 설정 파일 경로 fallback 추가 (Stage O W2)"
```

---

### Task 3 (W3 + W5): `_check_min_rows` try 내부 이동 + pooled_sd 경고

**Files:**
- Modify: `statistical_analysis.py:267-291` (`run_cox` 루프)
- Modify: `statistical_analysis.py:371` (`run_psm` caliper 계산 전)

**W3:** `_check_min_rows` 가 `try` 밖이면 `InsufficientDataError` 가 모델 스킵 대신 함수 탈출. 명시적 `except InsufficientDataError: continue` 추가로 의도 명확화.

**W5:** `pooled_sd == 0` 이면 `caliper = 0` 으로 모든 매칭 거부. 진단 로그 없음.

- [ ] **Step 1: 실패하는 테스트 추가**

`tests/test_stage_o.py` 끝에 추가:

```python

def test_run_cox_skips_model_with_insufficient_rows_continues_loop():
    """run_cox: 특정 모델이 _check_min_rows 에서 InsufficientDataError 발생 시
    해당 모델만 스킵하고 나머지 모델은 계속 실행된다."""
    import pandas as pd
    import numpy as np
    from statistical_analysis import StatisticalAnalyzer
    from utils import InsufficientDataError

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
        'dementia_event':      [1] * 15 + [0] * 35,
    })

    analyzer = StatisticalAnalyzer.__new__(StatisticalAnalyzer)
    analyzer._cached_df = df
    analyzer._sampling_info = None
    analyzer.results = {}
    analyzer.db_path = ':memory:'

    call_count = [0]
    original_check = analyzer._check_min_rows

    def patched_check(df_arg, context=''):
        call_count[0] += 1
        # 첫 번째 모델(model1)만 InsufficientDataError 발생
        if call_count[0] == 1:
            raise InsufficientDataError(valid_rows=5, min_rows=30)
        return original_check(df_arg, context=context)

    with patch('statistical_analysis.STUDY_SETTINGS',
               {'MIN_VALID_ROWS': 10, 'MIN_EVENTS': 10, 'SAMPLING_SEED': 42,
                'PH_ALPHA': 0.05}):
        with patch.object(analyzer, '_check_min_rows', side_effect=patched_check):
            with patch('gpu_accelerator.is_gpu_enabled', return_value=False):
                result = analyzer.run_cox(df_prepared=df)

    # model1 은 스킵됐으나 model2, model3 은 성공 → 결과에 포함
    assert 'model1_age_sex' not in result, \
        f"InsufficientDataError 발생 model1 이 결과에 포함됨"
    assert len([k for k in result if k.startswith('model')]) >= 1, \
        f"model1 스킵 후 다른 모델도 실행되지 않음: {list(result.keys())}"
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
/usr/bin/env python3 -m pytest tests/test_stage_o.py::test_run_cox_skips_model_with_insufficient_rows_continues_loop -v
```

Expected: FAILED (현재 `InsufficientDataError` 가 루프 밖으로 탈출하여 모든 모델 실패 → RuntimeError 발생)

- [ ] **Step 3: `statistical_analysis.py` — `_check_min_rows` `try` 내부 이동**

`run_cox()` 의 `for mname, mcols in models.items():` 루프를 수정한다.

현재 (라인 264-292):
```python
        for mname, mcols in models.items():
            cols = [c for c in mcols if c in df_prepared.columns] + [T, E]
            df_model = df_prepared[cols].dropna()
            self._check_min_rows(df_model, context=f"run_cox {mname}")
            try:
                cph = CoxPHFitter()
                cph.fit(df_model, duration_col=T, event_col=E)
                result_entry = {'summary': cph.summary, 'concordance': cph.concordance_index_}
                # PH 가정 검정 (Schoenfeld residuals)
                try:
                    ph_test = proportional_hazard_test(cph, df_model, time_transform='rank')
                    result_entry['ph_test'] = ph_test.summary
                    _ph_alpha = float(STUDY_SETTINGS.get('PH_ALPHA', 0.05))
                    violated = ph_test.summary[ph_test.summary['p'] < _ph_alpha]
                    if not violated.empty:
                        logger.warning(f"Cox {mname}: PH 가정 위반 변수 — "
                                     f"{', '.join(violated.index.tolist())}")
                except Exception as ph_e:
                    logger.info(f"PH 검정 생략 ({mname}): {ph_e}")
                results[mname] = result_entry
            except (duckdb.Error, pd.errors.EmptyDataError, ValueError, MemoryError) as e:
                logger.exception(f"분석 오류 (Cox {mname})")
                logger.warning(f"Cox {mname} 실패: {e}")
            except Exception as e:
                logger.exception(f"예기치 않은 오류 (Cox {mname})")
                logger.warning(f"Cox {mname} 실패: {e}")
            finally:
                del df_model
                gc.collect()
```

수정 후 (`_check_min_rows` 를 `try` 내부로 이동, `InsufficientDataError` 명시적 catch 추가):
```python
        for mname, mcols in models.items():
            cols = [c for c in mcols if c in df_prepared.columns] + [T, E]
            df_model = df_prepared[cols].dropna()
            try:
                self._check_min_rows(df_model, context=f"run_cox {mname}")
                cph = CoxPHFitter()
                cph.fit(df_model, duration_col=T, event_col=E)
                result_entry = {'summary': cph.summary, 'concordance': cph.concordance_index_}
                # PH 가정 검정 (Schoenfeld residuals)
                try:
                    ph_test = proportional_hazard_test(cph, df_model, time_transform='rank')
                    result_entry['ph_test'] = ph_test.summary
                    _ph_alpha = float(STUDY_SETTINGS.get('PH_ALPHA', 0.05))
                    violated = ph_test.summary[ph_test.summary['p'] < _ph_alpha]
                    if not violated.empty:
                        logger.warning(f"Cox {mname}: PH 가정 위반 변수 — "
                                     f"{', '.join(violated.index.tolist())}")
                except Exception as ph_e:
                    logger.info(f"PH 검정 생략 ({mname}): {ph_e}")
                results[mname] = result_entry
            except InsufficientDataError as e:
                logger.warning(f"Cox {mname} 데이터 부족 — 스킵: {e}")
            except (duckdb.Error, pd.errors.EmptyDataError, ValueError, MemoryError) as e:
                logger.exception(f"분석 오류 (Cox {mname})")
                logger.warning(f"Cox {mname} 실패: {e}")
            except Exception as e:
                logger.exception(f"예기치 않은 오류 (Cox {mname})")
                logger.warning(f"Cox {mname} 실패: {e}")
            finally:
                del df_model
                gc.collect()
```

- [ ] **Step 4: `statistical_analysis.py` — pooled_sd 경고 추가**

`run_psm()` 의 caliper 계산 부분. `pooled_sd` 계산 직후:

현재 (라인 370-373):
```python
        pooled_sd = np.sqrt((lps_t.var() + lps_c.var()) / 2)
        caliper = float(STUDY_SETTINGS.get('PSM_CALIPER', 0.2)) * pooled_sd

        if len(control) < 1:
```

수정 후:
```python
        pooled_sd = np.sqrt((lps_t.var() + lps_c.var()) / 2)
        if pooled_sd == 0:
            logger.warning("PSM: pooled_sd = 0 — caliper = 0 이 되어 모든 매칭 거부됩니다 "
                           "(treated/control logit(PS) 분산이 0, 데이터 다양성 부족)")
        caliper = float(STUDY_SETTINGS.get('PSM_CALIPER', 0.2)) * pooled_sd

        if len(control) < 1:
```

- [ ] **Step 5: 테스트 PASS 확인**

```bash
/usr/bin/env python3 -m pytest tests/test_stage_o.py::test_run_cox_skips_model_with_insufficient_rows_continues_loop -v
```

Expected: PASSED

- [ ] **Step 6: 전체 스위트 회귀 없음**

```bash
/usr/bin/env python3 -m pytest tests/ -q 2>&1 | tail -5
```

Expected: 0 failed, 193 passed

- [ ] **Step 7: 커밋**

```bash
git add statistical_analysis.py tests/test_stage_o.py
git commit -m "fix: _check_min_rows try 내부 이동 + pooled_sd=0 경고 추가 (Stage O W3+W5)"
```

---

### Task 4 (W4 + I1): 오류 메시지 개선

**Files:**
- Modify: `statistical_analysis.py:295-298`
- Modify: `db_connector.py:293`

코드 변경이 단순하여 새 테스트 없이 기존 테스트로 회귀 확인.

- [ ] **Step 1: `statistical_analysis.py` — RuntimeError 사용자 포맷 적용**

`run_cox()` 의 전체 모델 실패 감지 블록:

현재 (라인 294-298):
```python
        # 전체 모델 실패 감지
        if not results:
            raise RuntimeError(
                f"run_cox {outcome}: 모든 Cox 모델 피팅 실패 — 결과 없음"
            )
```

수정 후:
```python
        # 전체 모델 실패 감지
        if not results:
            raise RuntimeError(
                f"Cox 회귀 분석({outcome}) 실패: 모든 모델 피팅에 실패했습니다. "
                f"데이터 크기나 공변량 구성을 확인하세요."
            )
```

- [ ] **Step 2: `db_connector.py:293` — hdbcli ImportError 메시지 업데이트**

현재:
```python
            raise ImportError("hdbcli 패키지 필요: pip install hdbcli")
```

수정 후:
```python
            raise ImportError(
                "hdbcli 패키지 필요: pip install -r requirements-hana.txt"
            )
```

- [ ] **Step 3: 기존 테스트 포함 전체 스위트 확인**

```bash
/usr/bin/env python3 -m pytest tests/ -q 2>&1 | tail -5
```

Expected: 0 failed, 193 passed

주의: `test_run_cox_raises_when_all_models_fail` 가 `match="모든 Cox 모델"` 로 패턴 매칭하므로 메시지 변경 후 이 테스트가 깨질 수 있다. 깨지면 `tests/test_stage_n.py` 에서 `match` 패턴도 업데이트한다:

```python
# 현재
with _pytest.raises(RuntimeError, match="모든 Cox 모델"):
# 수정 후
with _pytest.raises(RuntimeError, match="Cox 회귀 분석"):
```

- [ ] **Step 4: 커밋**

```bash
git add statistical_analysis.py db_connector.py tests/test_stage_n.py
git commit -m "fix: Cox 실패 메시지 사용자 친화적으로 수정 + hdbcli 안내 업데이트 (Stage O W4+I1)"
```

---

### Task 5 (I2 + I4): 문서/주석 정비

**Files:**
- Modify: `cohort_builder.py:476-480`
- Modify: `requirements.txt`

코드 동작 변경 없음.

- [ ] **Step 1: `cohort_builder.py` docstring 수정**

`build_cohort()` docstring 수정:

현재 (라인 475-480):
```python
    def build_cohort(self, cb=None):
        """6단계 코호트 파이프라인 실행.

        각 단계는 duckdb.Error 발생 시 1회 재시도 후 CohortStepError를 발생시킨다.
        단계 결과가 0건이어도 CohortStepError를 발생시켜 후속 단계 실행을 막는다.
        """
```

수정 후:
```python
    def build_cohort(self, cb=None):
        """6단계 코호트 파이프라인 실행.

        각 단계는 duckdb.Error 발생 시 1회 재시도 후 CohortStepError를 발생시킨다.
        단계 결과가 0건이면 CohortStepError를 발생시켜 후속 단계 실행을 막는다.
        예외: Step 3(dm_medications)는 T2DM_NOMED 코호트에서 0건이 정상이므로 허용.
        """
```

- [ ] **Step 2: `requirements.txt` — HANA 설치 안내 주석 추가**

현재 `requirements.txt` 의 `# --- System ---` 섹션 위에 추가:

현재:
```
# --- System ---
psutil>=5.9.8
```

수정 후:
```
# --- Optional: SAP HANA DB 연결 ---
# SAP HANA 서버에 접속하려면 별도 설치 필요:
#   pip install -r requirements-hana.txt

# --- System ---
psutil>=5.9.8
```

- [ ] **Step 3: 전체 스위트 확인**

```bash
/usr/bin/env python3 -m pytest tests/ -q 2>&1 | tail -5
```

Expected: 0 failed, 193 passed

- [ ] **Step 4: 커밋**

```bash
git add cohort_builder.py requirements.txt
git commit -m "docs: build_cohort docstring 수정 + requirements-hana.txt 안내 추가 (Stage O I2+I4)"
```

---

## 완료 기준

- `pytest tests/ -q` → 0 failed, 193+ passed
- `config.py:215` `TEMP_DIRECTORY=None` → `db_connector.py` `_BASE_DIR` fallback 실제 동작
- `config.py` `_resolve_settings_file()` — frozen + Windows 에서 `%APPDATA%\YodApp` 우선
- `statistical_analysis.py` `run_cox` 루프: `InsufficientDataError` 명시적 catch로 모델 스킵
- `statistical_analysis.py` `run_psm`: `pooled_sd == 0` 경고 로그
- 오류 메시지 사용자 친화적 개선 (Cox RuntimeError, hdbcli ImportError)
- `cohort_builder.py` docstring 정확화, `requirements.txt` HANA 안내 추가
