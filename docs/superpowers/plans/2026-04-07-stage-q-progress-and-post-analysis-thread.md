# Stage Q: Progress Callback 완성 & Post-Analysis Thread 분리 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** "메시지 창에 변화가 없는 경우" 버그를 해소한다 — `_load_data`/`_prepare` 구간 progress_callback 누락 수정, `run_cox`/`run_competing_risks` 중간 진행 emit 추가, `_on_analysis` 내 `run_post_analysis` 메인 스레드 블로킹을 WorkerThread로 분리, `_run_step` 중복 로직 정리, `pywin32` 의존성 플랫폼 명시.

**Architecture:** Task 1(분석 시작 단계 progress emit) → Task 2(분석 중간 단계 emit 보강) → Task 3(_on_analysis WorkerThread 분리) → Task 4(_run_step 주석 정리) → Task 5(requirements.txt 플랫폼 명시) 순으로 구현. 모든 변경은 기존 시그니처 하위호환을 유지한다 (`cb=None` 기본값).

**Tech Stack:** Python 3.12, PyQt5 (QThread, pyqtSignal), DuckDB, pandas, lifelines, pytest, unittest.mock

---

## 파일 변경 맵

| 파일 | 변경 유형 | 내용 |
|------|-----------|------|
| `statistical_analysis.py` | Modify | `_load_data(cb=None)`, `_prepare(df, cb=None)`, `run_cox` 모델별 emit, `run_competing_risks` outcome/group emit |
| `tabs.py` | Modify | `_on_analysis` → post_worker 생성; `_on_post_analysis` 신규 슬롯 |
| `cohort_builder.py` | Modify | `_run_step` docstring 보완 (dead code 오해 방지) |
| `requirements.txt` | Modify | `pywin32` Windows 전용 주석 추가 |
| `tests/test_stage_q.py` | Create | Task 1~2 커버리지 테스트 |

---

## Task 1: `_load_data` + `_prepare` progress emit 추가

**Files:**
- Modify: `statistical_analysis.py:67-160` (`_load_data`)
- Modify: `statistical_analysis.py:180-230` (`_prepare`)
- Modify: `statistical_analysis.py:921-926` (`run_selected` — cb 전달)
- Create: `tests/test_stage_q.py`

### 배경

`run_selected`에서 `_load_data()` + `_prepare()` 호출 구간(라인 925-926) 동안 cb가 전혀 emit되지 않는다. 대용량 데이터에서 이 두 단계는 수십 초가 소요될 수 있어 메시지 창이 완전히 멈춘 것처럼 보인다.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_stage_q.py` 신규 파일:

```python
"""tests/test_stage_q.py — Stage Q: progress emit 커버리지"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from statistical_analysis import StatisticalAnalyzer


def _make_dm(total_rows=10):
    """최소 DataManager mock — _load_data가 동작할 수 있도록."""
    dm = MagicMock()
    dm.storage.get_row_count.return_value = total_rows
    # 비샘플링 경로: valid rows 반환
    sample_df = pd.DataFrame({
        'exposure_group': ['NON_DM'] * total_rows,
        'follow_up_days': [365] * total_rows,
        'follow_up_years': [1.0] * total_rows,
        'age_at_index': [55.0] * total_rows,
        'SEX_TYPE': ['1'] * total_rows,
    })
    dm.query.return_value = sample_df
    return dm


def test_load_data_emits_start_message():
    """_load_data(cb=...) 가 '분석 데이터 로딩 중...' 을 emit 해야 한다."""
    dm = _make_dm()
    analyzer = StatisticalAnalyzer(dm)
    messages = []
    analyzer._load_data(cb=messages.append)
    assert any("분석 데이터 로딩 중" in m for m in messages), \
        f"'분석 데이터 로딩 중' 메시지 없음. 실제 메시지: {messages}"


def test_load_data_emits_sampling_message():
    """샘플링 분기에서 '샘플링' 관련 메시지를 emit 해야 한다."""
    dm = _make_dm(total_rows=10)
    # get_safe_analysis_rows 를 5로 제한 → 샘플링 분기 진입
    with patch('statistical_analysis.mem_manager') as mock_mm:
        mock_mm.get_safe_analysis_rows.return_value = 5
        mock_mm.optimize_dtypes.side_effect = lambda df: df
        group_df = pd.DataFrame({'exposure_group': ['NON_DM'], 'cnt': [10]})
        dm.query.return_value = group_df  # group_counts 쿼리
        dm.execute.return_value = None
        # 실제 샘플 쿼리도 mock
        sample_df = pd.DataFrame({
            'exposure_group': ['NON_DM'] * 5,
            'follow_up_days': [365] * 5,
            'follow_up_years': [1.0] * 5,
        })
        dm.query.side_effect = [group_df, sample_df]
        dm.storage.get_row_count.return_value = 10

        analyzer = StatisticalAnalyzer(dm)
        messages = []
        try:
            analyzer._load_data(cb=messages.append)
        except Exception:
            pass  # 샘플링 경로 진입 여부만 확인
    assert any("샘플링" in m for m in messages), \
        f"샘플링 메시지 없음. 실제 메시지: {messages}"


def test_prepare_emits_progress_message():
    """_prepare(df, cb=...) 가 '전처리' 관련 메시지를 emit 해야 한다."""
    dm = _make_dm()
    analyzer = StatisticalAnalyzer(dm)
    df = pd.DataFrame({
        'exposure_group': ['NON_DM'] * 5,
        'SEX_TYPE': ['1'] * 5,
        'follow_up_years': [1.0] * 5,
        'age_at_index': [55.0] * 5,
    })
    messages = []
    analyzer._prepare(df, cb=messages.append)
    assert any("전처리" in m for m in messages), \
        f"'전처리' 메시지 없음. 실제 메시지: {messages}"


def test_run_selected_passes_cb_to_load_and_prepare(monkeypatch):
    """run_selected 가 cb 를 _load_data 와 _prepare 에 전달해야 한다."""
    dm = _make_dm()
    analyzer = StatisticalAnalyzer(dm)

    load_cb_received = []
    prepare_cb_received = []

    original_load = analyzer._load_data
    original_prepare = analyzer._prepare

    def patched_load(cb=None):
        load_cb_received.append(cb)
        raise pd.errors.EmptyDataError("테스트 중단")  # run_selected 중간 탈출

    monkeypatch.setattr(analyzer, '_load_data', patched_load)

    cb = MagicMock()
    try:
        analyzer.run_selected(cb=cb)
    except pd.errors.EmptyDataError:
        pass

    assert load_cb_received[0] is cb, "_load_data 에 cb 가 전달되지 않음"
```

- [ ] **Step 2: 테스트 실행 — FAIL 확인**

```bash
cd /Users/aidept/ptg_at_train/yod_diabetes_app
python -m pytest tests/test_stage_q.py -v 2>&1 | tail -20
```

기대: 4개 테스트 모두 FAIL (AttributeError 또는 AssertionError)

- [ ] **Step 3: `_load_data` 시그니처에 `cb=None` 추가 및 emit 삽입**

`statistical_analysis.py` 라인 67을 아래와 같이 수정:

```python
    def _load_data(self, cb=None):
        """메모리 안전 데이터 로드 — 1회 로드 후 캐시 재사용"""
        if self._cached_df is not None:
            return self._cached_df, self._sampling_info

        if cb: cb("분석 데이터 로딩 중...")
        min_valid = int(STUDY_SETTINGS.get('MIN_VALID_ROWS', 30))
```

그리고 샘플링 분기 진입 직후(라인 78, `logger.warning` 다음 줄)에 emit 추가:

```python
            logger.warning(f"분석 데이터 {total:,}건 > 안전 한도 {max_rows:,}건 → 층화 샘플링")
            if cb: cb(f"층화 샘플링 적용 중... ({total:,}건 → {max_rows:,}건 목표)")
```

`_cached_df` 로드 완료 직전 (라인 157, `mem_manager.optimize_dtypes` 바로 뒤):

```python
        # dtype 최적화
        self._cached_df = mem_manager.optimize_dtypes(self._cached_df)
        if cb: cb(f"데이터 로드 완료: {len(self._cached_df):,}건")
```

- [ ] **Step 4: `_prepare` 시그니처에 `cb=None` 추가 및 emit 삽입**

`statistical_analysis.py` 라인 180:

```python
    def _prepare(self, df, cb=None):
        """공변량 전처리 — 캐시 원본 보호를 위해 1회 copy 후 파생변수 추가"""
        if cb: cb("데이터 전처리 중...")
        prepared = df.copy()
```

- [ ] **Step 5: `run_selected`에서 cb를 `_load_data`, `_prepare`에 전달**

`statistical_analysis.py` 라인 925-926:

```python
        raw, info = self._load_data(cb=cb)
        df_prepared = self._prepare(raw, cb=cb)
```

- [ ] **Step 6: 테스트 실행 — PASS 확인**

```bash
python -m pytest tests/test_stage_q.py -v 2>&1 | tail -10
```

기대: 4개 PASSED

- [ ] **Step 7: 기존 테스트 회귀 확인**

```bash
python -m pytest tests/ -q --tb=no 2>&1 | tail -5
```

기대: 전체 PASSED (0 failed)

- [ ] **Step 8: 커밋**

```bash
git add statistical_analysis.py tests/test_stage_q.py
git commit -m "feat: _load_data/_prepare cb emit 추가 — 분석 시작 구간 메시지 창 무변화 수정 (Stage Q T1)"
```

---

## Task 2: `run_cox` + `run_competing_risks` 중간 진행 emit 추가

**Files:**
- Modify: `statistical_analysis.py:264-294` (`run_cox` 모델 루프)
- Modify: `statistical_analysis.py:668-735` (`run_competing_risks` outcome/group 루프)
- Modify: `tests/test_stage_q.py` (테스트 추가)

### 배경

`run_cox`는 시작 시 1회만 emit하고 3개 모델 피팅 동안 추가 emit 없다. `run_competing_risks`도 시작 1회 후 3 outcome × 5 group CIF 계산 동안 완전 무음이다.

- [ ] **Step 1: 실패하는 테스트 추가 (`tests/test_stage_q.py` 끝에 추가)**

```python
def test_run_cox_emits_per_model_progress():
    """run_cox 가 각 모델(model1/2/3) 피팅 전 메시지를 emit 해야 한다."""
    dm = _make_dm()
    analyzer = StatisticalAnalyzer(dm)

    # Cox 실행에 필요한 최소 DataFrame
    df = pd.DataFrame({
        'exposure_group': ['NON_DM'] * 40,
        'is_t1dm': [0] * 40, 'is_t2dm_oha': [0] * 40,
        'is_t2dm_insulin': [0] * 40, 'is_t2dm_nomed': [0] * 40,
        'age_at_index': [55.0] * 40,
        'male': [1] * 40,
        'follow_up_years': [1.0] * 40,
        'dementia_event': [0] * 35 + [1] * 5,
    })
    messages = []
    try:
        analyzer.run_cox('dementia_event', cb=messages.append, df_prepared=df)
    except Exception:
        pass
    model_messages = [m for m in messages if 'model' in m.lower() or '모델' in m.lower()]
    assert len(model_messages) >= 3, \
        f"모델별 진행 메시지 3개 미만. 실제: {messages}"


def test_run_competing_risks_emits_per_outcome_progress():
    """run_competing_risks 가 각 outcome 시작 시 메시지를 emit 해야 한다."""
    dm = _make_dm()
    analyzer = StatisticalAnalyzer(dm)

    df = pd.DataFrame({
        'exposure_group': ['NON_DM'] * 40,
        'is_t1dm': [0] * 40, 'is_t2dm_oha': [0] * 40,
        'is_t2dm_insulin': [0] * 40, 'is_t2dm_nomed': [0] * 40,
        'age_at_index': [55.0] * 40, 'male': [1] * 40,
        'follow_up_years': [1.0] * 40,
        'dementia_event': [0] * 35 + [1] * 5,
        'ad_event': [0] * 38 + [1] * 2,
        'vad_event': [0] * 39 + [1] * 1,
        'competing_death_event': [0] * 36 + [1] * 4,
    })
    messages = []
    try:
        analyzer.run_competing_risks(cb=messages.append, df_prepared=df)
    except Exception:
        pass
    # 각 outcome 시작 메시지 확인
    assert any('dementia_event' in m for m in messages), f"dementia_event 메시지 없음: {messages}"
    assert any('ad_event' in m for m in messages), f"ad_event 메시지 없음: {messages}"
```

- [ ] **Step 2: 테스트 실행 — FAIL 확인**

```bash
python -m pytest tests/test_stage_q.py::test_run_cox_emits_per_model_progress tests/test_stage_q.py::test_run_competing_risks_emits_per_outcome_progress -v 2>&1 | tail -10
```

기대: 2개 FAIL (AssertionError)

- [ ] **Step 3: `run_cox` 모델별 emit 추가**

`statistical_analysis.py` 라인 264, `for mname, mcols in models.items():` 루프 내부 첫 줄에 추가:

```python
        for mname, mcols in models.items():
            if cb: cb(f"Cox 회귀 ({outcome}) — {mname} 피팅 중...")
            cols = [c for c in mcols if c in df_prepared.columns] + [T, E]
```

- [ ] **Step 4: `run_competing_risks` outcome별 emit 추가**

`statistical_analysis.py` 라인 668, `for outcome in [...]:` 루프 내부, `if outcome not in df_prepared.columns:` 검사 이후에 추가:

```python
        for outcome in ['dementia_event', 'ad_event', 'vad_event']:
            if outcome not in df_prepared.columns:
                continue
            if cb: cb(f"경쟁위험 분석: {outcome} 처리 중...")
```

- [ ] **Step 5: 테스트 실행 — PASS 확인**

```bash
python -m pytest tests/test_stage_q.py -v 2>&1 | tail -10
```

기대: 6개 모두 PASSED

- [ ] **Step 6: 전체 회귀 확인**

```bash
python -m pytest tests/ -q --tb=no 2>&1 | tail -5
```

기대: 0 failed

- [ ] **Step 7: 커밋**

```bash
git add statistical_analysis.py tests/test_stage_q.py
git commit -m "feat: run_cox/run_competing_risks 모델·outcome별 중간 진행 emit 추가 (Stage Q T2)"
```

---

## Task 3: `_on_analysis` — `run_post_analysis` WorkerThread 분리

**Files:**
- Modify: `tabs.py:937-960` (`_on_analysis` 리팩토링)
- Modify: `tabs.py` (신규 슬롯 `_on_post_analysis` 추가)

### 배경

`_on_analysis`는 `finished` signal에 연결된 **메인 스레드 슬롯**이다. 여기서 `run_post_analysis()`를 직접 호출하면 KM 곡선 생성, Forest plot, PSM balance plot, CIF plot, Excel 내보내기가 메인 스레드에서 동기 실행되어 UI가 freeze된다.

이 Task는 Qt 스레딩 코드라 자동화 단위 테스트 대신 동작 검증 체크리스트로 대체한다.

- [ ] **Step 1: `_on_analysis` 수정 — post_worker 생성으로 교체**

`tabs.py` 라인 937-960을 아래와 같이 교체:

```python
    def _on_analysis(self, data):
        mw = self.ctx.main_window
        ar = data.get('result', {})
        self.ctx.all_results['analysis'] = ar

        # 샘플링 레이블 갱신 (다이얼로그는 start_analysis 에서 이미 표시됨)
        sampling_info = ar.get('sampling_info')
        if sampling_info is not None and sampling_info.applied:
            self._sampling_label = sampling_info.label
            self.ctx.sampling_label = sampling_info.label
        else:
            self._sampling_label = ""
            self.ctx.sampling_label = ""

        # 시각화 + 내보내기를 별도 WorkerThread 로 실행 (메인 스레드 블로킹 방지)
        self.log_signal.emit("시각화 및 결과 내보내기 중...")

        def do_post(progress_callback=None):
            from analysis_runner import run_post_analysis
            _log = progress_callback or (lambda m: None)
            return run_post_analysis(self.ctx.dm, ar, self.ctx.results_dir, log=_log)

        from main_app import WorkerThread
        self._post_worker = WorkerThread(do_post)
        self._post_worker.progress.connect(self.log_signal.emit)
        self._post_worker.finished.connect(self._on_post_analysis)
        self._post_worker.error.connect(mw._on_error)
        self._post_worker.start()
```

- [ ] **Step 2: `_on_post_analysis` 슬롯 추가 (`_on_analysis` 바로 다음 줄)**

```python
    def _on_post_analysis(self, data):
        mw = self.ctx.main_window
        mw.progress_bar.setVisible(False)
        mw._set_action_buttons_enabled(True)
        result = data.get('result', {})
        for err in result.get('errors', []):
            self.log_signal.emit(err)
        self.log_signal.emit(f"분석 완료! 결과: {self.ctx.results_dir}")
        QMessageBox.information(self, "완료", f"분석 완료\n{self.ctx.results_dir}")
```

- [ ] **Step 3: `_on_analysis`에서 `progress_bar.setVisible(False)` 및 `_set_action_buttons_enabled(True)` 제거 확인**

기존 `_on_analysis` 라인 939-940에 있던 두 줄이 삭제되어 있는지 확인:

```python
# 삭제되었어야 할 코드:
# mw.progress_bar.setVisible(False)
# mw._set_action_buttons_enabled(True)
```

이 두 줄은 `_on_post_analysis`에만 존재해야 한다.

- [ ] **Step 4: Python 문법 오류 없음 확인**

```bash
python -m py_compile tabs.py && echo "OK"
```

기대: `OK`

- [ ] **Step 5: 기존 테스트 회귀 확인**

```bash
python -m pytest tests/ -q --tb=no 2>&1 | tail -5
```

기대: 0 failed

- [ ] **Step 6: 커밋**

```bash
git add tabs.py
git commit -m "fix: _on_analysis run_post_analysis WorkerThread 분리 — 분석 완료 후 UI freeze 수정 (Stage Q T3)"
```

---

## Task 4: `_run_step` 중복 주석 정리

**Files:**
- Modify: `cohort_builder.py:24-50`

### 배경

`_run_step`은 `build_cohort` 내부에서 직접 사용되지 않지만, `tests/test_cohort_safety.py`와 `tests/test_stage_b.py`에서 직접 호출되어 테스트된다. 삭제가 아닌 docstring 명확화로 처리한다.

- [ ] **Step 1: `_run_step` docstring 업데이트**

`cohort_builder.py` 라인 24-28를 아래와 같이 수정:

```python
    def _run_step(self, step_num: int, step_name: str, sql: str, result_table: str) -> int:
        """단계 SQL 실행 + 1회 재시도 + 행 수 검증.

        성공 시 result_table의 행 수 반환.
        실패(duckdb.Error) 또는 결과 0건 시 CohortStepError 발생.

        Note: build_cohort 는 step_fn 기반 _safe_step 내부 함수를 사용한다.
              이 메서드는 단위 테스트(test_cohort_safety.py)에서 재시도·검증 로직
              직접 검증용으로 노출되어 있다.
        """
```

- [ ] **Step 2: 문법 확인 및 회귀 테스트**

```bash
python -m py_compile cohort_builder.py && echo "OK"
python -m pytest tests/test_cohort_safety.py tests/test_stage_b.py -q --tb=short 2>&1 | tail -10
```

기대: `OK`, 관련 테스트 모두 PASSED

- [ ] **Step 3: 커밋**

```bash
git add cohort_builder.py
git commit -m "docs: _run_step 테스트 노출 의도 docstring 명확화 (Stage Q T4)"
```

---

## Task 5: `requirements.txt` `pywin32` 플랫폼 명시

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: 주석 추가**

`requirements.txt` 의 `pywin32>=306` 줄을 아래와 같이 교체:

```
pywin32>=306  # Windows 전용 — macOS/Linux 에서는 무시됨 (win32timezone PyInstaller 포함 필요)
```

- [ ] **Step 2: 커밋**

```bash
git add requirements.txt
git commit -m "docs: pywin32 Windows 전용 의존성 주석 추가 (Stage Q T5)"
```

---

## 자체 점검 (Self-Review)

### 스펙 커버리지

| 이슈 | Task | 상태 |
|------|------|------|
| I1: `_load_data`/`_prepare` cb 누락 | Task 1 | ✅ |
| C1: `run_post_analysis` 메인 스레드 블로킹 | Task 3 | ✅ |
| I2: `run_cox`/`run_competing_risks` 중간 emit | Task 2 | ✅ |
| I3: `_run_step` dead code | Task 4 | ✅ (docstring) |
| M2: `pywin32` 플랫폼 명시 | Task 5 | ✅ |

### 시그니처 일관성

- `_load_data(self, cb=None)` — Task 1에서 추가. `run_selected`에서 `_load_data(cb=cb)` 호출. 기존 직접 호출 위치(`run_cox`, `run_competing_risks` 내 fallback)는 `cb` 없이 호출 → 기본값 `None`으로 하위호환.
- `_prepare(self, df, cb=None)` — Task 1에서 추가. 기존 직접 호출 위치는 `_prepare(raw)` → 하위호환.
- `_on_post_analysis` — Task 3에서 신규. `WorkerThread.finished` signal이 `dict`를 전달 → `data.get('result', {})` 패턴은 `_on_analysis`와 동일.

### 회귀 위험

- Task 3에서 `_on_analysis`의 `mw.progress_bar.setVisible(False)` + `_set_action_buttons_enabled(True)` 제거 — 이 두 줄이 `_on_post_analysis`에만 존재해야 한다. Step 3 체크리스트로 명시됨.
- `_post_worker`를 인스턴스 변수(`self._post_worker`)로 유지하지 않으면 GC에 의해 스레드가 즉시 소멸될 수 있음 — `self._post_worker = WorkerThread(do_post)` 패턴으로 방지.
