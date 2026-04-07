# Stage R: Stage Q 리뷰 후속 조치 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stage Q 종합 리뷰(Codex+Gemini+셀프)에서 도출된 Important 2건 + Minor 3건을 수정하여 progress_callback 경로 완전성 및 UX 일관성을 확보한다.

**Architecture:** Task 1(I2: analysis_text 연결) → Task 2(I1: fallback cb 전달) → Task 3(M1+M2: 중복 emit 제거 + 스킵 메시지) → Task 4(M3+M4: 오류경로 테스트 + 주석) 순서로 구현. 각 태스크는 독립 커밋.

**Tech Stack:** Python 3.12, PyQt5 (pyqtSignal), pandas, pytest, unittest.mock

---

## 파일 변경 맵

| 파일 | 변경 유형 | 내용 |
|------|-----------|------|
| `tabs.py` | Modify | `_post_worker.progress` → `analysis_text.append` 연결 추가 |
| `statistical_analysis.py` | Modify | `run_cox`/`run_psm` fallback cb 전달; 중복 emit 라인 238 제거; 스킵 emit 추가 |
| `tests/test_stage_q.py` | Modify | fallback 경로 cb 테스트 + 스킵 emit 테스트 + 오류경로 주석 |

---

## Task 1: I2 — `_post_worker.progress` → `analysis_text` 연결

**Files:**
- Modify: `tabs.py:962`

### 배경

`_post_worker.progress`는 현재 `self.log_signal.emit`에만 연결되어 있다. 1차 분석 워커는 `self.analysis_text.append`에도 연결되어 있어(라인 932), post-analysis 메시지가 분석 탭 텍스트 박스에 표시되지 않는 UX 불일치가 있다.

- [ ] **Step 1: `tabs.py` 라인 962 바로 다음에 한 줄 추가**

현재:
```python
        self._post_worker.progress.connect(self.log_signal.emit)
        self._post_worker.finished.connect(self._on_post_analysis)
```

변경 후:
```python
        self._post_worker.progress.connect(self.log_signal.emit)
        self._post_worker.progress.connect(lambda m: self.analysis_text.append(m))
        self._post_worker.finished.connect(self._on_post_analysis)
```

- [ ] **Step 2: 문법 확인**

```bash
cd /Users/aidept/ptg_at_train/yod_diabetes_app
python -m py_compile tabs.py && echo "OK"
```

기대: `OK`

- [ ] **Step 3: 회귀 확인**

```bash
python -m pytest tests/ -q --tb=no 2>&1 | tail -5
```

기대: 200 passed, 4 failed (pre-existing)

- [ ] **Step 4: 커밋**

```bash
git add tabs.py
git commit -m "fix: _post_worker progress를 analysis_text 에도 연결 — UX 일관성 (Stage R I2)"
```

---

## Task 2: I1 — `run_cox`/`run_psm` fallback 경로 cb 전달

**Files:**
- Modify: `statistical_analysis.py:239-241` (`run_cox` fallback)
- Modify: `statistical_analysis.py:324-326` (`run_psm` fallback)
- Modify: `tests/test_stage_q.py` (테스트 추가)

### 배경

`run_cox(cb=..., df_prepared=None)` 또는 `run_psm(cb=..., df_prepared=None)` 형태로 독립 호출될 때, 내부 `_load_data()`/`_prepare()` 호출에 cb가 전달되지 않아 데이터 로딩 중 메시지 창이 무음이 된다.

현재 코드:
```python
# run_cox 라인 239-241
if df_prepared is None:
    raw, _ = self._load_data()
    df_prepared = self._prepare(raw)

# run_psm 라인 324-326
if df_prepared is None:
    raw, _ = self._load_data()
    df_prepared = self._prepare(raw)
```

- [ ] **Step 1: 실패하는 테스트 추가 (`tests/test_stage_q.py` 끝에 추가)**

```python
def test_run_cox_standalone_passes_cb_to_load_data(monkeypatch):
    """run_cox(cb=..., df_prepared=None) 시 _load_data 에 cb 가 전달되어야 한다."""
    dm = MagicMock()
    analyzer = StatisticalAnalyzer(dm)
    load_cb_received = []

    def patched_load(cb=None):
        load_cb_received.append(cb)
        raise pd.errors.EmptyDataError("테스트 중단")

    monkeypatch.setattr(analyzer, '_load_data', patched_load)
    cb = MagicMock()
    try:
        analyzer.run_cox('dementia_event', cb=cb, df_prepared=None)
    except (pd.errors.EmptyDataError, Exception):
        pass

    assert load_cb_received and load_cb_received[0] is cb, \
        f"run_cox fallback: _load_data 에 cb 미전달. received={load_cb_received}"


def test_run_psm_standalone_passes_cb_to_load_data(monkeypatch):
    """run_psm(cb=..., df_prepared=None) 시 _load_data 에 cb 가 전달되어야 한다."""
    dm = MagicMock()
    analyzer = StatisticalAnalyzer(dm)
    load_cb_received = []

    def patched_load(cb=None):
        load_cb_received.append(cb)
        raise pd.errors.EmptyDataError("테스트 중단")

    monkeypatch.setattr(analyzer, '_load_data', patched_load)
    cb = MagicMock()
    try:
        analyzer.run_psm(cb=cb, df_prepared=None)
    except (pd.errors.EmptyDataError, Exception):
        pass

    assert load_cb_received and load_cb_received[0] is cb, \
        f"run_psm fallback: _load_data 에 cb 미전달. received={load_cb_received}"
```

- [ ] **Step 2: 테스트 실행 — FAIL 확인**

```bash
python -m pytest tests/test_stage_q.py::test_run_cox_standalone_passes_cb_to_load_data tests/test_stage_q.py::test_run_psm_standalone_passes_cb_to_load_data -v 2>&1 | tail -10
```

기대: 2개 FAIL (AssertionError)

- [ ] **Step 3: `run_cox` fallback cb 전달 수정**

`statistical_analysis.py` 라인 239-241을 아래로 교체:

```python
        if df_prepared is None:
            raw, _ = self._load_data(cb=cb)
            df_prepared = self._prepare(raw, cb=cb)
```

- [ ] **Step 4: `run_psm` fallback cb 전달 수정**

`statistical_analysis.py` 라인 324-326을 아래로 교체:

```python
        if df_prepared is None:
            raw, _ = self._load_data(cb=cb)
            df_prepared = self._prepare(raw, cb=cb)
```

- [ ] **Step 5: 테스트 PASS 확인**

```bash
python -m pytest tests/test_stage_q.py -v 2>&1 | tail -15
```

기대: 9개 모두 PASSED

- [ ] **Step 6: 전체 회귀 확인**

```bash
python -m pytest tests/ -q --tb=no 2>&1 | tail -5
```

기대: 200 passed, 4 failed (pre-existing)

- [ ] **Step 7: 커밋**

```bash
git add statistical_analysis.py tests/test_stage_q.py
git commit -m "fix: run_cox/run_psm fallback 경로에서 _load_data/_prepare 에 cb 전달 (Stage R I1)"
```

---

## Task 3: M1 + M2 — 중복 emit 제거 + 스킵 메시지 추가

**Files:**
- Modify: `statistical_analysis.py:238` (M1: 중복 emit 제거)
- Modify: `statistical_analysis.py:686-687` (M2: 스킵 emit 추가)
- Modify: `tests/test_stage_q.py` (M2 테스트 추가)

### M1 배경

`run_cox` 라인 238: `if cb: cb(f"Cox 회귀 ({outcome})...")`
`run_cox` 라인 269: `if cb: cb(f"Cox 회귀 ({outcome}) — {mname} 피팅 중...")`

라인 238의 요약 메시지는 라인 269의 첫 번째 모델 메시지와 outcome 이름이 중복된다. 라인 238을 제거하면 첫 메시지가 `"Cox 회귀 (dementia_event) — model1_age_sex 피팅 중..."`으로 자연스럽게 시작된다.

### M2 배경

`run_competing_risks`의 라인 676에서 emit 후 라인 686-687에서 `len(df_cr) < _min_cr`이면 `continue`로 스킵한다. 사용자 입장에서 "처리 중" 메시지 후 결과가 없어 혼란스럽다.

- [ ] **Step 1: M2 실패하는 테스트 추가 (`tests/test_stage_q.py` 끝에 추가)**

```python
def test_run_competing_risks_emits_skip_message_when_insufficient_rows():
    """행 수 부족으로 스킵될 때 스킵 메시지를 emit 해야 한다."""
    dm = MagicMock()
    analyzer = StatisticalAnalyzer(dm)

    # MIN_VALID_ROWS=30 기준 — 행 수를 29로 만들어 스킵 유도
    # competing_death_event 없으면 전체 스킵이므로 포함
    n = 29
    df = pd.DataFrame({
        'exposure_group': ['NON_DM'] * n,
        'is_t1dm': [0] * n, 'is_t2dm_oha': [0] * n,
        'is_t2dm_insulin': [0] * n, 'is_t2dm_nomed': [0] * n,
        'age_at_index': [55.0] * n, 'male': [1] * n,
        'follow_up_years': [1.0] * n,
        'dementia_event': [0] * 24 + [1] * 5,
        'competing_death_event': [0] * 25 + [1] * 4,
    })
    messages = []
    analyzer.run_competing_risks(cb=messages.append, df_prepared=df)

    skip_msgs = [m for m in messages if '스킵' in m or 'skip' in m.lower()]
    assert skip_msgs, f"스킵 메시지 없음. 실제: {messages}"
```

- [ ] **Step 2: 테스트 실행 — FAIL 확인**

```bash
python -m pytest tests/test_stage_q.py::test_run_competing_risks_emits_skip_message_when_insufficient_rows -v 2>&1 | tail -10
```

기대: FAIL (AssertionError)

- [ ] **Step 3: M1 — 라인 238 제거**

`statistical_analysis.py`에서 다음 줄을 삭제:

```python
        if cb: cb(f"Cox 회귀 ({outcome})...")
```

(라인 238. `def run_cox(...)` 바로 다음 줄)

- [ ] **Step 4: M2 — 스킵 emit 추가**

`statistical_analysis.py` 라인 686-687:

```python
            if len(df_cr) < _min_cr:
                continue
```

를 아래로 교체:

```python
            if len(df_cr) < _min_cr:
                if cb: cb(f"경쟁위험 분석: {outcome} 스킵 (유효 행 {len(df_cr)} < {_min_cr})")
                continue
```

- [ ] **Step 5: M2 테스트 PASS + 전체 회귀 확인**

```bash
python -m pytest tests/test_stage_q.py -v 2>&1 | tail -15
python -m pytest tests/ -q --tb=no 2>&1 | tail -5
```

기대: test_stage_q.py 10개 PASSED, 전체 200 passed

- [ ] **Step 6: 커밋**

```bash
git add statistical_analysis.py tests/test_stage_q.py
git commit -m "fix: run_cox 중복 emit 제거 + run_competing_risks 스킵 메시지 추가 (Stage R M1+M2)"
```

---

## Task 4: M3 + M4 — 오류경로 테스트 + 이중 참조 주석

**Files:**
- Modify: `tests/test_stage_q.py` (M3: 오류경로 테스트)
- Modify: `tabs.py:960-961` (M4: 주석 보완)

### M3 배경

`_post_worker`에서 오류 발생 시 `error.connect(mw._on_error)` → `_on_error`가 `progress_bar.setVisible(False)` + `_set_action_buttons_enabled(True)`를 수행함을 코드 리딩으로만 확인했다. 이를 테스트로 고정한다. Qt 환경 없이 테스트하려면 `main_app._on_error`를 직접 테스트한다.

### M4 배경

`tabs.py` 라인 960-961에서 `self._post_worker`와 `mw.worker` 이중 등록 패턴의 의도가 불명확하다.

- [ ] **Step 1: M3 테스트 추가 (`tests/test_stage_q.py` 끝에 추가)**

```python
def test_main_app_on_error_hides_progress_bar():
    """_on_error 가 progress_bar 를 숨기고 버튼을 활성화해야 한다 — _post_worker 오류 경로 검증."""
    import sys
    sys.path.insert(0, '/Users/aidept/ptg_at_train/yod_diabetes_app')
    from unittest.mock import MagicMock, patch

    # MainWindow._on_error 를 직접 테스트 (Qt 없이)
    # _on_error 내부 동작: progress_bar.setVisible(False), _set_action_buttons_enabled(True)
    with patch('main_app.QMessageBox') as mock_qmb:
        import main_app
        mw = MagicMock()
        # _on_error 를 언바운드 메서드로 직접 호출
        main_app.MainWindow._on_error(mw, "테스트 오류")

    mw.progress_bar.setVisible.assert_called_once_with(False)
    mw._set_action_buttons_enabled.assert_called_once_with(True)
```

- [ ] **Step 2: 테스트 실행 — PASS 확인**

```bash
python -m pytest tests/test_stage_q.py::test_main_app_on_error_hides_progress_bar -v 2>&1 | tail -10
```

기대: PASSED

- [ ] **Step 3: M4 — `tabs.py` 주석 보완**

`tabs.py` 라인 960-961의 주석을 보완:

현재:
```python
        self._post_worker = WorkerThread(do_post)
        mw.worker = self._post_worker  # closeEvent 에서 cleanup 가능하도록 등록
```

변경 후:
```python
        self._post_worker = WorkerThread(do_post)  # GC 방지: 인스턴스 변수 유지
        mw.worker = self._post_worker  # closeEvent cleanup + mw.worker 는 '현재 실행 중인 최신 워커' 관례
```

- [ ] **Step 4: 전체 테스트 확인**

```bash
python -m pytest tests/ -q --tb=no 2>&1 | tail -5
```

기대: 200 passed (+ 신규 1개 = 201 passed), 4 failed (pre-existing)

- [ ] **Step 5: 커밋**

```bash
git add tests/test_stage_q.py tabs.py
git commit -m "test: _on_error progress_bar 복구 경로 테스트 + 이중참조 주석 보완 (Stage R M3+M4)"
```

---

## 자체 점검

### 스펙 커버리지

| 이슈 | Task | 상태 |
|------|------|------|
| I2: analysis_text 연결 | Task 1 | ✅ |
| I1: run_cox/run_psm fallback cb | Task 2 | ✅ |
| M1: run_cox 중복 emit | Task 3 | ✅ |
| M2: 스킵 emit | Task 3 | ✅ |
| M3: 오류경로 테스트 | Task 4 | ✅ |
| M4: 이중 참조 주석 | Task 4 | ✅ |

### 시그니처 일관성

- `_load_data(cb=cb)` — Task 2에서 `run_cox`/`run_psm` 내부 fallback에 적용. Stage Q T1에서 이미 `_load_data(self, cb=None)` 정의됨.
- `_prepare(raw, cb=cb)` — 동일하게 Stage Q T1에서 `_prepare(self, df, cb=None)` 정의됨.

### M3 테스트 접근법 검토

Qt 없이 `MainWindow._on_error`를 테스트하는 방법으로 언바운드 메서드 직접 호출 + `MagicMock` 사용. `QMessageBox` import가 `main_app.py` 최상위에서 이루어지므로 `patch('main_app.QMessageBox')`로 막아야 한다. `main_app.py`에서 `from PyQt5.QtWidgets import QApplication, QMainWindow, ...` 가 최상위에 있으면 PyQt5 없이 import가 실패한다 — 이 경우 테스트가 collection 오류로 실패한다.

현재 환경(macOS 개발 머신)에서 `import main_app`이 성공하는지 확인이 필요하다. PyQt5가 설치되어 있다면 테스트가 동작하고, 없다면 기존 4개 pre-existing 오류와 동일하게 collection 단계에서 스킵된다. 즉, 기존 환경과 동일한 조건에서만 동작하므로 안전하다.

단, 만약 `import main_app`이 실패한다면 `pytest.importorskip('main_app')`으로 조건부 스킵을 추가해야 한다.
