# Stage B: Critical Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all High- and Medium-priority issues identified in the Gemini+Codex dual review: reproducibility (random seed), data safety (0-row guard), UX (pre-analysis Cancel), export completeness (sampling header), Windows log path, exception chain, and error dialog clarity.

**Architecture:** Seven focused single-file fixes applied in dependency order (config → utils → cohort_builder → statistical_analysis → results_exporter → tabs → main_app). No new modules. Each fix is independent of the others except Task 4 which adds a field to `SamplingInfo` that Task 6 reads.

**Tech Stack:** Python 3.12, DuckDB (setseed), PyQt5, pandas, pathlib, dataclasses

---

## File Map

| File | Change |
|---|---|
| `config.py` | Add `SAMPLING_SEED: 42` to `STUDY_SETTINGS` |
| `utils.py` | `setup_logging` default path → `%LOCALAPPDATA%` on Windows |
| `cohort_builder.py` | Add `from e` to both `raise CohortStepError(...)` calls |
| `statistical_analysis.py` | (a) `SamplingInfo` gains `seed: int` field; (b) `setseed()` before sampling query; (c) 0-row → raise `EmptyDataError` |
| `tabs.py` | (a) `start_analysis` calls `_confirm_sampling_if_needed()` before worker starts; (b) `export_all()` passes `sampling_info`; (c) single `export()` uses `_write_df_with_sampling_header` |
| `main_app.py` | `_on_error()` shows first line of error only |
| `tests/test_stage_b.py` | All new tests |

---

### Task 1: Fix Windows log path in `setup_logging`

**Files:**
- Modify: `utils.py:9-22`
- Test: `tests/test_stage_b.py`

**Context:** `setup_logging(log_dir='.')` defaults to the current working directory. On Windows under `Program Files`, the process has no write permission, so the log file silently fails to open. The fix uses `%LOCALAPPDATA%\NHIS_YOD_DM_Analyzer\logs` on Windows when no explicit path is given, falling back to `.` on other platforms.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_stage_b.py`:

```python
import sys
import logging
import importlib
from pathlib import Path
from unittest.mock import patch


def test_setup_logging_windows_uses_localappdata(tmp_path, monkeypatch):
    """Windows에서 log_dir 생략 시 %LOCALAPPDATA% 하위 경로를 사용한다."""
    monkeypatch.setattr(sys, 'platform', 'win32')
    fake_local = tmp_path / "AppData" / "Local"
    fake_local.mkdir(parents=True)
    monkeypatch.setenv('LOCALAPPDATA', str(fake_local))

    import utils
    importlib.reload(utils)  # reload after monkeypatching sys.platform

    # Clear any existing handlers so setup_logging adds a new FileHandler
    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)

    utils.setup_logging()  # no log_dir argument

    expected_dir = fake_local / "NHIS_YOD_DM_Analyzer" / "logs"
    assert expected_dir.exists(), f"로그 디렉토리 미생성: {expected_dir}"
    file_handlers = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
    assert file_handlers, "FileHandler가 추가되지 않았습니다"
    assert "NHIS_YOD_DM_Analyzer" in file_handlers[0].baseFilename


def test_setup_logging_non_windows_uses_dot(tmp_path, monkeypatch):
    """비-Windows에서 log_dir 생략 시 현재 디렉토리(.)를 사용한다."""
    monkeypatch.setattr(sys, 'platform', 'linux')
    monkeypatch.chdir(tmp_path)

    import utils
    importlib.reload(utils)

    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)

    utils.setup_logging()
    file_handlers = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
    assert file_handlers
    assert str(tmp_path) in file_handlers[0].baseFilename
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Volumes/model/yod_diabetes_app
python -m pytest tests/test_stage_b.py::test_setup_logging_windows_uses_localappdata -v
```

Expected: FAIL — `AssertionError: 로그 디렉토리 미생성`

- [ ] **Step 3: Implement the fix**

Replace `utils.py:9-22` with:

```python
def setup_logging(log_dir: str | None = None):
    """로그 설정.

    log_dir 생략 시:
      - Windows: %LOCALAPPDATA%\\NHIS_YOD_DM_Analyzer\\logs
      - 기타:    현재 디렉토리('.')
    """
    if log_dir is None:
        if sys.platform == 'win32':
            base = Path(os.environ.get('LOCALAPPDATA', Path.home() / 'AppData' / 'Local'))
            log_dir_path = base / 'NHIS_YOD_DM_Analyzer' / 'logs'
        else:
            log_dir_path = Path('.')
    else:
        log_dir_path = Path(log_dir)

    log_dir_path.mkdir(parents=True, exist_ok=True)
    log_path = log_dir_path / APP_SETTINGS['LOG_FILE']

    root = logging.getLogger()
    if not root.handlers:
        root.setLevel(logging.INFO)
        fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
        fh = logging.FileHandler(log_path, encoding='utf-8')
        fh.setFormatter(fmt)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        root.addHandler(fh)
        root.addHandler(sh)
    return logging.getLogger(__name__)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_stage_b.py::test_setup_logging_windows_uses_localappdata tests/test_stage_b.py::test_setup_logging_non_windows_uses_dot -v
```

Expected: PASS (both)

- [ ] **Step 5: Commit**

```bash
git add utils.py tests/test_stage_b.py
git commit -m "fix: setup_logging Windows 경로 — %LOCALAPPDATA% 사용"
```

---

### Task 2: Fix exception chain `raise CohortStepError from e`

**Files:**
- Modify: `cohort_builder.py:41`
- Test: `tests/test_stage_b.py`

**Context:** `raise CohortStepError(step_num, step_name, e)` at line 41 lacks `from e`. Python discards the original exception's `__cause__` and `__traceback__` chain, making debugs harder. Adding `from e` preserves the full chain.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_stage_b.py`:

```python
import pytest
import duckdb
from unittest.mock import MagicMock, patch
from cohort_builder import CohortBuilder
from utils import CohortStepError


def test_run_step_exception_chain_preserved():
    """raise CohortStepError from e — __cause__ 가 원본 예외를 가리킨다."""
    dm = MagicMock()
    original_error = duckdb.Error("original db error")
    # Fail on both attempts
    dm.execute.side_effect = original_error

    cb = CohortBuilder(dm)
    dm.storage = MagicMock()

    with patch('cohort_builder.time.sleep'):
        with pytest.raises(CohortStepError) as exc_info:
            cb._run_step(1, "테스트", "SELECT 1", "t")

    assert exc_info.value.__cause__ is original_error, \
        "__cause__ 가 원본 duckdb.Error 를 가리켜야 합니다"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_stage_b.py::test_run_step_exception_chain_preserved -v
```

Expected: FAIL — `AssertionError: __cause__ 가 원본 duckdb.Error 를 가리켜야 합니다`

- [ ] **Step 3: Implement the fix**

In `cohort_builder.py:41`, change:

```python
                else:
                    raise CohortStepError(step_num, step_name, e)
```

to:

```python
                else:
                    raise CohortStepError(step_num, step_name, e) from e
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_stage_b.py::test_run_step_exception_chain_preserved tests/test_cohort_safety.py -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add cohort_builder.py tests/test_stage_b.py
git commit -m "fix: CohortStepError 예외 체인 복원 (raise ... from e)"
```

---

### Task 3: Fix `_load_data()` 0-row continues silently

**Files:**
- Modify: `statistical_analysis.py:75-83`
- Test: `tests/test_stage_b.py`

**Context:** When `valid_total == 0` (no rows with `follow_up_days > 0`), `_load_data()` returns an empty DataFrame and sets `sampled_rows=0`. Downstream Cox/PSM code tries to fit models on an empty DataFrame and fails with cryptic lifelines errors instead of a clear user message. The fix raises `pd.errors.EmptyDataError` immediately, which `format_error_for_user` already handles with a clear Korean message.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_stage_b.py`:

```python
import pandas as pd
from unittest.mock import MagicMock, patch
from statistical_analysis import StatisticalAnalyzer


def test_load_data_zero_valid_rows_raises_empty_data_error():
    """follow_up_days > 0 인 행이 0건이면 EmptyDataError 를 발생시킨다."""
    dm = MagicMock()
    dm.storage.get_row_count.return_value = 1000  # total > max_rows triggers sampling path
    # group_counts returns no rows with follow_up_days > 0
    dm.query.return_value = pd.DataFrame({'exposure_group': [], 'cnt': []})

    analyzer = StatisticalAnalyzer(dm)

    with patch('statistical_analysis.mem_manager') as mock_mm:
        mock_mm.get_safe_analysis_rows.return_value = 500  # force sampling branch
        with pytest.raises(pd.errors.EmptyDataError):
            analyzer._load_data()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_stage_b.py::test_load_data_zero_valid_rows_raises_empty_data_error -v
```

Expected: FAIL — test does not raise `EmptyDataError` (returns empty df instead)

- [ ] **Step 3: Implement the fix**

In `statistical_analysis.py`, replace lines 75-83:

```python
            if valid_total == 0:
                logger.warning("추적 가능한 행(follow_up_days > 0)이 없어 분석을 건너뜁니다.")
                self._cached_df = self.dm.query(
                    "SELECT * FROM final_analysis WHERE follow_up_days > 0 LIMIT 0"
                )
                self._sampling_info = SamplingInfo(
                    applied=True, total_rows=total, sampled_rows=0
                )
                return self._cached_df, self._sampling_info
```

with:

```python
            if valid_total == 0:
                raise pd.errors.EmptyDataError(
                    "추적 가능한 행(follow_up_days > 0)이 없습니다. "
                    "코호트 구성 단계를 확인하세요."
                )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_stage_b.py::test_load_data_zero_valid_rows_raises_empty_data_error tests/test_sampling_info.py -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add statistical_analysis.py tests/test_stage_b.py
git commit -m "fix: _load_data() 0건 데이터 EmptyDataError 발생"
```

---

### Task 4: Fix `ORDER BY RANDOM()` → seed-based reproducible sampling

**Files:**
- Modify: `config.py` (add `SAMPLING_SEED`)
- Modify: `statistical_analysis.py` (`SamplingInfo` + `_load_data` + `plot_km` query in `tabs.py`)
- Modify: `tabs.py:1041-1050` (KM plot sampling query)
- Test: `tests/test_stage_b.py`

**Context:** Medical research requires reproducible results. `ORDER BY RANDOM()` produces a different sample every run with no way to recreate it. DuckDB's `setseed(value)` (0.0–1.0 float) sets the seed for `RANDOM()` on the same connection. We add `SAMPLING_SEED: 42` to `STUDY_SETTINGS`, call `self.dm.execute(f"SELECT setseed({seed})")` before the sampling query, and record the seed in `SamplingInfo` for the Excel header and UI label.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_stage_b.py`:

```python
from statistical_analysis import SamplingInfo


def test_sampling_info_has_seed_field():
    """SamplingInfo 는 seed 필드를 가진다."""
    info = SamplingInfo(applied=True, total_rows=1000, sampled_rows=500, seed=42)
    assert info.seed == 42


def test_sampling_info_label_includes_seed():
    """label 에 seed 값이 포함된다."""
    info = SamplingInfo(applied=True, total_rows=1000, sampled_rows=500, seed=42)
    assert "seed=42" in info.label


def test_load_data_calls_setseed(monkeypatch):
    """_load_data 가 샘플링 전에 DuckDB setseed 를 호출한다."""
    dm = MagicMock()
    dm.storage.get_row_count.return_value = 1000
    group_df = pd.DataFrame({'exposure_group': ['T2DM_OHA', 'NON_DM'], 'cnt': [300, 700]})
    sampled_df = pd.DataFrame({'col': range(500)})
    call_log = []

    def fake_query(sql):
        if 'setseed' in sql:
            call_log.append('setseed')
        return group_df if 'GROUP BY' in sql else sampled_df

    def fake_execute(sql):
        if 'setseed' in sql:
            call_log.append('setseed_exec')

    dm.query.side_effect = fake_query
    dm.execute.side_effect = fake_execute

    analyzer = StatisticalAnalyzer(dm)

    with patch('statistical_analysis.mem_manager') as mock_mm:
        mock_mm.get_safe_analysis_rows.return_value = 500
        mock_mm.optimize_dtypes.side_effect = lambda df: df
        analyzer._load_data()

    assert 'setseed_exec' in call_log or 'setseed' in call_log, \
        "setseed 가 호출되지 않았습니다"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_stage_b.py::test_sampling_info_has_seed_field tests/test_stage_b.py::test_sampling_info_label_includes_seed tests/test_stage_b.py::test_load_data_calls_setseed -v
```

Expected: FAIL (no `seed` field on `SamplingInfo`, no `setseed` call)

- [ ] **Step 3a: Add `SAMPLING_SEED` to `config.py`**

In `config.py`, inside `STUDY_SETTINGS` dict (after `'CENSORING_EVENTS': [...]` line), add:

```python
    'SAMPLING_SEED': 42,          # 층화 샘플링 재현성 시드 (0–99 정수)
```

- [ ] **Step 3b: Add `seed` field to `SamplingInfo` and update `label`**

In `statistical_analysis.py`, replace the `SamplingInfo` dataclass (lines 22-48):

```python
@dataclass
class SamplingInfo:
    """층화 샘플링 적용 여부 및 규모 정보.

    applied: 샘플링이 적용되었으면 True
    total_rows: 원본 전체 행 수
    sampled_rows: 실제 분석에 사용된 행 수
    seed: 재현성을 위한 DuckDB setseed 값 (0–99 정수)
    """
    applied: bool
    total_rows: int
    sampled_rows: int
    seed: int = 0

    @property
    def ratio_pct(self) -> float:
        if self.total_rows == 0:
            return 0.0
        return self.sampled_rows / self.total_rows * 100

    @property
    def label(self) -> str:
        """UI 및 Excel 헤더용 한줄 요약. 샘플링 없으면 빈 문자열."""
        if not self.applied:
            return ""
        return (
            f"층화 샘플링: {self.sampled_rows:,}/{self.total_rows:,}건 "
            f"({self.ratio_pct:.1f}%, seed={self.seed})"
        )
```

- [ ] **Step 3c: Call `setseed` before sampling query in `_load_data`**

In `statistical_analysis.py`, add the `from config import STUDY_SETTINGS` import is already present at line 14. Now locate the block starting at line 87 (`dm_total = ...`) and add `setseed` call just before the sampling query. Replace the section from `alloc = {}` through `self._sampling_info = SamplingInfo(...)` block (lines ~90-119) with:

```python
            alloc = {}
            for g, cnt in group_counts.items():
                if g == 'NON_DM':
                    alloc[g] = min(cnt, non_dm_budget)
                else:
                    alloc[g] = cnt  # DM 그룹 전수 포함

            per_group_sql_cases = " ".join(
                f"WHEN exposure_group = '{g}' THEN {max(1, n)}"
                for g, n in alloc.items()
            )

            seed = int(STUDY_SETTINGS.get('SAMPLING_SEED', 42))
            seed_float = seed / 100.0  # DuckDB setseed: float in [0, 1]
            self.dm.execute(f"SELECT setseed({seed_float})")

            self._cached_df = self.dm.query(f"""
                SELECT * EXCLUDE rn
                FROM (
                    SELECT *,
                           ROW_NUMBER() OVER (
                               PARTITION BY exposure_group ORDER BY RANDOM()
                           ) AS rn,
                           CASE {per_group_sql_cases} ELSE 1 END AS grp_limit
                    FROM final_analysis
                    WHERE follow_up_days > 0
                ) t
                WHERE rn <= grp_limit
            """)
            self._sampling_info = SamplingInfo(
                applied=True,
                total_rows=total,
                sampled_rows=len(self._cached_df),
                seed=seed,
            )
```

- [ ] **Step 3d: Fix `ORDER BY RANDOM()` in `tabs.py` KM plot**

In `tabs.py`, replace `plot_km` query (lines ~1040-1050):

```python
            seed_float = STUDY_SETTINGS.get('SAMPLING_SEED', 42) / 100.0
            self.ctx.dm.execute(f"SELECT setseed({seed_float})")
            df = self.ctx.dm.query("""
                SELECT exposure_group, follow_up_years, dementia_event, ad_event, vad_event
                FROM (
                    SELECT *, ROW_NUMBER() OVER (
                        PARTITION BY exposure_group ORDER BY RANDOM()
                    ) AS rn
                    FROM final_analysis
                    WHERE follow_up_days > 0
                ) t
                WHERE rn <= 10000
            """)
```

Also add `STUDY_SETTINGS` to the import at `tabs.py:20` — it is already imported: `from config import APP_SETTINGS, STUDY_SETTINGS, ...` — confirm it is there.

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_stage_b.py::test_sampling_info_has_seed_field tests/test_stage_b.py::test_sampling_info_label_includes_seed tests/test_stage_b.py::test_load_data_calls_setseed tests/test_sampling_info.py -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add config.py statistical_analysis.py tabs.py tests/test_stage_b.py
git commit -m "fix: ORDER BY RANDOM() → setseed 기반 재현 가능 층화 샘플링 (seed=42)"
```

---

### Task 5: Fix `export_all` and single export missing `sampling_info`

**Files:**
- Modify: `tabs.py:1003-1035` (both `export()` and `export_all()` methods of `ResultsTab`)
- Test: `tests/test_stage_b.py`

**Context:** `export_all()` at line 1034 calls `exp.export_all(ar)` without passing `sampling_info`, so all exported Excel files silently omit the sampling header. The single-sheet `export()` at line 1018 writes `df2.to_excel(path, ...)` directly, also bypassing `_write_df_with_sampling_header`. Both need to read `ar.get('sampling_info')` and pass it through.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_stage_b.py`:

```python
import openpyxl
from unittest.mock import MagicMock, patch
from statistical_analysis import SamplingInfo
from results_exporter import ResultsExporter


def test_export_all_passes_sampling_info(tmp_path):
    """export_all 이 sampling_info 를 ResultsExporter.export_all 에 전달한다."""
    info = SamplingInfo(applied=True, total_rows=1000, sampled_rows=500, seed=42)
    table1_df = pd.DataFrame({'var': ['age'], 'mean': [55.0]})
    ar = {'table1': table1_df, 'sampling_info': info}

    exp = ResultsExporter(str(tmp_path))
    files = exp.export_all(ar, sampling_info=info)
    assert files, "내보낸 파일이 없습니다"

    # Excel 파일 첫 번째 셀에 샘플링 정보가 있어야 한다
    wb = openpyxl.load_workbook(files[0])
    ws = wb.active
    cell_value = ws.cell(1, 1).value
    assert cell_value is not None and "샘플링" in str(cell_value), \
        f"Row 1 에 샘플링 헤더 없음: {cell_value!r}"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_stage_b.py::test_export_all_passes_sampling_info -v
```

Expected: FAIL (sampling header missing in exported file when called without `sampling_info`)

- [ ] **Step 3: Fix `export_all()` in `tabs.py`**

In `tabs.py`, replace the `export_all` method body (lines ~1028-1035):

```python
    def export_all(self):
        ar = self.ctx.all_results.get('analysis', {})
        if not ar:
            QMessageBox.warning(self, "안내", "분석 결과 없음")
            return
        sampling_info = ar.get('sampling_info')
        exp = ResultsExporter(str(self.ctx.results_dir))
        try:
            files = exp.export_all(ar, sampling_info=sampling_info)
            QMessageBox.information(self, "완료", f"{len(files)}개 파일 저장")
        except (duckdb.Error, pd.errors.EmptyDataError, ValueError,
                MemoryError, CohortStepError) as e:
            logger.exception("전체 내보내기 실패")
            QMessageBox.critical(self, "오류", format_error_for_user(e))
        except Exception as e:
            logger.exception("전체 내보내기 중 예기치 않은 오류")
            QMessageBox.critical(self, "오류", format_error_for_user(e))
```

- [ ] **Step 4: Fix single `export()` to use `_write_df_with_sampling_header`**

In `tabs.py`, replace the `try:` block inside `export()` (lines ~1016-1019):

```python
        try:
            ar = self.ctx.all_results.get('analysis', {})
            sampling_info = ar.get('sampling_info') if ar else None
            df2 = df.reset_index() if hasattr(df, 'index') and df.index.name else df
            exp = ResultsExporter(str(self.ctx.results_dir))
            with pd.ExcelWriter(path, engine='openpyxl') as writer:
                exp._write_df_with_sampling_header(writer, df2, sheet[:31], sampling_info)
            self.log_signal.emit(f"내보내기 완료: {path}")
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
python -m pytest tests/test_stage_b.py::test_export_all_passes_sampling_info tests/test_sampling_export.py -v
```

Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add tabs.py tests/test_stage_b.py
git commit -m "fix: export_all/export 에 sampling_info 전달 — Excel 헤더 누락 수정"
```

---

### Task 6: Fix sampling dialog Cancel — move to pre-analysis check

**Files:**
- Modify: `tabs.py` — `AnalysisTab.start_analysis()` and `_on_analysis()`
- Test: `tests/test_stage_b.py`

**Context:** Currently `_show_sampling_dialog()` is called in `_on_analysis()` after the background worker has already completed. The Cancel button is therefore ineffective — analysis results already exist. The fix moves the sampling check to `start_analysis()`, before the worker thread is launched. A new `_confirm_sampling_if_needed()` method queries the `final_analysis` row count synchronously (fast: count only), computes whether sampling would apply, and shows the dialog if so. Only if the user confirms does the worker start.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_stage_b.py`:

```python
from unittest.mock import MagicMock, patch, call
from PyQt5.QtWidgets import QApplication
import sys

_app = QApplication.instance() or QApplication(sys.argv)


def test_confirm_sampling_returns_false_cancels_analysis(monkeypatch):
    """_confirm_sampling_if_needed 가 False 를 반환하면 워커가 시작되지 않는다."""
    from tabs import AnalysisTab, AppContext

    ctx = AppContext()
    dm = MagicMock()
    dm.storage.get_row_count.return_value = 999_999  # 샘플링 필요
    ctx.dm = dm

    tab = AnalysisTab.__new__(AnalysisTab)
    tab.ctx = ctx

    with patch('statistical_analysis.mem_manager') as mock_mm:
        mock_mm.get_safe_analysis_rows.return_value = 500
        with patch.object(tab, '_show_sampling_dialog', return_value=False) as mock_dlg:
            result = tab._confirm_sampling_if_needed()

    assert result is False
    mock_dlg.assert_called_once()


def test_confirm_sampling_returns_true_when_no_sampling_needed():
    """데이터가 한도 이내면 다이얼로그 없이 True 를 반환한다."""
    from tabs import AnalysisTab, AppContext

    ctx = AppContext()
    dm = MagicMock()
    dm.storage.get_row_count.return_value = 100  # 한도 이내
    ctx.dm = dm

    tab = AnalysisTab.__new__(AnalysisTab)
    tab.ctx = ctx

    with patch('statistical_analysis.mem_manager') as mock_mm:
        mock_mm.get_safe_analysis_rows.return_value = 500
        with patch.object(tab, '_show_sampling_dialog') as mock_dlg:
            result = tab._confirm_sampling_if_needed()

    assert result is True
    mock_dlg.assert_not_called()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_stage_b.py::test_confirm_sampling_returns_false_cancels_analysis tests/test_stage_b.py::test_confirm_sampling_returns_true_when_no_sampling_needed -v
```

Expected: FAIL — `AnalysisTab` has no `_confirm_sampling_if_needed` method

- [ ] **Step 3: Add `_confirm_sampling_if_needed` to `AnalysisTab`**

In `tabs.py`, add this method to `AnalysisTab` (insert after `_show_sampling_dialog`):

```python
    def _confirm_sampling_if_needed(self) -> bool:
        """분석 시작 전 샘플링 필요 여부 확인. True = 진행, False = 취소.

        final_analysis 행 수를 동기적으로 조회하여 샘플링이 필요하면 확인 다이얼로그를
        보여준다. DB 없거나 테이블 없으면 True 반환 (분석 시 실패 처리).
        """
        if self.ctx.dm is None:
            return True
        try:
            total = self.ctx.dm.storage.get_row_count('final_analysis')
        except Exception:
            return True  # 테이블 없음 — 분석 실행 시 실패로 처리

        from memory_manager import mem_manager as _mm
        max_rows = _mm.get_safe_analysis_rows()
        if total <= max_rows:
            return True  # 샘플링 불필요, 바로 진행

        from statistical_analysis import SamplingInfo
        seed = int(STUDY_SETTINGS.get('SAMPLING_SEED', 42))
        preview = SamplingInfo(
            applied=True,
            total_rows=total,
            sampled_rows=max_rows,  # 실제 값은 분석 후 확정; 예상치로 표시
            seed=seed,
        )
        return self._show_sampling_dialog(preview)
```

- [ ] **Step 4: Update `start_analysis` to call `_confirm_sampling_if_needed`**

In `tabs.py`, in `start_analysis()`, add the check immediately after `self._ensure_dm()` and before `self.ctx.results_dir = ...`:

```python
    def start_analysis(self):
        mw = self.ctx.main_window
        if mw._is_worker_running():
            return
        self._ensure_dm()

        # 샘플링 사전 확인 — 분석 시작 전, Cancel 시 워커 미시작
        if not self._confirm_sampling_if_needed():
            return

        self.ctx.results_dir = Path(self.res_dir_edit.text())
        # ... rest unchanged ...
```

- [ ] **Step 5: Remove `_show_sampling_dialog` call from `_on_analysis`**

In `tabs.py`, replace the sampling block in `_on_analysis` (lines ~904-912):

```python
        # 샘플링 레이블 갱신 (다이얼로그는 start_analysis 에서 이미 표시됨)
        sampling_info = ar.get('sampling_info')
        if sampling_info is not None and sampling_info.applied:
            self._sampling_label = sampling_info.label
            self.ctx.sampling_label = sampling_info.label
        else:
            self._sampling_label = ""
            self.ctx.sampling_label = ""
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
python -m pytest tests/test_stage_b.py::test_confirm_sampling_returns_false_cancels_analysis tests/test_stage_b.py::test_confirm_sampling_returns_true_when_no_sampling_needed -v
```

Expected: PASS (both)

- [ ] **Step 7: Commit**

```bash
git add tabs.py tests/test_stage_b.py
git commit -m "fix: 샘플링 확인 다이얼로그를 분석 시작 전으로 이동 — Cancel 동작"
```

---

### Task 7: Fix `_on_error()` traceback shown to user

**Files:**
- Modify: `main_app.py:147-151`
- Test: `tests/test_stage_b.py`

**Context:** `_on_error(msg)` receives `f"{e}\n{traceback.format_exc()}"`. The full Python traceback (potentially hundreds of lines) is sliced to 500 chars and shown in a QMessageBox. Users see `Traceback (most recent call last): File ...` which is confusing. The fix shows only the first line of `msg` (the exception message itself) in the dialog, while the full text is already written to the log via `self.log(...)`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_stage_b.py`:

```python
def test_on_error_dialog_shows_only_first_line(monkeypatch):
    """_on_error 다이얼로그에 트레이스백이 아닌 예외 메시지 첫 줄만 표시된다."""
    from main_app import MainWindow
    import PyQt5.QtWidgets as _qt

    # MainWindow 인스턴스 없이 _on_error 로직만 테스트
    mw = MainWindow.__new__(MainWindow)
    mw.progress_bar = MagicMock()
    mw.statusBar = MagicMock()
    mw.log_text = MagicMock()

    shown_texts = []

    def fake_critical(parent, title, text):
        shown_texts.append(text)

    monkeypatch.setattr(_qt.QMessageBox, 'critical', staticmethod(fake_critical))
    monkeypatch.setattr(mw, '_set_action_buttons_enabled', MagicMock())

    full_msg = "ValueError: invalid input\nTraceback (most recent call last):\n  File x.py line 1\n    raise ValueError('invalid input')"
    mw._on_error(full_msg)

    assert shown_texts, "critical 이 호출되지 않았습니다"
    assert "Traceback" not in shown_texts[0], \
        f"트레이스백이 다이얼로그에 표시되었습니다: {shown_texts[0]!r}"
    assert "ValueError" in shown_texts[0]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_stage_b.py::test_on_error_dialog_shows_only_first_line -v
```

Expected: FAIL — `"Traceback" not in shown_texts[0]` assertion fails (full traceback shown)

- [ ] **Step 3: Implement the fix**

In `main_app.py`, replace `_on_error` (lines 147-151):

```python
    def _on_error(self, msg):
        self.progress_bar.setVisible(False)
        self._set_action_buttons_enabled(True)
        self.log(f"오류: {msg}")
        # 다이얼로그에는 첫 번째 줄(예외 메시지)만 표시; 전체 트레이스백은 로그에 기록됨
        user_msg = msg.split('\n')[0][:300]
        QMessageBox.critical(self, "오류", user_msg)
```

- [ ] **Step 4: Run all Stage B tests**

```bash
python -m pytest tests/test_stage_b.py -v
```

Expected: all PASS

- [ ] **Step 5: Run full test suite**

```bash
python -m pytest tests/ -v --tb=short
```

Expected: all previous tests still PASS

- [ ] **Step 6: Commit**

```bash
git add main_app.py tests/test_stage_b.py
git commit -m "fix: _on_error 다이얼로그에 트레이스백 대신 예외 메시지 첫 줄만 표시"
```

---

## Self-Review

**Spec coverage check:**

| Issue | Task | Status |
|---|---|---|
| `setup_logging` Windows path (High) | Task 1 | ✓ |
| `raise CohortStepError from e` (Medium) | Task 2 | ✓ |
| `_load_data()` 0-row continues (High) | Task 3 | ✓ |
| `ORDER BY RANDOM()` no seed (High) | Task 4 | ✓ |
| `export_all` missing `sampling_info` (High) | Task 5 | ✓ |
| Single `export()` missing `sampling_info` | Task 5 | ✓ |
| Sampling Cancel ineffective post-analysis (High) | Task 6 | ✓ |
| `_on_error` full traceback in dialog (Medium) | Task 7 | ✓ |

**Placeholder scan:** No TBD, no "implement later", all code blocks complete.

**Type consistency:**
- `SamplingInfo.seed: int = 0` introduced in Task 4 Step 3b; used in Task 4 Steps 3c, 3d and Task 6 Step 3 (`seed=seed`) — consistent.
- `_confirm_sampling_if_needed()` returns `bool`; `start_analysis` uses `if not ...: return` — consistent.
- `_write_df_with_sampling_header(writer, df, sheet_name, sampling_info)` signature unchanged — consistent with Task 5.

**Dependency order:** Tasks are independent except Task 6 reads `SamplingInfo.seed` (defined in Task 4). Run Task 4 before Task 6, or run them in the same session in order.
