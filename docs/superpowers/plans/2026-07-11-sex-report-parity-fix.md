# Sex Report Parity Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct unknown-sex encoding and boolean DOCX bucketing without changing feature names/order or protected model artifacts.

**Architecture:** Keep the existing `sex_m` feature and align the active batch/training producer with serving's `1.0/0.0/0.5` contract. At the report boundary, reject Python and NumPy booleans before numeric coercion so they remain unknown.

**Tech Stack:** Python 3.12, pandas, NumPy, pytest, python-docx

---

### Task 1: Lock the producer encoding contract

**Files:**
- Create: `tests/test_hana_app/test_ml_runner_sex_mapping.py`
- Modify: `hana_app/core/ml_runner.py:459`

- [ ] **Step 1: Write failing producer tests**

Create tests that construct `PatientFeatures` and call `_patient_features_to_row` directly. Assert `"1" → 1.0`, `"2" → 0.0`, and `None`, `""`, and `"9" → 0.5`. Assert values are floats and add an exact `FEATURE_COLS` names/order snapshot ending in `"age", "sex_m"`.

```python
@pytest.mark.parametrize(
    ("raw_sex", "expected"),
    [("1", 1.0), ("2", 0.0), (None, 0.5), ("", 0.5), ("9", 0.5)],
)
def test_patient_features_sex_mapping(raw_sex, expected):
    row = _patient_features_to_row(_make(sex=raw_sex))
    assert row["sex_m"] == expected
    assert isinstance(row["sex_m"], float)
```

- [ ] **Step 2: Verify RED**

Run:

```bash
python3.12 -m pytest tests/test_hana_app/test_ml_runner_sex_mapping.py -v
```

Expected: cases for `None`, `""`, and `"9"` fail because current code returns `0`; float-type cases fail because current known values are integers.

- [ ] **Step 3: Implement the minimal tri-state mapping**

In `_patient_features_to_row`, replace the binary expression with:

```python
"sex_m": 1.0 if f.sex == "1" else (0.0 if f.sex == "2" else 0.5),
```

Do not modify `FEATURE_COLS`, serving, eligibility loading, legacy `sex_male`, or monitoring.

- [ ] **Step 4: Verify GREEN and local parity**

```bash
python3.12 -m pytest tests/test_hana_app/test_ml_runner_sex_mapping.py tests/test_hana_app/test_ml_runner_validation.py tests/test_serving/test_feature_contract.py -v
```

Expected: all pass; feature names and order remain unchanged.

### Task 2: Reject boolean sex values in DOCX summaries

**Files:**
- Modify: `tests/test_hana_app/test_report_exporter.py`
- Modify: `hana_app/core/report_exporter.py:1209`

- [ ] **Step 1: Write failing report tests**

Add one parametrized test that builds a report from Python `True/False` and NumPy `np.bool_(True/False)`. For every pair assert male `0`, female `0`, and unknown `2`.

```python
@pytest.mark.parametrize(
    "values_in",
    [[True, False], [np.bool_(True), np.bool_(False)]],
)
def test_docx_analysis_subject_treats_boolean_sex_as_unknown(values_in):
    # Build two patient rows, parse section 7, and assert both are unknown.
```

- [ ] **Step 2: Verify RED**

```bash
python3.12 -m pytest tests/test_hana_app/test_report_exporter.py::test_docx_analysis_subject_treats_boolean_sex_as_unknown -v
```

Expected: fail because `True == 1` and `False == 0` place values in male/female buckets.

- [ ] **Step 3: Implement the minimal boolean mask**

Add `import numpy as np`. Before `pd.to_numeric`, mask Python and NumPy booleans:

```python
_sex_raw = analysis_subject_df["sex_m"]
_sex_raw = _sex_raw.mask(_sex_raw.map(lambda value: isinstance(value, (bool, np.bool_))))
_sex_vals = pd.to_numeric(_sex_raw, errors="coerce")
```

Keep exact numeric/string `0` and `1` behavior unchanged.

- [ ] **Step 4: Verify GREEN**

```bash
python3.12 -m pytest tests/test_hana_app/test_report_exporter.py -v
```

Expected: all report exporter tests pass, including fractional, NaN, raw `2`, string, all-unknown, and boolean cases.

### Task 3: Verify train-serving parity and repository safety

**Files:**
- Verify only; do not touch protected artifacts.

- [ ] **Step 1: Run focused suites under Windows Python 3.12**

```powershell
& .\.venv_hana\Scripts\python.exe -m pytest tests\test_hana_app tests\test_serving tests\test_features -q
```

Expected: all pass. Existing sklearn warnings are acceptable only if unchanged.

- [ ] **Step 2: Run closest additional ETL coverage**

```powershell
& .\.venv_hana\Scripts\python.exe -m pytest tests\test_etl -q
```

Expected: all pass without generated parquet or artifact writes.

- [ ] **Step 3: Check diagnostics and diff scope**

Run `py_compile` on both production files and both test files, then `git diff --check`. Confirm only the approved spec/plan plus `ml_runner.py`, `report_exporter.py`, and their two tests changed; leave pre-existing agent/graph dirty files untouched.

- [ ] **Step 4: Cross-family review**

Request independent Claude-family and OpenAI-family read-only review of the exact diff. Both must confirm feature names/order, serving alignment, report behavior, and protected/frozen-track safety before completion is reported.

## Commit Strategy

Do not commit or push unless explicitly requested. If later requested, use two atomic commits: producer mapping with its direct test, then report boolean handling with its direct test. Keep design/plan documentation separate from code commits if the user wants documentation committed.
