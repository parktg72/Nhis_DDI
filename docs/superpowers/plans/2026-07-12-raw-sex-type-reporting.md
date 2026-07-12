# Raw SEX_TYPE Reporting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve raw HANA `SEX_TYPE` as report metadata, build DOCX sex summaries only from that raw field, and prevent metadata leakage into model features.

**Architecture:** `_patient_features_to_row()` emits both model feature `sex_m` and report metadata `sex_type`. Explicit metadata exclusion sets block `sex_type` from generic selection/training. DOCX maps exact raw strings `"1"/"2"` and never falls back to `sex_m`.

**Tech Stack:** Python 3.12, pandas, NumPy, pytest, python-docx

---

### Task 1: Preserve raw metadata and block feature leakage

**Files:**
- Modify: `hana_app/core/ml_runner.py`
- Modify: `scripts/features/selector.py`
- Modify: `scripts/features/normalizer.py`
- Modify: `scripts/train/dataset.py`
- Modify: `tests/test_hana_app/test_ml_runner_sex_mapping.py`
- Add: `tests/test_features/test_sex_type_exclusion.py`
- Add: `tests/test_train/test_sex_type_exclusion.py`

- [ ] **Step 1: Write failing propagation tests**

Extend the row-mapping test so every raw value is preserved while model semantics remain unchanged:

```python
@pytest.mark.parametrize("raw_sex", ["1", "2", None, "", "9"])
def test_patient_features_row_carries_raw_sex_type(raw_sex):
    row = _patient_features_to_row(_make(sex=raw_sex))
    assert row["sex_type"] == raw_sex

def test_feature_cols_excludes_sex_type():
    assert "sex_type" not in FEATURE_COLS
```

- [ ] **Step 2: Write failing leakage tests**

Assert `sex_type` belongs to selector/training/normalizer metadata exclusions and never appears in selected model features. Use real `FeatureSelector` and dataset construction APIs already present in their respective test suites.

```python
def test_metadata_sets_exclude_raw_sex_type():
    assert "sex_type" in META_COLS
    assert "sex_type" in CATEGORICAL_COLS
    assert "sex_type" in NON_FEATURE_COLS
```

- [ ] **Step 3: Verify RED**

```bash
python3.12 -m pytest tests/test_hana_app/test_ml_runner_sex_mapping.py tests/test_features/test_sex_type_exclusion.py tests/test_train/test_sex_type_exclusion.py -v
```

Expected: raw row lacks `sex_type`; exclusion assertions fail; generic candidate/dataset test includes `sex_type`.

- [ ] **Step 4: Implement minimal propagation and exclusions**

Add report metadata beside the existing tri-state model feature:

```python
"sex_m": 1.0 if f.sex == "1" else (0.0 if f.sex == "2" else 0.5),
"sex_type": f.sex,
```

Add `"sex_type"` to `META_COLS`, `CATEGORICAL_COLS`, and `NON_FEATURE_COLS`. Do not alter `FEATURE_COLS`, serving, or `RequestFeatureBuilder`.

- [ ] **Step 5: Verify GREEN**

```bash
python3.12 -m pytest tests/test_hana_app/test_ml_runner_sex_mapping.py tests/test_features/test_sex_type_exclusion.py tests/test_train/test_sex_type_exclusion.py tests/test_serving/test_feature_contract.py tests/test_features/test_features.py -v
```

Expected: all pass; exact model feature names/order unchanged.

### Task 2: Switch DOCX summary to raw SEX_TYPE only

**Files:**
- Modify: `hana_app/core/report_exporter.py`
- Modify: `tests/test_hana_app/test_report_exporter.py`

- [ ] **Step 1: Write failing raw-report tests**

Add real DOCX tests for:

```python
sex_type=["1", "2", "9"]  # male 1, female 1, unknown 1
```

Add disagreement test where `sex_type="2"` and `sex_m=1.0`; expected report is female. Add absent-column test expecting `⚠ 원본 성별 데이터 없음` and no sex count rows. Add invalid/missing raw values test expecting unknown.

- [ ] **Step 2: Migrate existing report fixtures**

Existing deduplication/female/unknown report tests must provide raw strings `sex_type="1"` or `"2"`. Remove obsolete bool-in-`sex_m` test and production NumPy bool mask because report no longer reads `sex_m`.

- [ ] **Step 3: Verify RED**

```bash
python3.12 -m pytest tests/test_hana_app/test_report_exporter.py -v -k "sex or analysis_subject"
```

Expected: current implementation still reads `sex_m`, ignores `sex_type`, and fails no-source/disagreement assertions.

- [ ] **Step 4: Implement strict raw mapping**

Replace the sex summary source with exact normalized raw strings:

```python
if "sex_type" in analysis_subject_df.columns and total_n > 0:
    sex_values = analysis_subject_df["sex_type"].astype("string").str.strip()
    male_n = int(sex_values.eq("1").sum())
    female_n = int(sex_values.eq("2").sum())
    unknown_n = total_n - male_n - female_n
    # emit 남/여/미상 rows and existing low-male warning
elif total_n > 0:
    target_rows.append([
        "⚠ 원본 성별 데이터 없음",
        "sex_type 컬럼이 없어 성별을 추정하지 않습니다. 원본 자격DB 데이터로 피처를 다시 생성하세요.",
    ])
```

Never read or fall back to `sex_m` in this report block.

- [ ] **Step 5: Verify GREEN**

```bash
python3.12 -m pytest tests/test_hana_app/test_report_exporter.py -v
```

Expected: all report tests pass; old data gets warning, not inferred counts.

### Task 3: Final parity and safety verification

- [ ] **Step 1: Run Windows Python 3.12 suites**

```powershell
& .\.venv_hana\Scripts\python.exe -m pytest tests\test_hana_app tests\test_serving tests\test_features tests\test_train -q
& .\.venv_hana\Scripts\python.exe -m pytest tests\test_etl -q
```

- [ ] **Step 2: Compile and inspect scope**

Run `py_compile` for changed Python files, `git diff --check`, and verify no changes under `mlruns/`, generated parquet, `out/`, or `packages_win/py312/`.

- [ ] **Step 3: Cross-family read-only review**

Require independent OpenAI-family and Google/Anthropic-family PASS on raw reporting, leakage guards, exact `FEATURE_COLS`, and protected/frozen-track safety.

## Commit Strategy

Do not commit or push unless requested. If later requested: one atomic commit for raw metadata propagation + exclusions + direct tests, then one atomic commit for DOCX raw reporting + direct tests. Keep spec/plan docs in a separate documentation commit if requested.
