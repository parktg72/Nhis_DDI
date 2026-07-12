# Sex Report Parity Fix Design

## Goal

Correct two confirmed sex-reporting defects without changing model feature names, feature order, serving request schema, or protected model artifacts.

## Scope

1. Align the active training/batch producer with the existing serving contract:
   - raw `"1"` → `sex_m=1.0` (male)
   - raw `"2"` → `sex_m=0.0` (female)
   - missing, empty, or invalid raw sex → `sex_m=0.5` (unknown)
2. Prevent Python and NumPy boolean values from being interpreted as valid numeric sex codes in DOCX summaries.
3. Add regression coverage for producer encoding, report boolean handling, and unchanged feature schema/order.

## Architecture

The fix stays at the two existing conversion boundaries. `hana_app/core/ml_runner.py::_patient_features_to_row` will emit the same tri-state `sex_m` values already used by `serving/predictor.py`; no column is added, removed, or reordered. `hana_app/core/report_exporter.py::build_docx_bytes` will mask boolean inputs before numeric coercion so only explicit numeric/string representations of 0 and 1 enter female/male buckets.

No metadata sidecar is introduced because the existing `sex_m=0.5` sentinel preserves unknown state without schema drift. The legacy `sex_male` pipeline and monitoring convention are out of scope.

## Data Flow

```text
HANA-confirmed SEX_TYPE semantics ("1" male, "2" female, other unknown)
  → PatientFeatures.sex
  → _patient_features_to_row tri-state sex_m (1.0 / 0.0 / 0.5)
  → unchanged FEATURE_COLS ordering
  → model training/report DataFrame
  → DOCX exact buckets (1 male, 0 female, other unknown; bool rejected)
```

## Error Handling

- Missing and invalid raw values become the established neutral sentinel `0.5`, not female.
- Boolean values are treated as unknown even though Python equality considers `True == 1` and `False == 0`.
- Missing `sex_m` columns and empty cohorts retain current behavior.

## Testing

Tests are written and observed failing before production edits.

- Producer test: `"1"`, `"2"`, `None`, `""`, and invalid values map to `1.0`, `0.0`, and `0.5` as specified.
- Report test: Python/NumPy booleans count as unknown.
- Schema regression: `FEATURE_COLS` names and order remain unchanged.
- Verification: relevant `tests/test_hana_app`, `tests/test_serving`, and `tests/test_features` under Windows Python 3.12; no training or protected-artifact writes.

## Non-Goals

- No model retraining, `mlruns/` changes, generated parquet writes, or holdout evaluation.
- No changes to `RequestFeatureBuilder`, serving request fields, feature names/order, legacy `sex_male`, or monitoring conventions.
- No commit or push unless separately requested.
