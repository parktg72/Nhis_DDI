# Raw SEX_TYPE Reporting Design

## Goal

Build DOCX sex summaries from original HANA `SEX_TYPE` values (`1=male`, `2=female`) instead of derived model feature `sex_m`, while preserving exact train-serving feature names and order.

## Data Contract

- `PatientFeatures.sex` remains the raw HANA string.
- `_patient_features_to_row()` carries it into feature rows as report metadata named `sex_type`.
- `sex_type` is never added to `FEATURE_COLS` or `RequestFeatureBuilder`.
- Every generic feature-selection path explicitly excludes `sex_type`.
- DOCX mapping is exact: `"1"` → male, `"2"` → female, all other/missing values → unknown.

## Report Behavior

`build_docx_bytes()` reads only `sex_type` for the sex summary. It does not reconstruct or fall back to `sex_m`.

- Raw column present: render male, female, and unknown rows.
- Raw column absent in historical feature data: render `⚠ 원본 성별 데이터 없음` and no inferred male/female counts.
- Both columns present but disagree: raw `sex_type` wins for reporting; `sex_m` remains model-only.
- Duplicate patients continue to use the existing patient-level deduplication before counts.

## Feature-Leakage Guard

Because some generic training paths select every column except known metadata, `sex_type` must be added to their metadata exclusion sets. The change does not alter `FEATURE_COLS`, serving feature order, model artifacts, or request schema.

## Persistence

New in-memory and saved feature DataFrames retain `sex_type` as metadata. Existing parquet files are not rewritten. Old files without `sex_type` receive the explicit no-source warning.

## Testing

Tests are written and observed failing before production edits.

- Row conversion preserves raw `sex_type` while `sex_m` keeps `1.0/0.0/0.5` model semantics.
- DOCX counts raw `1/2/invalid` correctly and ignores contradictory `sex_m`.
- DOCX without `sex_type` shows the no-source warning and does not infer counts.
- Generic train and feature selectors exclude `sex_type`.
- `FEATURE_COLS` names/order and serving contract remain unchanged.

## Non-Goals

- No HANA schema/query change.
- No retraining, artifact rewrite, parquet migration, or holdout evaluation.
- No `sex_m` removal; it remains the model feature.
- No commit or push unless separately requested.
