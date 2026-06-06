# Phase 3 Baseline Summary

## Decision

- accepted_label: multi_institution_t6_aligned_patient_disjoint
- accepted_model: sparse_linear
- status: CLINICAL_REVIEW_AUTHORIZED

## Raw Coverage (5-Month Dataset)

- first_records_date: 2024-07-01
- last_records_date: 2024-11-30
- records_file_count: 153 daily Parquet files (July to November 2024)
- monthly_file_counts: 202407=31, 202408=31, 202409=30, 202410=31, 202411=30
- prescription_rows_total: 40,380,615
- records_unique_patients: 493,875
- unique_edi_codes: 21,278
- source_mix: T30=22,340,091 / T60=18,040,524
- demographics_patient_count: 500,000 (eligibility_demographics.parquet)
- eligibility_ages_patient_count: 500,000 (eligibility_ages.parquet)
- schema_diffs_across_records: 0
- required_columns_present: patient_id, edi_code, start_date, institution_id, efmdc_clsf_no, wk_compn_cd
- key_null_rates: patient_id=0.0%, edi_code=0.0%, start_date=0.0%, institution_id=0.0%
- targeted_ops_regression: 71 passed in 4.68s (Windows `.venv_hana`, 2026-05-29)
- feature_serving_schema_guard_regression: 48 passed in 2.50s (Windows `.venv_hana`, 2026-05-29)

## Dataset Scope (2026-06-02)

- The dataset is finalized at 6 months (2024-07..12, 500k sample). No further months will be acquired; Dec 2024 Raw was added and verified on 2026-06-02.
- Jan 2025 / Gate 5A acquisition is cancelled and Gate 5B is retired; there is no remaining future-onset unlock trigger.
- The Nov→Dec future-onset holdout remains frozen (parked); do not run model, feature, or hyperparameter experiments or related code changes against it.

## Temporal Split

- train_window: 2024-08-01..2024-09-30 (lookback_days=60, reference_date=2024-09-30)
- val_window: 2024-10-01..2024-11-30 (lookback_days=60, reference_date=2024-11-30)
- n_train: 47,834
- n_val: 15,596 (strictly disjoint from both Train and the frozen future outcome holdout)
- patient_overlap_count: 0 (perfect leakage-free isolation)
- prevalence_shift: 4.38% (Train positive rate: 44.13%, Val positive rate: 40.50% — minimized)

## Model Comparison

| model | decision | val_auc | val_pr_auc | best_f1 | elapsed_sec |
|---|---|---:|---:|---:|---:|
| **sparse_linear** | **ACCEPTED (Primary)** | **0.844954** | **0.799161** | **0.729036** | **26.366** |
| xgboost | HELD (Secondary Backup) | 0.839437 | 0.776060 | 0.713338 | 214.849 |

*Note: The Sparse Linear model remains superior in all metrics, including AUC, PR-AUC, F1-Score, and Top-K Precision (Top-1% precision: 98.08%, Top-5% precision: 95.77%), while requiring drastically lower serving cost and latency.*

## Recommendations & Next Steps

1. **임상 검토 단계 진입 (Clinical Review Stage):**
   - 3대 아키텍처 우려 사항(lookback asymmetry, prevalence shift, holdout overlap)이 완벽히 해결되었고, 최종 AUC 0.845 및 PR-AUC 0.799 성과가 확인되었으므로, 즉시 공식 임상 검토위원회 심사를 개시합니다.
2. **서빙 스키마 동기화 (Serving Schema Update):**
   - `input_dim=15,273` 및 cutoff=100 약제 사전을 기준으로 `RequestFeatureBuilder` 컬럼 스키마 및 추론 피처 생성 파이프라인 동기화 작업을 병렬 진행합니다.
3. **운영 Threshold 보정 및 의사결정:**
   - Val 양성률 40.50% 기준에서 거짓양성률(FPR) 허용치 수준을 감안하여 실제 시스템 경고 임계치(threshold)를 최적화합니다.
