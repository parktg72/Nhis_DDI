# Phase 3 Baseline Summary

## Decision

- accepted_label: multi_institution_t6_exact30_patient_disjoint
- accepted_model: sparse_linear
- status: BASELINE_LOCKED

## Raw Coverage

- first_records_date: 2024-10-01
- last_records_date: 2024-11-30
- records_file_count: 61
- additional_month_available: False

## Temporal Split

- train_window: 2024-10-02..2024-10-31 (lookback_days=29, inclusive exact 30-day window)
- val_window: 2024-11-01..2024-11-30 (lookback_days=29, inclusive exact 30-day window)
- n_train: 67939
- n_val: 33879
- patient_overlap_count: 0

## Model Comparison

| model | decision | val_auc | val_pr_auc | best_f1 | elapsed_sec |
|---|---|---:|---:|---:|---:|
| sparse_linear | ACCEPTED | 0.84914 | 0.650496 | 0.612092 | 37.451 |
| xgboost_quick | HELD | 0.752892 | 0.490353 | 0.498509 | 106.53 |

## Future Outcome Track

- decision: WEAK_FEASIBLE_RESEARCH_TRACK
- baseline_replacement: False
- n_evaluable: 42168
- positive_rate_pct: 11.9593
- censoring_rate_pct: 13.9522
- next_unblock: acquire 2024-12 Raw month for 3-window temporal holdout

| model | val_auc | val_pr_auc | precision@top1% | precision@top5% | recall@top5% |
|---|---:|---:|---:|---:|---:|
| drug_only | 0.63438 | 0.188207 | 0.294118 | 0.253555 | 0.106046 |
| augmented_oct_inst_count | 0.64552 | 0.196427 | 0.247059 | 0.263033 | 0.11001 |

## Rejected Or Held Candidates

| candidate | decision | reason |
|---|---|---|
| sick_code_adr_proxy | REJECTED | Weak/noisy proxy label from earlier MLP smoke. |
| therapeutic_dup_t6 | SANITY_ONLY | Rule reconstruction from drug_code features; useful for pipeline sanity, not clinical prediction. |
| ddi_contraindicated | REJECTED | D000718(Metformin)+contrast-agent dominance and low D-code reachability; D000718 exclusion collapses positive rate below 1%. |
| xgboost_quick | HELD | Lower AUC/PR-AUC and higher cost on sparse multi-hot drug-code features. |

## Recommendations

- Use sparse_linear + multi_institution_t6_exact30_patient_disjoint as the Phase 3 proxy baseline.
- Acquire 2024-09 or 2024-12 Raw data for longer-gap temporal holdout before claiming generalization.
- Design a future clinical outcome label before clinical-risk claims.
- Revisit XGBoost only after dense embeddings or additional engineered temporal/institution features.
- Revisit DDI only if a direct EDI code to HIRA DUR D-code mapping table is available.
