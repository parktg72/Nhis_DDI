# Phase 3 Next Session Checklist

Last updated: 2026-05-29

## Current State

- Baseline status: `CLINICAL_REVIEW_AUTHORIZED`
- Accepted label: `multi_institution_t6_aligned_patient_disjoint`
- Accepted model: `sparse_linear`
- Baseline reference result: 60-day aligned perfect patient-disjoint temporal AUC `0.844954`, PR-AUC `0.799161`
- Raw coverage currently available: `2024-07-01..2024-11-30` (5 months sampling data)
- Next Step Blocker: Bypassed. Transitioning from research track to official clinical review.
- Repo handoff state: DAG, Smoke DL serving, packaging, hana_app, and cleanup
  slices are committed and pushed.

Decision documents:

- `docs/reports/phase3_baseline_summary.md`
- Obsidian: `mode_11_hana.md` (and related logs)


## 2026-05-28 Gate Correction And Raw Data Verification

### FROZEN — Nov->Dec Holdout Remains Exhausted

Do not run additional model, feature, or hyperparameter experiments against
the current Nov->Dec patient-disjoint holdout. The holdout is frozen because it
has already been used for repeated ablations and XGBoost robustness checks.
Per the 2026-06-02 correction: the dataset is finalized at 6 months
(2024-07..12); **Jan 2025 / Gate 5A acquisition is cancelled and Gate 5B is
retired. There is no remaining future-onset unlock condition — the holdout is
parked.**

Freeze references:

- ledger: `data/reports/future_onset_research_freeze_ledger.json`
- handoff: `data/reports/future_onset_research_freeze_handoff.md`
- split manifest: `data/reports/future_onset_research_freeze_datasplit_manifest.csv`
- frozen model: `data/models/future_onset_xgb_efmdc_demo_frozen_20260526.ubj`

### 2024-07..11 Raw Data Verification (Hermes confirmed)

- Total record files: `153` (`records_20240701.parquet` .. `records_20241130.parquet`)
- Monthly counts: `202407=31`, `202408=31`, `202409=30`, `202410=31`, `202411=30`
- Total prescription rows: `40,380,615`
- Unique patients in records: `493,875`; unique EDI codes: `21,278`
- Source mix: `T30=22,340,091` / `T60=18,040,524`
- Eligibility files: `eligibility_ages.parquet=500,000 rows`, `eligibility_demographics.parquet=500,000 rows`
- Schema diffs across records files: `0`
- Required columns present in all records files: `patient_id`, `edi_code`, `start_date`, `institution_id`, `efmdc_clsf_no`, `wk_compn_cd`
- Null rates for `patient_id`, `edi_code`, `start_date`, `institution_id`: `0.0%`
- Targeted ops regression: `71 passed in 4.68s` (Windows `.venv_hana`, 2026-05-29)
- Feature/serving schema guard regression: `48 passed in 2.50s` (Windows `.venv_hana`, 2026-05-29)

Status: `2024-07..11` data is established training/context/clinical-review
baseline data. It is **not** a frozen Nov->Dec holdout unlock condition.

Dataset scope (2026-06-02) — **Jan 2025 / Gate 5A CANCELLED**:

The dataset is finalized at 6 months (2024-07..12, 500k sample). Dec 2024 Raw was
added and verified on 2026-06-02 (31 files, schema/null QA clean). **Jan 2025 Raw
will not be acquired**, so the former Gate 5A unlock trigger is cancelled and
Gate 5B is retired. There is no remaining future-onset unlock condition.

The Nov→Dec future-onset holdout remains frozen (parked). Do not run model,
feature, or hyperparameter experiments or related code changes against it.
Proceed only with freeze-safe same-window baseline and DL operationalization work.

### Gate 1 — Schema And Integrity

- `records_20241130.parquet` rows: 351,697
- `records_20241231.parquet` rows: 801,730
- Missing/new Dec columns vs Nov: none
- Required missing Dec columns: none
- Null rates for `patient_id`, `edi_code`, `start_date`, `institution_id`: 0.0% in both Nov and Dec

### Gate 2 — Dec Same-Window Dataset

- dataset: `data/datasets/multi_inst_t6_20241231_l29`
- n_patients: 85,907
- input_dim: 14,705
- positive_rate_pct: 25.0364
- unknown_drug_rate_pct: 0.6499
- zero_vector_rate_pct: 0.0
- note: Dec row/patient counts are about 2x Nov; schema/null/vocab checks are normal, so record as likely year-end prescribing concentration.

### Gate 3 — Oct→Dec Same-Window Temporal Smoke

- report: `data/datasets/multi_inst_t6_temporal_20241031_to_20241231/sparse_training_smoke_report.json`
- val_auc: 0.843441
- val_pr_auc: 0.674430
- best_f1: 0.625442
- precision@top5%: 0.873371
- patient_overlap_count: 10,871
- patient_overlap_val_rate_pct: 12.6544
- decision: metric pass, but do not claim patient-disjoint generalization from this Dec smoke because overlap is nonzero.

Patient-disjoint rerun:

- filtered validation dataset: `data/datasets/multi_inst_t6_20241231_l29_disjoint_oct`
- removed overlap patients: 10,871 (12.6544% of Dec source)
- filtered positive_rate_pct: 24.6015
- report: `data/datasets/multi_inst_t6_temporal_20241031_to_20241231_disjoint_oct/sparse_training_smoke_report.json`
- val_auc: 0.839296
- val_pr_auc: 0.663741
- precision@top5%: 0.866205
- patient_overlap_val_rate_pct: 0.0
- decision: same-window sparse-linear signal remains robust under patient-disjoint Dec validation.

### Gate 4 — Future Outcome 3-Window Track

- validation dataset: `data/datasets/future_multi_inst_onset_t6_20241130_to_20241231_with_oct_inst_count`
- temporal report: `data/datasets/future_multi_inst_onset_t6_temporal_20241031_to_20241231/sparse_training_smoke_report.json`
- n_val_dataset: 25,749
- val_positive_rate_pct: 13.2627
- censoring_rate_pct: 13.2556
- input_dim: 14,706 (`drug_vocab` + prior institution count scalar)
- val_auc: 0.647111
- val_pr_auc: 0.211640
- precision@top1%: 0.325581
- precision@top5%: 0.287267
- patient_overlap_val_rate_pct: 14.7889
- decision: not confirmed for generalization (`AUC < 0.70`), but top-K precision remains above prevalence; keep as research track only.
- next feature directions: medication class features, demographics, trend/delta features, and institution utilization trajectory.

Patient-disjoint rerun:

- filtered validation dataset: `data/datasets/future_mi_t6_20241231_disjoint_octnov`
- removed overlap patients: 3,808 (14.7889% of Nov->Dec source)
- filtered positive_rate_pct: 14.0787
- excluded overlap positive_rate_pct: 8.5609
- report: `data/datasets/future_mi_t6_temporal_20241031_to_20241231_disjoint/sparse_training_smoke_report.json`
- val_auc: 0.630412
- val_pr_auc: 0.215572
- precision@top5%: 0.297814
- patient_overlap_val_rate_pct: 0.0
- decision: future outcome remains not confirmed. The lower disjoint AUC strengthens the research-only conclusion.
- asymmetry note: same-window overlap patients have higher positive rate, but future-onset overlap patients have lower new-onset rate; this is consistent with persistence suppressing marginal new onset.

Demographics v1 rerun:

- feature source: `data/Raw/eligibility_demographics.parquet`
- added features: `age_years_div_100`, `sex_type_1_flag`
- train dataset: `data/datasets/future_mi_t6_20241031_to_20241130_with_inst_demo`
- patient-disjoint validation dataset: `data/datasets/future_mi_t6_20241130_to_20241231_with_inst_demo_disjoint_octnov`
- input_dim: 14,708 (`drug_vocab` + prior institution count + 2 demographics scalars)
- demographics_missing_patient_rate_pct: 0.0
- report: `data/datasets/future_mi_t6_temporal_20241031_to_20241231_disjoint_inst_demo/sparse_training_smoke_report.json`
- val_auc: 0.633293
- val_pr_auc: 0.216363
- precision@top1%: 0.322727
- precision@top5%: 0.292350
- recall@top5%: 0.103917
- patient_overlap_val_rate_pct: 0.0
- decision: demographics add only a tiny AUC gain vs patient-disjoint drug+institution baseline (`0.633293` vs `0.630412`) and do not meet the `0.64` marginal-signal threshold. Move next to medication class features before any clinical claim.

Medication-class v1 rerun:

- feature source: `efmdc_clsf_no` from feature-window Raw records
- vocab strategy: train feature-window vocab with `__NULL_EFMDC__` and `__UNK_EFMDC__`; validation reuses train vocab
- train dataset: `data/datasets/future_mi_t6_20241031_to_20241130_with_inst_efmdc`
- patient-disjoint validation dataset: `data/datasets/future_mi_t6_20241130_to_20241231_with_inst_efmdc_disjoint_octnov`
- class vocab size: 116 (`114` nonblank train classes + null + unknown)
- input_dim: 14,822 (`drug_vocab` + prior institution count + 116 medication class columns)
- train class null row rate: 39.7552% within evaluable kept rows
- validation class null row rate: 38.1858% before patient-disjoint filtering
- validation class OOV row rate: 0.0%
- report: `data/datasets/future_mi_t6_temporal_20241031_to_20241231_disjoint_inst_efmdc/sparse_training_smoke_report.json`
- val_auc: 0.642860
- val_pr_auc: 0.226443
- precision@top1%: 0.345455
- precision@top5%: 0.303279
- recall@top5%: 0.107802
- patient_overlap_val_rate_pct: 0.0
- decision: medication class is the first richer feature with material patient-disjoint gain vs drug+institution baseline (`AUC +0.012448`, `PR-AUC +0.010871`, top5 precision +0.005465), but it remains below the `0.70` continuation threshold. Keep future-onset research-only.

Medication-class + demographics ablation:

- train dataset: `data/datasets/future_mi_t6_20241031_to_20241130_with_inst_efmdc_demo`
- patient-disjoint validation dataset: `data/datasets/future_mi_t6_20241130_to_20241231_with_inst_efmdc_demo_disjoint_octnov`
- input_dim: 14,824 (`drug_vocab` + prior institution count + 2 demographics scalars + 116 medication class columns)
- report: `data/datasets/future_mi_t6_temporal_20241031_to_20241231_disjoint_inst_efmdc_demo/sparse_training_smoke_report.json`
- val_auc: 0.645007
- val_pr_auc: 0.227231
- precision@top1%: 0.345455
- precision@top5%: 0.310565
- recall@top5%: 0.110392
- patient_overlap_val_rate_pct: 0.0
- decision: current best sparse-linear future-onset research feature bundle. It improves over class-only (`AUC +0.002147`, top5 precision +0.007286), but still remains below `AUC=0.70`; do not promote to baseline or clinical claim.

XGBoost research-only comparison:

- feature bundle: drug multi-hot + prior institution count + medication class + demographics
- train dataset: `data/datasets/future_mi_t6_20241031_to_20241130_with_inst_efmdc_demo`
- patient-disjoint validation dataset: `data/datasets/future_mi_t6_20241130_to_20241231_with_inst_efmdc_demo_disjoint_octnov`
- quick pilot report: `data/datasets/future_mi_t6_temporal_20241031_to_20241231_disjoint_inst_efmdc_demo_xgb50_d4/sparse_training_smoke_report.json`
- quick pilot result: AUC 0.679427, PR-AUC 0.250919, precision@top5% 0.326047, elapsed 71.841 sec
- full report: `data/datasets/future_mi_t6_temporal_20241031_to_20241231_disjoint_inst_efmdc_demo_xgb300_d6/sparse_training_smoke_report.json`
- full config: `n_estimators=300`, `max_depth=6`, `early_stopping_rounds=20`, `n_estimators_used=83`
- val_auc: 0.680783
- val_pr_auc: 0.253017
- precision@top1%: 0.445455
- precision@top5%: 0.333333
- recall@top5%: 0.118485
- patient_overlap_val_rate_pct: 0.0
- elapsed_sec: 141.192
- decision: XGBoost breaks the sparse-linear ceiling (`AUC +0.035776` vs best linear class+demo), but it still misses the `AUC >= 0.70` milestone. Keep future-onset as research-only and do not promote to clinical or baseline status.
- research state: drug+institution+efmdc_class+demographics with XGBoost depth 6, early stopped at 83 trees, reached AUC 0.681 and precision@top1 0.445 (about 3.2x random prevalence) on patient-disjoint Nov->Dec holdout; research track only.

XGBoost seed sensitivity:

- config: `xgb50_depth4`, same fixed class+demographics train/validation datasets
- seeds: 7, 42, 99
- report summary: `data/reports/xgb_seed_sensitivity_summary.json`
- auc values: 0.678496, 0.679427, 0.681903
- auc_mean: 0.679942
- auc_std: 0.001438
- auc_min: 0.678496
- pr_auc_mean: 0.250639
- precision@top5_mean: 0.334244
- decision: robustly above sparse-linear ceiling because `min_auc >= 0.665` and `auc_std < 0.010`. Still below the `0.70` milestone, so keep research-only.

Future-onset research freeze:

- status: `RESEARCH_TRACK_FROZEN`
- freeze date: 2026-05-26
- reason: Dec patient-disjoint holdout has been used for multiple ablations; further tuning on this holdout would create implicit validation overfit.
- frozen model: `data/models/future_onset_xgb_efmdc_demo_frozen_20260526.ubj`
- frozen model sha256: `118bc77ce5c0023bbd59f11b029736dbab8177104ed9b788353d00f68daa4458`
- freeze ledger: `data/reports/future_onset_research_freeze_ledger.json`
- datasplit manifest: `data/reports/future_onset_research_freeze_datasplit_manifest.csv`
- handoff README: `data/reports/future_onset_research_freeze_handoff.md`
- guardrail: do not run model, feature, or hyperparameter experiments or related code changes against the same Nov→Dec holdout.
- allowed next step: none for future-onset — dataset finalized at 6 months (2024-07..12); Jan 2025 / Gate 5A acquisition cancelled and Gate 5B retired. The holdout is parked; proceed only with freeze-safe same-window / DL work.

## Future-onset Track: Parked (Jan 2025 / Gate 5A Cancelled, 2026-06-02)

The former "Resume When Gate 5A Arrives" protocol is cancelled. The dataset is
finalized at 6 months (2024-07..12); Jan 2025 Raw will not be acquired, so there
is no Dec→Jan new unseen holdout and no future-onset unlock trigger. Gate 5A and
Gate 5B are both retired.

The Nov→Dec future-onset holdout remains frozen (parked) because it was already
used for repeated ablations. Do not run model, feature, or hyperparameter
experiments or related code changes against it. Future-onset research is parked
indefinitely; proceed only with freeze-safe same-window baseline and DL
operationalization work.

## Do Not Do

- Do not reintroduce Gate 5A/5B or a Jan 2025 acquisition; the dataset is finalized at 6 months (2024-07..12) and the future-onset track is parked.
- Do not run model, feature, or hyperparameter experiments or related code changes against the frozen Nov→Dec holdout.
- Do not claim future-onset clinical generalization from the frozen Nov->Dec research track.
- Do not rebuild vocabulary silently; if vocab changes, record `input_dim`, SHA256, and reason.
- Do not change serving feature schema unless the training feature schema and serving `RequestFeatureBuilder` are updated and tested together.
