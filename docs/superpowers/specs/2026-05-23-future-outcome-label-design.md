# Future Outcome Label 설계 문서

**Goal:** 기존 same-window proxy baseline을 넘어서, 10월 처방 feature로 11월 outcome을 예측하는 feature/label window 분리 태스크를 정의한다.

**Scope:** 이 문서는 label/dataset 설계 문서다. 구현 계획이나 학습 튜닝 계획은 별도 문서에서 다룬다.

**Baseline Context:** Phase 3 baseline은 `multi_institution_t6_exact30_patient_disjoint` + `sparse_linear`로 lock 완료됐다. 최초 설계 당시 Raw는 `2024-10-01..2024-11-30`만 존재했으나, 2026-05-26 현재 `2024-12` Raw가 확보되어 `Nov -> Dec` temporal holdout 구축이 가능하다.

---

## 1. 설계 원칙

### Feature/Label Window 분리

기존 baseline은 같은 window 안의 처방 feature와 같은 window 안의 proxy label을 비교하는 구조였다. Future outcome label은 아래처럼 관측 window와 outcome window를 분리한다.

| 구분 | 기간 | 역할 |
|---|---|---|
| Observation window | 2024-10-02..2024-10-31 | drug multi-hot feature 생성 |
| Outcome window | 2024-11-01..2024-11-30 | future label 생성 |
| Feature reference date | 2024-10-31 | 10월 feature cohort 기준일 |
| Outcome reference date | 2024-11-30 | 11월 outcome 산출 기준일 |

`lookback_days=29`는 양 끝 날짜를 모두 포함하므로 exact 30-day window다.

### 일반화 주장 제한

2024-12 Raw 확보 전에는 `Oct -> Nov`로 학습한 모델을 별도 미래 월에서 검증할 수 없었다. 이제 `Nov -> Dec` pair를 추가로 구축해 temporal holdout을 실행할 수 있다.

프로덕션 또는 임상적 성능 주장은 `Nov -> Dec` temporal holdout을 생성하고 성능/환자 overlap/censoring shift를 검토한 뒤에만 가능하다.

---

## 2. Cohort 정의

### Feature Cohort

기본 cohort는 `records_20241031.parquet` 기준 active patient다.

조건:

- `patient_id`가 2024-10-31 reference file에 존재
- 2024-10-02..2024-10-31 observation window에 처방 history가 1건 이상 존재 (`oct_history_rows >= 1`)
- drug feature vocab은 Phase 3 baseline과 동일한 vocab을 사용

`oct_history_rows == 0`인 환자는 10월 baseline 상태를 확인할 수 없으므로 strict onset 태스크에서 제외한다.

### Outcome Evaluable Cohort

Outcome label은 11월 기록이 관측 가능한 환자에게만 부여한다.

조건:

- feature cohort에 포함
- 2024-11-01..2024-11-30 outcome window에 청구/처방 history가 1건 이상 존재

11월 기록이 없는 환자는 negative가 아니라 **censored**로 처리하고 학습/평가에서 제외한다.

### Censoring 규칙

| 상태 | 처리 | 이유 |
|---|---|---|
| Nov history >= 1 and Nov institution_count < T | negative | 11월 관측 가능, threshold 미달 |
| Nov history >= 1 and Nov institution_count >= T | positive candidate | 11월 관측 가능, threshold 도달 |
| Nov history = 0 | censored/excluded | 사망, 전원, 처방 없음, 시스템 이탈 등을 구분할 수 없음 |

Nov 기록이 없는 환자를 default negative로 두면 informative censoring bias가 생길 수 있으므로 금지한다.

---

## 3. Primary Outcome

### `nov_multi_institution_new_onset_t6`

Primary label은 10월에는 threshold 미달이었으나 11월에 threshold에 도달한 환자를 positive로 정의한다.

```text
positive = (oct_institution_count < 6) AND (nov_institution_count >= 6)
negative = (oct_institution_count < 6) AND (nov_institution_count < 6) AND nov_history_rows >= 1
excluded = (oct_institution_count >= 6) OR (nov_history_rows == 0)
```

이 정의는 “이미 10월에 고위험이던 환자가 11월에도 고위험인 persistence”를 학습하지 않도록 한다. 목표 질문은 다음과 같다.

> 10월에는 threshold 미달이던 환자 중, 11월에 다기관 고위험 상태로 전환될 환자를 10월 처방 패턴만으로 식별할 수 있는가?

`oct_institution_count >= 6` 환자는 predictive task에서는 제외하지만 버리지 않는다. 이들은 별도 `persistence_cohort`로 audit report에 기록한다.

```text
persistence_cohort = oct_institution_count >= 6 AND nov_history_rows >= 1
persistence_rate = count(oct_institution_count >= 6 AND nov_institution_count >= 6) / persistence_cohort
```

이 메트릭은 “이미 고위험인 환자가 다음 달에도 고위험으로 남는 비율”을 설명하기 위한 descriptive statistic이며, primary training label에 섞지 않는다.

### Threshold Sensitivity

Primary threshold는 Phase 3 baseline과 일관되게 `T=6`으로 둔다. 민감도 분석은 `T=5`, `T=7`을 포함한다.

| threshold | 목적 |
|---:|---|
| 5 | 양성률 증가 시 signal 안정성 확인 |
| 6 | baseline-aligned primary |
| 7 | 더 엄격한 escalation outcome 확인 |

구현은 threshold `T`를 파라미터로 받아야 한다. 파일명과 metadata에는 실제 threshold를 명시한다. 예: `future_multi_inst_onset_t6`.

---

## 4. Secondary / Exploratory Outcomes

### Secondary: `nov_multi_institution_prevalence_t6`

```text
positive = nov_institution_count >= 6
negative = nov_institution_count < 6 AND nov_history_rows >= 1
excluded = nov_history_rows == 0
```

이 label은 prevalence 측정과 sensitivity 용도로만 사용한다. `institution_count`는 persistence가 강하기 때문에, 모델이 “10월 고이용 환자는 11월에도 고이용”이라는 상태 지속성을 학습할 위험이 크다. Primary training target으로 사용하지 않는다.

### Exploratory: `nov_institution_surge_k3`

```text
positive = (nov_institution_count - oct_institution_count) >= 3
negative = (nov_institution_count - oct_institution_count) < 3 AND nov_history_rows >= 1
excluded = nov_history_rows == 0
```

이 outcome은 threshold와 독립적인 급증 신호를 잡기 위한 탐색용이다. `k=3`을 기본으로 두고, 양성률이 너무 낮으면 `k=2` sensitivity를 검토한다.

### Slice: Clean Onset vs Escalation

Primary positive는 두 slice로 나누어 리포트한다.

| slice | 정의 | 의미 |
|---|---|---|
| clean_onset | `oct_institution_count == 0 AND nov_institution_count >= 6` | 10월 다기관 신호가 없다가 11월에 발생 |
| escalation | `0 < oct_institution_count < 6 AND nov_institution_count >= 6` | 기존 이용이 있었으나 threshold 이상으로 상승 |

두 slice의 비율과 성능을 별도 리포트에 포함한다.

---

## 5. Feature 설계

### Required Features

초기 구현은 Phase 3 baseline과 같은 feature schema를 사용한다.

- sparse CSR drug multi-hot
- input_dim: 14,705
- vocab cutoff: 100
- `_unk` token 유지
- feature source: 2024-10-02..2024-10-31 observation window의 `drug_code`

### Optional Audit Features

초기 학습 feature에는 넣지 않지만 audit report에는 포함한다.

- `oct_institution_count`
- `nov_institution_count`
- `oct_history_rows`
- `nov_history_rows`
- censoring count/rate
- clean_onset/escalation counts

`oct_institution_count`를 학습 feature로 넣을지는 별도 실험에서 결정한다. Primary label의 정의 자체가 `oct_institution_count < T`로 cohort를 제한하므로, 첫 구현에서는 drug multi-hot만 사용해 기존 baseline과 비교 가능성을 유지한다.

---

## 6. Data Flow

1. Load feature cohort from `records_20241031.parquet`.
2. Build observation histories from `2024-10-02..2024-10-31`.
3. Build outcome histories from `2024-11-01..2024-11-30`.
4. Compute `oct_institution_count` and `nov_institution_count` for each feature cohort patient.
5. Mark Nov-missing patients as censored.
6. Build primary/secondary/exploratory labels.
7. Build sparse CSR X from Oct observation histories only.
8. Save dataset artifacts:
   - `X_csr.npz`
   - `y.npy`
   - `patient_ids.npy`
   - `metadata.json`
   - `label_audit.json`

No feature column may use November data.

---

## 7. Evaluation Design

### What Can Be Evaluated Now

For the original Oct -> Nov pair alone, the immediate evaluation is feasibility and internal validation.

Allowed:

- label positive rate
- censoring rate
- clean_onset/escalation composition
- stratified internal CV within the eligible cohort
- calibration and top-K review within the same single-period task
- temporal holdout evaluation after building the new Nov -> Dec pair from 2024-12 Raw

Not allowed:

- claim external temporal generalization
- claim production readiness
- claim December holdout success before the Nov -> Dec dataset and report are generated

### Required Metrics

Class imbalance is expected to be stronger than same-window prevalence. Metrics must include PR-oriented and operational measures.

| metric | required | reason |
|---|---|---|
| ROC-AUC | yes | ranking quality baseline |
| PR-AUC | yes | imbalanced outcome robustness |
| best F1 | yes | operating threshold search |
| precision@top1% | yes | intervention workload estimate |
| precision@top5% | yes | broader outreach estimate |
| recall@top1% | yes | tight outreach missed-positive estimate |
| recall@top5% | yes | missed positive estimate |
| positive_rate_pct | yes | label feasibility |
| censoring_rate_pct | yes | cohort bias audit |

Success criteria for proceeding to full model comparison:

- primary label positive rate is at least 2%
- censored patients do not dominate the feature cohort
- PR-AUC exceeds prevalence by a meaningful margin
- precision@top5% is materially above base positive rate

These are feasibility criteria, not clinical deployment criteria.

### 2026-05-26 Temporal Holdout Result

After 2024-12 Raw arrived, the planned `Nov -> Dec` validation pair was built and evaluated with the existing `Oct -> Nov` augmented dataset as train.

| item | value |
|---|---:|
| train pair | Oct features -> Nov outcome |
| validation pair | Nov features -> Dec outcome |
| validation n | 25,749 |
| validation positive rate | 13.2627% |
| validation censoring rate | 13.2556% |
| input_dim | 14,706 |
| val_auc | 0.647111 |
| val_pr_auc | 0.211640 |
| precision@top1% | 0.325581 |
| precision@top5% | 0.287267 |
| recall@top5% | 0.108346 |
| patient overlap, validation rate | 14.7889% |

Patient-disjoint rerun:

| item | value |
|---|---:|
| validation n after overlap removal | 21,941 |
| removed overlap patients | 3,808 |
| validation positive rate after removal | 14.0787% |
| overlap patient positive rate | 8.5609% |
| val_auc | 0.630412 |
| val_pr_auc | 0.215572 |
| precision@top1% | 0.304545 |
| precision@top5% | 0.297814 |
| recall@top5% | 0.105860 |
| patient overlap, validation rate | 0.0% |

Decision: `WEAK_FEASIBLE_RESEARCH_TRACK` remains the correct status. The holdout AUC is below the 0.70 continuation threshold, so this sparse drug-code + prior institution-count model is not confirmed for temporal generalization. The PR-AUC and top-K precision exceed prevalence, so a signal exists, but it is not strong enough for baseline replacement or clinical claims.

Interpretation caveats:

- Patient-disjoint validation lowers AUC from 0.647111 to 0.630412, strengthening the research-only conclusion.
- Same-window multi-institution prediction remained strong on Dec (`AUC=0.843441`), while future-onset prediction remained weak (`AUC=0.647111`), suggesting the main limitation is future outcome signal rather than 2024-12 schema/vocab integrity.
- Future outcome overlap patients had lower new-onset rate (8.5609% vs 14.0787%), consistent with prior multi-institution history suppressing marginal new onset. This is an onset-vs-persistence asymmetry; same-window overlap patients had higher positive rate.
- Demographics v1 (`age_years_div_100`, `sex_type_1_flag`) on the patient-disjoint Nov->Dec holdout reached `AUC=0.633293`, `PR-AUC=0.216363`, and `precision@top5%=0.292350`. This is only a tiny AUC gain over drug+prior-institution (`0.630412`) and below the `0.64` marginal-signal threshold, so demographics alone do not change the research-only status.
- Medication-class v1 (`efmdc_clsf_no` multi-hot with `__NULL_EFMDC__` and `__UNK_EFMDC__`) on the same patient-disjoint holdout reached `AUC=0.642860`, `PR-AUC=0.226443`, and `precision@top5%=0.303279`. This is the first richer feature with material gain over drug+prior-institution (`AUC +0.012448`), but it remains below the `0.70` continuation threshold.
- Medication-class + demographics reached `AUC=0.645007`, `PR-AUC=0.227231`, and `precision@top5%=0.310565`, making it the current best sparse-linear future-onset research bundle. It still remains below the `0.70` continuation threshold.
- XGBoost on the class+demographics bundle reached `AUC=0.680783`, `PR-AUC=0.253017`, and `precision@top5%=0.333333` with zero patient overlap. This breaks the sparse-linear ceiling but remains below the `0.70` milestone, so it is a research-only model comparison rather than a baseline or clinical claim.
- XGBoost quick seed sensitivity with `xgb50_depth4` was stable across seeds 7, 42, and 99 (`AUC mean=0.679942`, `std=0.001438`, `min=0.678496`). This confirms the nonlinear gain is not a single-seed artifact, while still remaining below the `0.70` milestone.
- The future-onset research track is frozen on 2026-05-26 for the current Dec holdout. Frozen artifacts are `data/reports/future_onset_research_freeze_ledger.json`, `data/reports/future_onset_research_freeze_datasplit_manifest.csv`, `data/reports/future_onset_research_freeze_handoff.md`, and `data/models/future_onset_xgb_efmdc_demo_frozen_20260526.ubj`.
- Dataset scope (2026-06-02): the dataset is finalized at 6 months (2024-07..12, 500k sample). Jan 2025 / Gate 5A acquisition is cancelled and Gate 5B is retired; there is no remaining future-onset unlock trigger.
- The Nov→Dec future-onset holdout remains frozen (parked); do not perform model, feature, or hyperparameter experiments or related code changes against it.

---

## 8. Constraints and Risks

### 30-Day Feature Window Limitation

Current Raw starts at 2024-10-01, so Oct feature window cannot support 60-day or 90-day chronic medication history. This can under-represent maintenance therapies and long-cycle prescriptions.

### Date Boundary Handling

Current Raw uses date-level parquet partitions and normalized date columns. Window boundaries must be inclusive by date:

- observation: `2024-10-02 <= date <= 2024-10-31`
- outcome: `2024-11-01 <= date <= 2024-11-30`

No November row may enter feature construction. If future Raw includes timestamps, timestamps must be normalized to a single timezone before window assignment.

### Temporal Holdout Status

The original design had only one feature/outcome pair: Oct -> Nov. With 2024-12 Raw now available, the next required validation pair is Nov -> Dec. Until that holdout is built and reviewed, existing Oct -> Nov random-split estimates remain research feasibility metrics.

### Informative Censoring

Nov-missing patients are excluded because absence of claims may mean many different states. Censoring rate must be reported, and a high censoring rate should block model interpretation.

### Persistence Leakage

Prevalence outcome can be dominated by patient utilization persistence. This is why `Nov>=T` is secondary-only and `Oct<T AND Nov>=T` is primary.

### Institution-Level Concentration

Because the outcome is based on institution count, large institutions or regional patterns may dominate. Per-institution performance variance is not part of first implementation but should be tracked before operational use.

---

## 9. Output Contract

The implementation should produce a label audit report with at least:

```json
{
  "label_source": "future_multi_institution_onset",
  "feature_window": {"start": "2024-10-02", "end": "2024-10-31"},
  "outcome_window": {"start": "2024-11-01", "end": "2024-11-30"},
  "threshold": 6,
  "n_feature_cohort": 0,
  "n_evaluable": 0,
  "n_censored": 0,
  "censoring_rate_pct": 0.0,
  "label_positive": 0,
  "label_positive_rate_pct": 0.0,
  "clean_onset_positive": 0,
  "escalation_positive": 0,
  "persistence_cohort_size": 0,
  "persistence_rate_pct": 0.0,
  "oct_history_zero_excluded": 0,
  "oct_institution_count_percentiles": {},
  "nov_institution_count_percentiles": {},
  "label_semantics": "positive when oct_institution_count < T and nov_institution_count >= T"
}
```

Dataset metadata must include:

- source Raw date range
- vocab path/hash
- feature window
- outcome window
- label definition
- censoring policy
- temporal holdout status

---

## 10. Implementation Readiness

This spec is implementable with current data because both required windows exist:

- Oct observation window: available
- Nov outcome window: available

With 2024-12 Raw now available, the next implementation plan should build:

1. future label audit
2. future label dataset builder
3. `Nov -> Dec` future outcome dataset
4. temporal holdout training smoke using `Oct -> Nov` as train and `Nov -> Dec` as validation
5. summary report that explicitly separates feasibility, holdout performance, censoring shift, and patient overlap

Original planning estimates for the Oct -> Nov pair:

| quantity | estimate |
|---|---:|
| Oct feature cohort | ~40,000 |
| Oct persistence cohort (`oct_count >= 6`) | ~9,000-10,000 |
| evaluable strict-onset cohort after censoring | ~21,000-24,000 |
| onset positives | ~1,260-2,880 |
| onset positive rate | ~6-12% |

These are planning estimates. The audit implementation must report measured values.

---

## 11. Agent Review Summary

Claude:

- Primary should be `Oct<T AND Nov>=T` onset, not simple `Nov>=T` prevalence.
- Prevalence can be secondary only because persistence can inflate performance.
- Nov-missing patients must be censored/excluded, not negative.
- Include temporal holdout caveat until the `Nov -> Dec` validation run is measured.
- `oct_institution_count >= 6` should be excluded from primary training and recorded as `persistence_cohort`.

agy:

- Onset/escalation primary is appropriate.
- Split primary positives into clean onset and escalation slices.
- Nov no-record patients must be treated as censored to avoid informative censoring bias.
- PR-AUC and precision@topK are required because positive rate may drop sharply.
- Additional month temporal validation must be a production gate.
- `oct_history_rows >= 1` should be required for strict baseline observability.
- Threshold `T` should be parameterized in implementation.

---

## 12. Self-Review

- No placeholder requirements remain.
- Primary, secondary, and exploratory outcomes are distinct.
- Censoring is explicitly defined and does not conflict with label definitions.
- The document does not claim temporal generalization without 2024-12 data.
- Implementation scope is bounded to label/dataset feasibility, not production deployment.
