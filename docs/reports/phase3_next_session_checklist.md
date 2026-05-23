# Phase 3 Next Session Checklist

Last updated: 2026-05-23

## Current State

- Baseline status: `BASELINE_LOCKED`
- Accepted label: `multi_institution_t6_exact30_patient_disjoint`
- Accepted model: `sparse_linear`
- Baseline reference result: patient-disjoint temporal AUC `0.849140`, PR-AUC `0.650496`
- Future outcome track: `WEAK_FEASIBLE_RESEARCH_TRACK`
- Raw coverage currently available: `2024-10-01..2024-11-30`
- Current blocker: no 2024-12 Raw month

Decision documents:

- `docs/reports/phase3_baseline_summary.md`
- `data/reports/phase3_baseline_summary.json`
- Obsidian: `mode_11_hana_2026-05-18.md`

## Gate 0: Confirm 2024-12 Raw Arrived

```bash
ls data/Raw/records_202412*.parquet | sed -n '1,5p'
ls data/Raw/records_202412*.parquet | wc -l
```

Pass criteria:

- Expected count: 31 daily parquet files for `2024-12-01..2024-12-31`
- Required file for reference-date workflows: `data/Raw/records_20241231.parquet`

If the files are absent, stop. Do not run more model experiments on the current two-month data.

## Gate 1: Schema And Integrity Check

Compare 2024-12 schema with known-good 2024-11 schema.

```bash
cmd.exe /c '.venv_hana\Scripts\python.exe -c "import pyarrow.parquet as pq, json; a=pq.ParquetFile(r\"data\Raw\records_20241130.parquet\").schema_arrow; b=pq.ParquetFile(r\"data\Raw\records_20241231.parquet\").schema_arrow; print(json.dumps({\"missing_in_dec\": sorted(set(a.names)-set(b.names)), \"new_in_dec\": sorted(set(b.names)-set(a.names))}, ensure_ascii=False, indent=2))"'
```

Required columns:

- `patient_id`
- `edi_code`
- `start_date`
- `institution_id`
- `efmdc_clsf_no`
- `wk_compn_cd`

Null-rate spot check:

```bash
cmd.exe /c '.venv_hana\Scripts\python.exe -c "import pandas as pd, json; cols=[\"patient_id\",\"edi_code\",\"start_date\",\"institution_id\"]; df=pd.read_parquet(r\"data\Raw\records_20241231.parquet\", columns=cols); print(json.dumps({c: round(float(df[c].isna().mean()*100),4) for c in cols}, indent=2))"'
```

Pass criteria:

- No missing required columns
- `patient_id`, `edi_code`, `start_date` null rate is effectively `0%`
- `institution_id` null rate does not materially exceed prior-month levels

Fallback:

- If required columns are missing, stop and fix ETL/export.
- If null rates spike, run schema/data-quality audit before any model evaluation.

## Gate 2: Build 2024-12 Same-Window Holdout

Use the locked vocabulary unless a deliberate vocab rebuild is being evaluated. The baseline summary currently expects `input_dim=14705`.

```bash
cmd.exe /c '.venv_hana\Scripts\python.exe scripts\ops\build_sparse_training_dataset.py --raw-dir data\Raw --vocab-path data\vocab\drug_vocab.json --output-dir data\datasets\multi_inst_t6_20241231_l29 --reference-date 20241231 --lookback-days 29 --label-source multi_institution --multi-institution-threshold 6'
```

Inspect metadata:

```bash
python3 -c 'import json; d=json.load(open("data/datasets/multi_inst_t6_20241231_l29/metadata.json",encoding="utf-8")); print(json.dumps({k:d[k] for k in ["n_patients","input_dim","label_positive_rate_pct","unknown_drug_rate_pct","zero_vector_rate_pct"]}, ensure_ascii=False, indent=2))'
```

Pass criteria:

- `input_dim` remains `14705`
- label positive rate is not wildly outside the Oct/Nov range (`~22-24%`)
- `zero_vector_rate_pct` remains low
- unknown drug rate is not a schema/export failure signal

## Gate 3: Same-Window 3-Month Temporal Holdout

Train on the October locked training dataset and validate on December.

```bash
cmd.exe /c '.venv_hana\Scripts\python.exe scripts\ops\sparse_training_smoke.py --model linear --train-dataset-dir data\datasets\multi_inst_t6_20241031_l29 --val-dataset-dir data\datasets\multi_inst_t6_20241231_l29 --output-dir data\datasets\multi_inst_t6_temporal_20241031_to_20241231 --epochs 20 --batch-size 2048 --seed 42 --device cpu'
```

Review overlap in the output report:

```bash
python3 -c 'import json; d=json.load(open("data/datasets/multi_inst_t6_temporal_20241031_to_20241231/sparse_training_smoke_report.json",encoding="utf-8")); print(json.dumps({k:d[k] for k in ["patient_overlap_count","patient_overlap_val_rate_pct","train_label_positive_rate_pct","val_label_positive_rate_pct"]}, ensure_ascii=False, indent=2)); print(json.dumps(d["train"], ensure_ascii=False, indent=2))'
```

Success criteria:

- Primary: temporal holdout AUC `>= 0.80`
- Strong pass: AUC drop from Nov patient-disjoint baseline (`0.849140`) is `< 0.05`
- PR-AUC remains materially above prevalence
- Patient overlap is reported. If overlap is high, create a patient-disjoint December validation variant before claiming pass.

Fallback:

- If AUC drops below `0.80`, inspect label prevalence shift and unknown drug rate first.
- If patient overlap is high, do not compare against the patient-disjoint baseline until overlap is removed.

## Gate 4: Future Outcome 3-Window Track

The current future outcome track is `WEAK_FEASIBLE` using internal random split only. With December Raw, build a true temporal pair:

- Train dataset: Oct features -> Nov label
- Validation dataset: Nov features -> Dec label

Build the Nov -> Dec dataset:

```bash
cmd.exe /c '.venv_hana\Scripts\python.exe scripts\ops\build_future_outcome_dataset.py --raw-dir data\Raw --vocab-path data\vocab\drug_vocab.json --output-dir data\datasets\future_multi_inst_onset_t6_20241130_to_20241231_with_oct_inst_count --feature-reference-date 20241130 --outcome-reference-date 20241231 --lookback-days 29 --threshold 6 --add-institution-count-feature'
```

Run temporal feasibility:

```bash
cmd.exe /c '.venv_hana\Scripts\python.exe scripts\ops\sparse_training_smoke.py --model linear --train-dataset-dir data\datasets\future_multi_inst_onset_t6_20241031_to_20241130_with_oct_inst_count --val-dataset-dir data\datasets\future_multi_inst_onset_t6_20241130_to_20241231_with_oct_inst_count --output-dir data\datasets\future_multi_inst_onset_t6_temporal_20241031_to_20241231 --epochs 20 --batch-size 2048 --seed 42 --device cpu'
```

Success criteria:

- This track becomes worth continued modeling if temporal AUC `>= 0.70`
- PR-AUC should exceed outcome prevalence by a meaningful margin
- precision@top5% should materially exceed base positive rate

Fallback:

- If temporal AUC remains `< 0.70`, keep future outcome as a research track and prioritize richer features.
- Candidate richer features: medication class features, demographics, trend features, prior institution count deltas.

## Gate 5: Update Decision Trail

After any December holdout run:

```bash
cmd.exe /c '.venv_hana\Scripts\python.exe scripts\ops\phase3_baseline_summary.py --output-dir data\reports'
cp data/reports/phase3_baseline_summary.md docs/reports/phase3_baseline_summary.md
```

Also update Obsidian:

- `C:\Users\ptg\OneDrive\문서\Obsidian Vault\mode_11_hana_2026-05-18.md`

Record:

- Raw coverage
- dataset paths
- same-window holdout result
- future outcome temporal result
- pass/fail decision
- whether baseline remains locked or changes

## Do Not Do

- Do not claim temporal generalization from the current two-month internal split.
- Do not replace the locked baseline unless December holdout passes the criteria above.
- Do not treat missing December records as negative labels in future outcome tasks.
- Do not rebuild vocabulary just because 2024-12 arrived. Reuse `data/vocab/drug_vocab.json` (`input_dim=14705`) so train/validation dimensions stay aligned; new drug codes should map to `_unk`.
- Do not rebuild vocabulary silently; if vocab changes, record `input_dim`, SHA256, and reason.
