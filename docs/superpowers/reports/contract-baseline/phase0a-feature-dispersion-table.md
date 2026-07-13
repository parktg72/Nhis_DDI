# Phase 0A Cross-Source Feature Dispersion Table

## Purpose and policy

This report records current dispersion among four source-level contracts. It does not treat difference as a defect, does not require zero diff, and does not propose flattening profile boundaries. The repository remains `RESEARCH_TRACK_FROZEN`; Nov→Dec holdout use is prohibited, while Gate 5A/5B and 2025-01 acquisition are retired (`AGENTS.md:5-16`, `AGENTS.md:30-39`).

## Sources

| Key | Symbol | Role and container semantics | Source |
|---|---|---|---|
| B | `_BUILDER_KNOWN_COLS` | Online builder capability baseline; unordered `frozenset`, 24 names. | `serving/predictor.py:43-59` |
| F | `FEATURE_COLS` | Page 3/UI training default; ordered list, 22 names. | `hana_app/core/ml_runner.py:50-73`, use at `hana_app/core/ml_runner.py:1755-1758` |
| E | `ETL_NUMERIC_COLS` | Feature-engineering numeric inventory; ordered list, 14 names. | `scripts/features/feature_engineer.py:34-47` |
| D | `ML_DATASET_REQUIRED_COLUMNS` | Patient-level dataset presence contract; ordered tuple, 9 columns including identifier and label. | `scripts/datasets/contracts.py:23-33`, validation at `scripts/datasets/contracts.py:92-97` |

## Method and source snapshot

Every name, count, matrix cell, and set difference in this report was extracted from source at commit `3d8d64e78601a3ff56dc38034a9da62853e6b656` (`3d8d64e`, *docs: record raw sex type reporting design*), with the `serving/`, `hana_app/`, and `scripts/` trees unmodified in the working copy at extraction time. The extraction used the Python 3.12 `ast` script in [Appendix A of `phase0a-profile-contract-map.md`](./phase0a-profile-contract-map.md#appendix-a--constant-extractor-and-its-output) — the single extractor shared by both Phase 0A Task 1 reports. It parses the four cited assignments and imports no repository module. Two of the four constants are annotated assignments (`_BUILDER_KNOWN_COLS`, `ML_DATASET_REQUIRED_COLUMNS`) and one wraps its literal in a `frozenset()` call, so the script resolves `ast.Assign` and `ast.AnnAssign` and unwraps `frozenset()/set()/list()/tuple()` calls before evaluation.

Reproduce with `python3.12 phase0a_extract.py .` from the repository root at that commit. The `== counts ==`, `== union ==`, and `== set differences ==` blocks of its output — reproduced verbatim in Appendix A.2 and excerpted below — are the direct evidence for the matrix and difference tables that follow.

```text
== counts ==
B _BUILDER_KNOWN_COLS: 24
F FEATURE_COLS: 22
E ETL_NUMERIC_COLS: 14
D ML_DATASET_REQUIRED_COLUMNS: 9

== set differences ==
B \ F (n=2): avg_drug_duration, long_term_drug_count
F \ B (n=0): none
B \ E (n=10): avg_drug_duration, cyp_high_risk_pairs, cyp_max_enzyme_risk, cyp_risk_score, dup_efmdc, has_hepatic_risk_drug, has_high_risk_drug, has_renal_risk_drug, long_term_drug_count, sex_m
E \ B (n=0): none
F \ E (n=8): cyp_high_risk_pairs, cyp_max_enzyme_risk, cyp_risk_score, dup_efmdc, has_hepatic_risk_drug, has_high_risk_drug, has_renal_risk_drug, sex_m
E \ F (n=0): none
D \ E (n=2): patient_id, risk_level
E \ D (n=7): age, dup_atc3, dup_atc4, dup_atc5, dup_same_ingredient, qt_risk_count, triple_whammy
```

## Cross-source presence matrix

Cells below are the `== union ==` block of the same run (`Y` per source in B, F, E, D order).

| Name | B | F | E | D |
|---|:---:|:---:|:---:|:---:|
| `age` | Y | Y | Y | – |
| `avg_drug_duration` | Y | – | – | – |
| `cyp_high_risk_pairs` | Y | Y | – | – |
| `cyp_max_enzyme_risk` | Y | Y | – | – |
| `cyp_risk_score` | Y | Y | – | – |
| `ddi_contraindicated` | Y | Y | Y | Y |
| `ddi_major` | Y | Y | Y | Y |
| `ddi_minor` | Y | Y | Y | Y |
| `ddi_moderate` | Y | Y | Y | Y |
| `drug_count` | Y | Y | Y | Y |
| `drug_count_7d` | Y | Y | Y | Y |
| `dup_atc3` | Y | Y | Y | – |
| `dup_atc4` | Y | Y | Y | – |
| `dup_atc5` | Y | Y | Y | – |
| `dup_efmdc` | Y | Y | – | – |
| `dup_same_ingredient` | Y | Y | Y | – |
| `has_hepatic_risk_drug` | Y | Y | – | – |
| `has_high_risk_drug` | Y | Y | – | – |
| `has_renal_risk_drug` | Y | Y | – | – |
| `institution_count` | Y | Y | Y | Y |
| `long_term_drug_count` | Y | – | – | – |
| `patient_id` | – | – | – | Y |
| `qt_risk_count` | Y | Y | Y | – |
| `risk_level` | – | – | – | Y |
| `sex_m` | Y | Y | – | – |
| `triple_whammy` | Y | Y | Y | – |

Every matrix cell is derived from the four assignments at `serving/predictor.py:45-53`, `hana_app/core/ml_runner.py:50-73`, `scripts/features/feature_engineer.py:37-43`, and `scripts/datasets/contracts.py:23-33`.

## Exact set differences

| Comparison | Left-only names | Interpretation | Source |
|---|---|---|---|
| B \ F | `avg_drug_duration`, `long_term_drug_count` | Online builder capability exceeds the UI default list by two duration features. | `serving/predictor.py:45-53`; `hana_app/core/ml_runner.py:50-73` |
| F \ B | none | `FEATURE_COLS` is a strict subset of B in current source. | Same assignments above. |
| B \ E | `avg_drug_duration`, `cyp_high_risk_pairs`, `cyp_max_enzyme_risk`, `cyp_risk_score`, `dup_efmdc`, `has_hepatic_risk_drug`, `has_high_risk_drug`, `has_renal_risk_drug`, `long_term_drug_count`, `sex_m` | B covers online/resource-derived and duration/sex features not enumerated in E. | `serving/predictor.py:45-53`; `scripts/features/feature_engineer.py:37-43` |
| E \ B | none | E is a subset of B in current source. | Same assignments above. |
| F \ E | `cyp_high_risk_pairs`, `cyp_max_enzyme_risk`, `cyp_risk_score`, `dup_efmdc`, `has_hepatic_risk_drug`, `has_high_risk_drug`, `has_renal_risk_drug`, `sex_m` | UI defaults include eight names outside E. | `hana_app/core/ml_runner.py:50-73`; `scripts/features/feature_engineer.py:37-43` |
| E \ F | none | E is a subset of F in current source. | Same assignments above. |
| D \ E | `patient_id`, `risk_level` | Dataset contract adds identifier and label columns, which are not numeric model features. | `scripts/datasets/contracts.py:23-33`; `scripts/features/feature_engineer.py:36-47` |
| E \ D | `age`, `dup_atc3`, `dup_atc4`, `dup_atc5`, `dup_same_ingredient`, `qt_risk_count`, `triple_whammy` | E contains seven numeric names outside the minimum dataset presence contract. | Same assignments above. |

## Semantics behind notable differences

- `dup_efmdc` is present in B/F and absent from E. Online serving documents it as the EDI→EFMDC bridge with a `0.0` degraded fallback, while UI rows expose the aggregated value (`serving/predictor.py:43-53`, `serving/predictor.py:1072-1074`; `hana_app/core/ml_runner.py:440-465`).
- `sex_m` is present in B/F and absent from E. Online serving defaults missing sex to `0.5`; UI rows also use `0.5` for an unknown value, while `FeatureEngineer.run()` separately creates a column named `sex_male`, not `sex_m` (`serving/predictor.py:1038-1039`; `hana_app/core/ml_runner.py:458-463`; `scripts/features/feature_engineer.py:124-127`).
- The `sex_m` **input domains also differ** between the two producers. The UI `ml_runner` row builder encodes from the raw HANA sex strings `"1"`/`"2"` and additionally emits a `sex_type` metadata column carrying that raw value (metadata, not a `FEATURE_COLS` model input), while the serving request domain is `M`/`F` enforced by Pydantic validation. Same output codes, different input vocabularies — a contract distinction, not a defect claim:

  ```python
  # hana_app/core/ml_runner.py:459-460 — raw HANA "1"/"2" + sex_type metadata
  "sex_type": f.sex,
  "sex_m": 1.0 if f.sex == "1" else (0.0 if f.sex == "2" else 0.5),
  # serving/predictor.py:1039 — request M/F domain
  feat["sex_m"] = float(req.patient_sex == "M") if req.patient_sex else 0.5
  # serving/schemas.py:76 — request validation
  patient_sex: Optional[str] = Field(None, pattern="^[MF]$", description="성별 (M/F)")
  ```
- `avg_drug_duration` and `long_term_drug_count` are B-only among these four sources and are computed online from request duration values (`serving/predictor.py:45-53`, `serving/predictor.py:1050-1053`).
- `patient_id` and `risk_level` are D-only because D is a minimum dataset contract that includes metadata and a label; its validator checks presence rather than feature eligibility or physical order (`scripts/datasets/contracts.py:23-33`, `scripts/datasets/contracts.py:92-97`).
- E is not the complete physical Parquet schema: `FeatureEngineer.run()` starts from an ETL DataFrame, merges CYP/temporal frames, creates label/sex columns, normalizes/selects, and writes the resulting DataFrame (`scripts/features/feature_engineer.py:80-144`).

## Historical context: commit `d201743`

`F \ B` being empty today is partly the result of one earlier alignment commit, recorded here as background rather than as evidence about current behavior. Commit `d201743` (2026-04-29, `fix(serving): RequestFeatureBuilder 컬럼명 학습 파이프라인과 정렬`) renamed the serving builder's `sex_male` to `sex_m` against `ml_runner`'s `FEATURE_COLS`, added `dup_atc3` and the three `has_*_risk_drug` flags, added a `0.0` fallback branch for `cyp_max_enzyme_risk`, and updated `_BUILDER_KNOWN_COLS` accordingly (commit message of `d201743`).

Three limits on what that commit supports:

- It aligned **names** between the serving builder and `FEATURE_COLS`. It is not evidence that the two producers compute equal **values**, and it says nothing about physical order — see the order section below and the `sex_m` input-domain difference above.
- Its scope was `serving/predictor.py` against `hana_app/core/ml_runner.py`. It did not touch `scripts/features/feature_engineer.py`, which still creates a column named `sex_male` (`scripts/features/feature_engineer.py:124-127`); the E-side naming divergence therefore survives it.
- Its `dup_efmdc = 0.0` fixed-value constraint has since been superseded: current source documents `dup_efmdc` as an EDI→EFMDC bridge output with `0.0` only as a degraded fallback (`serving/predictor.py:43-53`, `serving/predictor.py:1072-1074`). Where this report and that commit message disagree, the source at `3d8d64e` governs.

The commit is also cited in current source as the precedent motivating the DDI semantic-version reload guard (`scripts/etl/prescription_aggregator.py:204-209`).

## Order implications

B is unordered, whereas F, E, and D have declared sequence syntax. Even for the latter three, the sequences serve different purposes: UI selection order, a numeric inventory, and minimum required-column presence respectively (`serving/predictor.py:45-59`; `hana_app/core/ml_runner.py:50-73`; `scripts/features/feature_engineer.py:36-47`; `scripts/datasets/contracts.py:23-33`, `scripts/datasets/contracts.py:92-97`). Serving vector order comes from bundle metadata and name alignment, while physical training Parquet order is affected by DataFrame merges and transforms (`serving/predictor.py:371-418`, `serving/predictor.py:1112-1142`; `scripts/features/feature_engineer.py:80-144`). Therefore set equality, where it exists, is not evidence of positional equality.

## Baseline conclusion

The four lists are dispersed by role and container semantics. This baseline neither merges them nor marks their differences as failures. Any later contract work should preserve profile identity and test the explicitly chosen ordered boundary rather than impose zero diff (`AGENTS.md:10-16`; current boundaries cited throughout this report).
