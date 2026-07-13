# Phase 0A Profile Contract Map

## Scope and method

This is a source-level snapshot of the four current profiles. It was produced with Python 3.12 `ast` parsing and text inspection only; no `serving`, `hana_app`, `scripts`, or `rules` module was imported, and no model, Parquet, generated output, or protected artifact was opened. The policy baseline remains `RESEARCH_TRACK_FROZEN`: the Nov→Dec holdout is excluded from model, feature, ablation, and hyperparameter work; Gate 5A/5B and 2025-01 acquisition are retired rather than active unlocks (`AGENTS.md:5-16`, `AGENTS.md:30-39`).

**Source snapshot:** every constant, count, and line citation below was read from commit `3d8d64e78601a3ff56dc38034a9da62853e6b656` (`3d8d64e`, *docs: record raw sex type reporting design*), with the `serving/`, `hana_app/`, and `scripts/` trees unmodified in the working copy at extraction time. Constant values were produced by the extractor in [Appendix A](#appendix-a--constant-extractor-and-its-output), whose full deterministic output is reproduced there; behavioral claims (defaults, thresholds, guards, reload paths) come from reading the cited lines. Re-running the appendix script against the same commit reproduces every constant, count, and set difference used by this report and by [`phase0a-feature-dispersion-table.md`](./phase0a-feature-dispersion-table.md).

The extractor must resolve both `ast.Assign` and `ast.AnnAssign`, because two of the four feature contracts are annotated assignments (`_BUILDER_KNOWN_COLS: frozenset[str] = frozenset({...})`, `ML_DATASET_REQUIRED_COLUMNS: tuple[str, ...] = (...)`), and it must unwrap `frozenset()/set()/list()/tuple()` calls, because `ast.literal_eval` alone rejects a call node.

The profile names below describe distinct current contracts. Their differences are recorded, not normalized.

## 1. `tabular_binary`

### 1.1 Feature source and order

The serving capability set is `_BUILDER_KNOWN_COLS`, an unordered `frozenset` of 24 names. `_INTENTIONAL_FEATURE_ALLOWLIST` is empty, so `_FEATURE_ALLOWED` currently equals that set (`serving/predictor.py:_BUILDER_KNOWN_COLS`, `serving/predictor.py:43-59`). Sorted for inventory only:

```text
age, avg_drug_duration, cyp_high_risk_pairs, cyp_max_enzyme_risk,
cyp_risk_score, ddi_contraindicated, ddi_major, ddi_minor, ddi_moderate,
drug_count, drug_count_7d, dup_atc3, dup_atc4, dup_atc5, dup_efmdc,
dup_same_ingredient, has_hepatic_risk_drug, has_high_risk_drug,
has_renal_risk_drug, institution_count, long_term_drug_count,
qt_risk_count, sex_m, triple_whammy
```

The set does **not** define vector order. `MLModel.load()` obtains the ordered `feature_names` list from bundle state, and `RequestFeatureBuilder.build()` constructs an insertion-ordered dictionary in that sequence before converting the one-row DataFrame to a float vector (`serving/predictor.py:371-418`, `serving/predictor.py:1112-1142`).

### 1.2 Dtype and default semantics

The online builder declares `feat` as `dict[str, float]`; counts and boolean-like flags are converted to floats, and the final vector is `astype(float)` (`serving/predictor.py:1029-1048`, `serving/predictor.py:1085-1110`, `serving/predictor.py:1141-1142`). Current fallback semantics are:

| Feature group | Current value/default behavior | Source |
|---|---|---|
| `drug_count`, `drug_count_7d`, duplicate counts | Use the EDI→WK/DrugMaster parity path when available; otherwise use EDI/ATC-derived counts, with `dup_efmdc=0.0` and `drug_count_7d=drug_count`. | `serving/predictor.py:_count_dup_features`, `serving/predictor.py:904-953`, `serving/predictor.py:1029-1074`, `serving/predictor.py:1107-1110` |
| `institution_count` | Distinct non-empty institution IDs, as float. | `serving/predictor.py:1034-1037` |
| `age` | `float(patient_age or 0)`. | `serving/predictor.py:1038-1038` |
| `sex_m` | `1.0` for `M`, `0.0` for a supplied non-`M` value, and `0.5` when absent. | `serving/predictor.py:1039-1039`; request validation limits supplied values to `M/F` at `serving/schemas.py:71-77` |
| `ddi_*` | Float severity counts; all four are zero when the standardizer, DDI matrix, or DrugMaster is unavailable, or fewer than two mapped records exist. | `serving/predictor.py:_count_ddi`, `serving/predictor.py:869-885`, `serving/predictor.py:1041-1048` |
| Duration features | Mean duration or `0.0` for no drugs; count of durations ≥30 days. | `serving/predictor.py:1050-1053` |
| CYP features | Extracted when a CYP extractor and ATC codes exist; otherwise all three are `0.0`. | `serving/predictor.py:1076-1083` |
| Risk flags and `triple_whammy` | Component/DrugMaster path only when rule semantics are active; otherwise name/ATC fallback, each cast to float. | `serving/predictor.py:1085-1105` |
| Unknown bundle feature name | Name alignment uses `feat.get(name, 0.0)`; schema validation normally prevents this, while active lenient mode records degraded loading. | `serving/predictor.py:93-139`, `serving/predictor.py:1112-1116` |

### 1.3 Labels, threshold, and intervention

The common serving label space is `Red`, `Yellow`, `Green`, `Normal`, ordered 3→0 (`serving/schemas.py:RiskLevel`, `serving/schemas.py:22-37`). `MLModel._threshold` starts at `0.5` and loads `best_threshold` with the same fallback; `classify()` emits Red at `t`, Yellow at `0.6t`, Green at `0.3t`, otherwise Normal (`serving/predictor.py:305-320`, `serving/predictor.py:371-386`, `serving/predictor.py:611-619`). Final response intervention is selected from the final hybrid level: Red `즉각 개입`, Yellow `복약 상담`, Green/Normal `관여 안 함` (`serving/schemas.py:INTERVENTION_MAP`, `serving/schemas.py:283-290`; `serving/predictor.py:1501-1517`).

### 1.4 Semantic and schema guards

`DDI_FEATURE_SEMANTICS_VERSION` is `ddi.v2`, meaning WK→DrugMaster→DB-code severity over overlapping pairs (`scripts/etl/prescription_aggregator.py:DDI_FEATURE_SEMANTICS_VERSION`, `scripts/etl/prescription_aggregator.py:204-209`). If any loaded feature begins with `ddi_`, missing or mismatched bundle metadata rejects the model (`serving/predictor.py:389-406`).

Strict schema validation rejects any bundle feature outside `_FEATURE_ALLOWED`. The temporary lenient path accepts `FEATURE_SCHEMA_LENIENT` values `1`, `true`, or `yes` only before the sunset and records the missing columns for degraded `0.0` fallback (`serving/predictor.py:_validate_feature_schema`, `serving/predictor.py:93-139`).

### 1.5 Reload and rollback

`reload_model()` loads into a new object and swaps it under `_ml_lock` only when `load()` returns true. A false return leaves the previous model referenced, providing rollback-by-retention (`serving/predictor.py:1185-1204`, `serving/predictor.py:1286-1294`). The production inference path snapshots the current model under the lock, builds in bundle feature order, calls `predict_proba()`, then `classify()` (`serving/predictor.py:1405-1416`).

## 2. `hierarchical`

### 2.1 Feature source, dtype, and order

The ordered feature contract comes from `stage_meta.json["feature_cols"]`, written from the training function's `feature_cols` argument and read before model loading (`hana_app/core/hierarchical_runner.py:463-485`, `hana_app/core/hierarchical_runner.py:620-640`; `serving/predictor.py:652-669`). Empty feature lists or names outside the serving allowed set reject loading; reload repeats both checks (`serving/predictor.py:701-717`, `serving/predictor.py:1296-1315`). The serving builder supplies a float NumPy vector in the exact metadata order, while `predict_risk()` explicitly requires columns to match training order (`serving/predictor.py:1112-1142`; `hana_app/core/hierarchical_runner.py:744-766`). Defaults therefore inherit the online builder semantics in section 1.2.

### 2.2 Labels, actions, and interventions

The ordered Yellow subtype tuple is:

```text
Y_TRIPLE, Y_DOUBLE, Y_DDI_MAJOR, Y_DDI_MOD, Y_DUP, Y_FRAG
```

`STAGE2_LABELS` appends `No_Alert`, yielding seven ordered Stage-2 outputs (`hana_app/core/hierarchical_runner.py:YELLOW_SUBTYPE_LABELS`, `hana_app/core/hierarchical_runner.py:28-37`). Bundle `stage2_labels` must exactly equal this order, and the loaded encoder/classes are checked again after deserialization (`serving/predictor.py:670-685`, `serving/predictor.py:736-757`). Serving maps `Red` to Red, any `Y_*` to Yellow, and `No_Alert` to Normal (`serving/predictor.py:_stage2_label_to_risk`, `serving/predictor.py:1346-1358`).

| Output | Action | Source |
|---|---|---|
| Red | `즉각 개입` | `hana_app/core/hierarchical_runner.py:RED_ACTION`, `hana_app/core/hierarchical_runner.py:685-688` |
| `Y_DDI_MAJOR` | `약사 전화` | `hana_app/core/hierarchical_runner.py:ACTION_BY_LABEL`, `hana_app/core/hierarchical_runner.py:690-702` |
| `Y_TRIPLE` | `문자 안내` | `hana_app/core/hierarchical_runner.py:ACTION_BY_LABEL`, `hana_app/core/hierarchical_runner.py:690-702` |
| `Y_DOUBLE`, `Y_DDI_MOD`, `Y_DUP`, `Y_FRAG` | `모니터링` | `hana_app/core/hierarchical_runner.py:ACTION_BY_LABEL`, `hana_app/core/hierarchical_runner.py:690-702` |
| `No_Alert` | `관여 안 함` | `hana_app/core/hierarchical_runner.py:ACTION_BY_LABEL`, `hana_app/core/hierarchical_runner.py:690-702` |

The response's general intervention still comes from the four-level `INTERVENTION_MAP`; subtype `action` is an additional field (`serving/schemas.py:283-290`, `serving/predictor.py:1501-1517`).

### 2.3 Thresholds and backstops

`tau_red` and `tau_review` are stored in `stage_meta.json`. `p_red >= tau_red` returns Red without Stage 2; lower probabilities receive a Stage-2 label, with `red_suspect=true` in the review band `tau_review <= p_red < tau_red` (`hana_app/core/hierarchical_runner.py:_dispatch_result`, `hana_app/core/hierarchical_runner.py:705-741`; `hana_app/core/hierarchical_runner.py:793-804`). If Stage 1 degrades for insufficient class support, its fixed thresholds are `tau_red=1.0`, `tau_review=0.5` and its probability is always zero (`hana_app/core/hierarchical_runner.py:497-517`).

Two model-independent floors can only escalate: contraindication trigger `RED_CONTRAINDICATED` can force Red, and the subtype floor can guarantee `Y_DDI_MAJOR` or `Y_TRIPLE` when its rule conditions are met (`serving/predictor.py:1438-1476`; builder behavior at `serving/predictor.py:982-1007`).

### 2.4 Semantic versions and serving dependency

The hierarchical loader hard-rejects missing/mismatched `ddi_feature_semantics_version` against `ddi.v2` (`serving/predictor.py:686-700`). `FEATURE_SEMANTICS_VERSION` is `rulefeat.v1` (`scripts/etl/prescription_aggregator.py:211-216`), but current serving code does **not** reject a mismatch: it compares bundle metadata at prediction time and activates the component-based rule-feature path only on equality (`serving/predictor.py:780-782`, `serving/predictor.py:1385-1396`). This implementation is narrower than the nearby source comment claiming a reload guard, so the runtime gate is the authoritative observed behavior (`scripts/etl/prescription_aggregator.py:211-216`; `serving/predictor.py:686-717`).

Serving has a direct runtime dependency on `hana_app.core.hierarchical_runner`: the loader imports `STAGE2_LABELS`, `predict_risk_single()` imports `predict_risk`, and the subtype floor imports `ACTION_BY_LABEL` (`serving/predictor.py:674-678`, `serving/predictor.py:784-798`, `serving/predictor.py:1469-1475`). The production chain is `RequestFeatureBuilder.build()` → `HierarchicalPredictor.predict_risk_single()` → `hana_app.core.hierarchical_runner.predict_risk()` (`serving/predictor.py:1385-1404`).

### 2.5 Reload and rollback

`reload_hierarchical()` builds and validates a new predictor, then swaps under `_hier_lock` only after load, non-empty feature, and schema checks pass. False/validation failures retain the previous object (`serving/predictor.py:1296-1315`). Startup prefers a valid hierarchical directory and otherwise falls back to the single model path (`serving/predictor.py:1232-1259`).

## 3. `ui_experimental`

### 3.1 Feature source, order, and dtype/defaults

`FEATURE_COLS` is an ordered 22-name list used as the training default and as the Page 3 multiselect order (`hana_app/core/ml_runner.py:FEATURE_COLS`, `hana_app/core/ml_runner.py:50-73`, `hana_app/core/ml_runner.py:1676-1758`; `hana_app/pages/3_🤖_모델_학습.py:1114-1140`):

```text
drug_count, drug_count_7d, institution_count, ddi_contraindicated, ddi_major,
ddi_moderate, ddi_minor, triple_whammy, qt_risk_count, dup_same_ingredient,
dup_atc5, dup_atc4, dup_atc3, dup_efmdc, has_high_risk_drug,
has_renal_risk_drug, has_hepatic_risk_drug, cyp_risk_score,
cyp_max_enzyme_risk, cyp_high_risk_pairs, age, sex_m
```

The UI row builder preserves native numeric feature values, casts boolean flags to integers, defaults missing age to `-1`, and encodes sex as `1.0`, `0.0`, or `0.5` (`hana_app/core/ml_runner.py:_patient_features_to_row`, `hana_app/core/ml_runner.py:440-465`). This differs from online age's `0` default and is a contract difference, not a claim of defect (`serving/predictor.py:1038-1039`).

The `sex_m` **input domains differ** even though both paths emit the same `1.0/0.0/0.5` codes. The UI row builder maps the raw HANA sex strings `"1"`/`"2"` and additionally emits a `sex_type` metadata column carrying the raw value, while the serving request accepts only `M`/`F` after Pydantic validation:

```python
# hana_app/core/ml_runner.py:459-460 — raw HANA string domain + sex_type metadata
"sex_type": f.sex,
"sex_m": 1.0 if f.sex == "1" else (0.0 if f.sex == "2" else 0.5),
```

```python
# serving/predictor.py:1039 — request M/F domain
feat["sex_m"] = float(req.patient_sex == "M") if req.patient_sex else 0.5
# serving/schemas.py:76 — request validation
patient_sex: Optional[str] = Field(None, pattern="^[MF]$", description="성별 (M/F)")
```

`sex_type` is metadata, not a `FEATURE_COLS` model input; any code equating the two domains without mapping would misencode, so the distinction is part of the contract inventory.

### 3.2 Labels and thresholds

`RISK_LABEL_MAP` is `Red:3`, `Yellow:2`, `Green:1`, `Normal:0`. The UI row's `risk_binary` is 1 for both Red and Yellow, unlike `scripts/features/feature_engineer.py`'s `is_high_risk`, which is 1 only for Red (`hana_app/core/ml_runner.py:RISK_LABEL_MAP`, `hana_app/core/ml_runner.py:75-80`, `hana_app/core/ml_runner.py:461-463`; `scripts/features/feature_engineer.py:45-47`, `scripts/features/feature_engineer.py:120-123`). UI training supports `risk_binary` or four-class `risk_label`; optional binary threshold optimization is off by default and uses `0.5` unless explicitly run (`hana_app/core/ml_runner.py:1676-1707`, `hana_app/core/ml_runner.py:2084-2096`, `hana_app/core/ml_runner.py:_optimize_threshold`, `hana_app/core/ml_runner.py:2417-2438`).

### 3.3 Training path and separation

The UI path builds patient features through `aggregate_patient_features`, can stratify data, performs train/test stratification and cross-validation, and returns metrics; it is distinct from the operational serving bundle load path (`hana_app/core/ml_runner.py:823-882`, `hana_app/core/ml_runner.py:1676-1717`, `hana_app/core/ml_runner.py:1863-1963`; operational selection at `serving/predictor.py:1232-1259`). Page 3 offers `FEATURE_COLS` as selectable inputs and invokes `train_model` or `train_hierarchical` in its own UI flow (`hana_app/pages/3_🤖_모델_학습.py:23-27`, `hana_app/pages/3_🤖_모델_학습.py:1114-1140`, `hana_app/pages/3_🤖_모델_학습.py:1884-1910`). It has no serving intervention mapping or hot-reload/rollback contract of its own; those behaviors live in the operational predictor and response schema (`serving/predictor.py:1180-1328`, `serving/predictor.py:1501-1517`; `serving/schemas.py:283-290`).

Memory protection is active in both feature construction and training via `MemoryGuard`; it escalates at 80/90/95 percent and raises `MemoryLimitExceeded` at the hard stop (`hana_app/core/ml_runner.py:823-858`, `hana_app/core/ml_runner.py:1729-1758`; `hana_app/core/memory_guard.py:68-99`, `hana_app/core/memory_guard.py:131-195`). `page_guards.py` defines HANA validation helpers, but Page 3 does not import them in the current source; its live/local mode handling is inline, so `page_guards.py` must not be represented as an active Page 3 dependency (`hana_app/core/page_guards.py:9-52`; `hana_app/pages/3_🤖_모델_학습.py:18-49`).

### 3.4 Dispersion and safe misclassification payload

`FEATURE_COLS` is a strict subset of `_BUILDER_KNOWN_COLS`; only `avg_drug_duration` and `long_term_drug_count` are builder-only (`serving/predictor.py:45-53`; `hana_app/core/ml_runner.py:50-73`). `_SAFE_MISCLS_FEATURES` contains 20 of the 22 UI features, excluding `age` and `sex_m`; present non-null values are JSON-normalized for case reporting (`hana_app/core/ml_runner.py:_SAFE_MISCLS_FEATURES`, `hana_app/core/ml_runner.py:91-121`). These differences are inventory findings and are not a zero-diff target.

## 4. `dl_history`

### 4.1 Dataset and bundle contract

The ordered required event columns are `patient_id`, `drug_code`, `prescription_date`. Required bundle files are `model.pt`, `model_config.json`, `drug_vocab.json`, `edge_index.pt`, `feature_normalizer.pkl`, and `schema_version.json`; the manifest is `MANIFEST.json` and its hash algorithm is `sha256` (`scripts/datasets/contracts.py:DL_DATASET_REQUIRED_COLUMNS`, `scripts/datasets/contracts.py:35-55`). Manifest validation checks the track, run/schema IDs, hash algorithm, required files, and hashes before loading (`scripts/datasets/contracts.py:136-176`, `scripts/datasets/contracts.py:201-258`).

### 4.2 Dtype, defaults, lookback, architecture, and output

Lookback defaults to 365 days and is constrained to 7..1825; artifact and runtime values must be equal or `LookbackMismatchError` is raised (`scripts/datasets/contracts.py:53-59`, `scripts/datasets/contracts.py:100-125`; `serving/dl_predictor.py:93-101`). Encoding supports only `multi_hot`: the input is a float list initialized to `0.0`, observed drug positions become `1.0`, and OOV drugs use `_unk` when available or are dropped with a warning for legacy vocabularies (`serving/dl_predictor.py:29-32`, `serving/dl_predictor.py:239-259`, `serving/dl_predictor.py:262-317`). `gat` and `gcn` are the graph architectures that receive `edge_index`; other configured architectures use the single-tensor call (`serving/dl_predictor.py:29-32`, `serving/dl_predictor.py:342-350`). Output labels come from `model_config.json`; inference uses softmax/sigmoid probabilities and selects the maximum-probability label, with no profile decision threshold or action/intervention mapping in this auxiliary result (`serving/dl_predictor.py:123-179`, `serving/dl_predictor.py:239-259`, `serving/dl_predictor.py:352-360`).

### 4.3 Production role and reload behavior

When a validated DL bundle and history provider are both present, serving fetches the patient's bounded history and calls `DLModel.predict()`; failures are captured in `dl_error` (`serving/predictor.py:1418-1436`). The provider boundary validates the three required columns, and the concrete extractor adapter maps the confirmed configured EDI column into `drug_code` (`serving/hana_history.py:51-66`, `serving/hana_history.py:112-186`). The DL result is auxiliary: final `risk_level` is computed from rule/tabular or hierarchical paths before `dl_prediction` is attached to the response (`serving/predictor.py:1375-1417`, `serving/predictor.py:1451-1517`).

`reload_dl()` validates/loads a new object before swapping it under `_dl_lock`; current `DLModel.load()` returns true after validation or raises before state update (`serving/predictor.py:1317-1328`; `serving/dl_predictor.py:93-121`). Unlike the other reload methods, `reload_dl()` neither branches on the boolean return nor returns false, so its rollback contract depends on `DLModel.load()` continuing to raise on all failures; a future false-without-exception load would be swapped and reported successful (`serving/predictor.py:1286-1328`).

## 5. Semantic versions and `FEATURE_SCHEMA_LENIENT`

| Item | Current state | Enforcement | Source |
|---|---|---|---|
| `DDI_FEATURE_SEMANTICS_VERSION` | `ddi.v2` | Hard load guard for tabular models using `ddi_*` and for hierarchical bundles. | `scripts/etl/prescription_aggregator.py:204-209`; `serving/predictor.py:389-406`, `serving/predictor.py:686-700` |
| `FEATURE_SEMANTICS_VERSION` | `rulefeat.v1` | Hierarchical runtime feature-path gate; no current hard load rejection. | `scripts/etl/prescription_aggregator.py:211-216`; `serving/predictor.py:780-782`, `serving/predictor.py:1385-1396` |
| Code sunset default | `2026-08-01` | Lenient is allowed only while `today < sunset`. | `serving/predictor.py:_FEATURE_SCHEMA_LENIENT_SUNSET_DEFAULT`, `serving/predictor.py:62-90` |
| Accepted enable values | `1`, `true`, `yes` (case-insensitive after stripping) | Still subject to sunset. | `serving/predictor.py:_validate_feature_schema`, `serving/predictor.py:109-139` |
| Sunset override | `FEATURE_SCHEMA_LENIENT_SUNSET_DATE=YYYY-MM-DD` | Invalid date fails closed to strict mode. | `serving/predictor.py:_is_feature_schema_lenient_allowed`, `serving/predictor.py:68-90` |

Source-only environment inspection on 2026-07-12 observed both `FEATURE_SCHEMA_LENIENT` and `FEATURE_SCHEMA_LENIENT_SUNSET_DATE` unset. Under the code default, that date is before the 2026-08-01 sunset, but lenient remains inactive because the enable variable is unset; the interpretation follows `serving/predictor.py:68-90` and `serving/predictor.py:109-139`. This is the implementer's process environment, not evidence of the Windows production environment.

```console
$ python3 -c "import os; print(repr(os.environ.get('FEATURE_SCHEMA_LENIENT')), \
  repr(os.environ.get('FEATURE_SCHEMA_LENIENT_SUNSET_DATE')))"
None None   # 2026-07-12, implementer WSL environment — not Windows production
```

```python
# serving/predictor.py:65 — code-default sunset
_FEATURE_SCHEMA_LENIENT_SUNSET_DEFAULT: date = date(2026, 8, 1)
# serving/predictor.py:114-117 — accepted enable values, gated by sunset
lenient_env = os.environ.get("FEATURE_SCHEMA_LENIENT", "").strip().lower() in (
    "1", "true", "yes",
)
lenient_active = lenient_env and _is_feature_schema_lenient_allowed()
```

Operational risk: once `today >= sunset`, an unknown-feature bundle is strictly rejected even if the enable variable is set; before sunset, lenient loading silently supplies `0.0` for unknown features while recording schema drift (`serving/predictor.py:93-139`, `serving/predictor.py:408-418`).

## 6. Required resource fallbacks

| Resource | Current fallback/impact | Source |
|---|---|---|
| DDI matrix | Optional load. If absent, online `ddi_*` counts are all zero and DDI alert enrichment returns no alerts. | `serving/predictor.py:1185-1209`, `serving/predictor.py:869-901`, `serving/predictor.py:1520-1529` |
| CYP extractor | Optional load. If missing or no ATC codes exist, all three CYP features are `0.0`. | `serving/predictor.py:1220-1230`, `serving/predictor.py:1076-1083` |
| CodeStandardizer / DrugMaster | Optional load. Missing standardizer skips EDI→ATC enrichment; parity DDI/duplicate/rule paths return zero/`None` and fall back as described in section 1.2. `_drug_master()` reads it from the standardizer and returns `None` on absence/error. | `serving/predictor.py:832-836`, `serving/predictor.py:838-885`, `serving/predictor.py:904-1007`, `serving/predictor.py:1018-1025`, `serving/predictor.py:1211-1219` |
| SafetyNet | Missing/uninitialized module path returns Normal/no reasons/no alerts; an error from an injected initialized instance is propagated. | `serving/predictor.py:_run_safety_net`, `serving/predictor.py:252-262`, initialization at `serving/predictor.py:1261-1271` |
| DuplicateDetector | Missing/uninitialized module path returns `(0, [])`; an error from an injected initialized instance is propagated. | `serving/predictor.py:_run_duplicate_detector`, `serving/predictor.py:265-298`, initialization at `serving/predictor.py:1272-1277` |
| DL history provider | DL inference is skipped unless both bundle and provider are present; the primary final level still proceeds. | `serving/predictor.py:1418-1436`, `serving/predictor.py:1451-1517` |

## 7. Physical DataFrame and Parquet order risk

| Surface | Order contract and risk | Source |
|---|---|---|
| Online tabular/hierarchical | Bundle `feature_names`/`feature_cols` is the order authority. Dict-by-name alignment is safe for known names, but final conversion is positional; the frozenset is not an order contract. | `serving/predictor.py:45-59`, `serving/predictor.py:371-418`, `serving/predictor.py:1112-1142` |
| UI training | `FEATURE_COLS` is ordered and `df[_feature_cols]` is the selected training order. User selection can intentionally supply another ordered subset. | `hana_app/core/ml_runner.py:50-73`, `hana_app/core/ml_runner.py:1676-1758`, `hana_app/pages/3_🤖_모델_학습.py:1135-1140` |
| Feature-engineering Parquet | `FeatureEngineer.run()` begins with the physical ETL frame, appends CYP and temporal columns through left merges, then label/sex transformations, normalizer, and selector before writing `ml_features_{partition}.parquet`. Thus physical order is pipeline/merge/transform dependent, not defined by `ETL_NUMERIC_COLS` alone. | `scripts/features/feature_engineer.py:34-47`, `scripts/features/feature_engineer.py:80-144` |
| Dataset required tuple | `ML_DATASET_REQUIRED_COLUMNS` has a declared tuple order, but `validate_required_columns()` converts actual columns to a set and checks presence only; it does not enforce physical order. | `scripts/datasets/contracts.py:23-33`, `scripts/datasets/contracts.py:92-97` |

Consequently, equal logical feature sets do not prove equal physical DataFrame/Parquet order. Any consumer that bypasses name selection and uses positional values is exposed to train/serve skew; the current serving path avoids that only when bundle metadata carries the correct ordered names (`serving/predictor.py:1112-1142`; `hana_app/core/hierarchical_runner.py:752-766`).

## Appendix A — constant extractor and its output

This is the single extractor used for both Phase 0A Task 1 reports. It is self-contained (standard library only), imports no repository module, and writes nothing. Run it from the repository root on Python 3.12 at source snapshot commit `3d8d64e78601a3ff56dc38034a9da62853e6b656`:

```bash
git -C <repo> rev-parse HEAD          # expect 3d8d64e78601a3ff56dc38034a9da62853e6b656
python3.12 phase0a_extract.py .       # script body below; run from the repository root
```

### A.1 Script

```python
#!/usr/bin/env python3.12
"""Phase 0A Task 1 constant extractor (source-only, no repo imports).

Parses the cited modules with `ast` and evaluates the target assignments.
Handles `ast.Assign` and `ast.AnnAssign`, `frozenset()/set()/list()/tuple()`
wrapper calls, `|` set unions, and `+` sequence concatenation of already
extracted names. Prints a deterministic report: constants, counts, and the
set differences used by the Phase 0A reports.

Usage: python3.12 phase0a_extract.py [REPO_ROOT]
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

TARGETS: dict[str, list[str]] = {
    "serving/predictor.py": [
        "_BUILDER_KNOWN_COLS",
        "_INTENTIONAL_FEATURE_ALLOWLIST",
        "_FEATURE_ALLOWED",
    ],
    "hana_app/core/ml_runner.py": [
        "FEATURE_COLS",
        "RISK_LABEL_MAP",
        "_SAFE_MISCLS_FEATURES",
    ],
    "scripts/features/feature_engineer.py": ["ETL_NUMERIC_COLS"],
    "scripts/datasets/contracts.py": [
        "ML_DATASET_REQUIRED_COLUMNS",
        "DL_DATASET_REQUIRED_COLUMNS",
        "DL_BUNDLE_REQUIRED_FILES",
    ],
    "hana_app/core/hierarchical_runner.py": [
        "YELLOW_SUBTYPE_LABELS",
        "STAGE2_LABELS",
    ],
    "scripts/etl/prescription_aggregator.py": [
        "DDI_FEATURE_SEMANTICS_VERSION",
        "FEATURE_SEMANTICS_VERSION",
    ],
}

WRAPPERS = {"frozenset": frozenset, "set": set, "list": list, "tuple": tuple}


class ExtractError(ValueError):
    """The target assignment is not reducible to a literal value."""


def _eval(node: ast.expr, env: dict[str, object]) -> object:
    """Evaluate a constant expression: literals, wrapper calls, | and +."""
    if isinstance(node, ast.Name):
        if node.id not in env:
            raise ExtractError(f"unresolved name: {node.id}")
        return env[node.id]
    if isinstance(node, ast.Call):
        func = node.func
        if not isinstance(func, ast.Name) or func.id not in WRAPPERS:
            raise ExtractError(f"unsupported call: {ast.dump(node)[:80]}")
        if node.keywords or len(node.args) > 1:
            raise ExtractError(f"unsupported call args: {func.id}")
        inner = _eval(node.args[0], env) if node.args else ()
        return WRAPPERS[func.id](inner)  # type: ignore[arg-type]
    if isinstance(node, ast.BinOp):
        left, right = _eval(node.left, env), _eval(node.right, env)
        if isinstance(node.op, ast.BitOr):
            return left | right  # type: ignore[operator]
        if isinstance(node.op, ast.Add):
            return left + right  # type: ignore[operator]
        raise ExtractError(f"unsupported operator: {type(node.op).__name__}")
    return ast.literal_eval(node)


def extract(path: Path, names: list[str]) -> dict[str, tuple[object, int]]:
    """Return {name: (value, lineno)} for module-level Assign/AnnAssign targets."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    env: dict[str, object] = {}
    found: dict[str, tuple[object, int]] = {}
    wanted = set(names)
    for node in tree.body:  # module level only
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            if node.value is None or not isinstance(node.target, ast.Name):
                continue
            targets = [node.target.id]
            value = node.value
        else:
            continue
        for name in targets:
            try:
                resolved = _eval(value, env)
            except (ExtractError, ValueError):
                continue  # non-constant assignment (e.g. Path(...) expressions)
            env[name] = resolved  # keep for later name references
            if name in wanted:
                found[name] = (resolved, node.lineno)
    missing = wanted - found.keys()
    if missing:
        raise ExtractError(f"{path}: not extracted: {sorted(missing)}")
    return found


def render(value: object) -> str:
    """Deterministic rendering: sets sorted, sequences in declared order."""
    if isinstance(value, (frozenset, set)):
        if not value:
            return "set()  # empty"
        return f"{{{', '.join(repr(v) for v in sorted(value))}}}  # unordered, sorted for display"
    if isinstance(value, tuple):
        return f"({', '.join(repr(v) for v in value)})"
    if isinstance(value, list):
        return f"[{', '.join(repr(v) for v in value)}]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k!r}: {v!r}" for k, v in value.items()) + "}"
    return repr(value)


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    values: dict[str, object] = {}
    print("== constants ==")
    for rel in TARGETS:  # dict order = declared order above (deterministic)
        for name, (value, lineno) in sorted(extract(root / rel, TARGETS[rel]).items()):
            values[name] = value
            size = len(value) if isinstance(value, (list, tuple, set, frozenset, dict)) else "-"
            print(f"{rel}:{lineno} {name} n={size} {render(value)}")

    b = set(values["_BUILDER_KNOWN_COLS"])
    f = set(values["FEATURE_COLS"])
    e = set(values["ETL_NUMERIC_COLS"])
    d = set(values["ML_DATASET_REQUIRED_COLUMNS"])

    print("\n== counts ==")
    for key, s in (("B _BUILDER_KNOWN_COLS", b), ("F FEATURE_COLS", f),
                   ("E ETL_NUMERIC_COLS", e), ("D ML_DATASET_REQUIRED_COLUMNS", d)):
        print(f"{key}: {len(s)}")

    print("\n== union (presence matrix rows) ==")
    for name in sorted(b | f | e | d):
        cells = "".join("Y" if name in s else "-" for s in (b, f, e, d))
        print(f"{name} {cells}")

    print("\n== set differences ==")
    pairs = (("B", b, "F", f), ("F", f, "B", b), ("B", b, "E", e), ("E", e, "B", b),
             ("F", f, "E", e), ("E", e, "F", f), ("D", d, "E", e), ("E", e, "D", d))
    for ln, ls, rn, rs in pairs:
        diff = sorted(ls - rs)
        print(f"{ln} \\ {rn} (n={len(diff)}): {', '.join(diff) if diff else 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

### A.2 Output at commit `3d8d64e`

Verbatim stdout. The `== counts ==`, `== union ==`, and `== set differences ==` blocks are the evidence for the presence matrix and every set difference in [`phase0a-feature-dispersion-table.md`](./phase0a-feature-dispersion-table.md); the `== constants ==` block is the evidence for the sorted inventories, ordered lists, label tuples, and semantic-version strings in sections 1–5 above. Line numbers are the assignment statement's first line.

```text
== constants ==
serving/predictor.py:45 _BUILDER_KNOWN_COLS n=24 {'age', 'avg_drug_duration', 'cyp_high_risk_pairs', 'cyp_max_enzyme_risk', 'cyp_risk_score', 'ddi_contraindicated', 'ddi_major', 'ddi_minor', 'ddi_moderate', 'drug_count', 'drug_count_7d', 'dup_atc3', 'dup_atc4', 'dup_atc5', 'dup_efmdc', 'dup_same_ingredient', 'has_hepatic_risk_drug', 'has_high_risk_drug', 'has_renal_risk_drug', 'institution_count', 'long_term_drug_count', 'qt_risk_count', 'sex_m', 'triple_whammy'}  # unordered, sorted for display
serving/predictor.py:59 _FEATURE_ALLOWED n=24 {'age', 'avg_drug_duration', 'cyp_high_risk_pairs', 'cyp_max_enzyme_risk', 'cyp_risk_score', 'ddi_contraindicated', 'ddi_major', 'ddi_minor', 'ddi_moderate', 'drug_count', 'drug_count_7d', 'dup_atc3', 'dup_atc4', 'dup_atc5', 'dup_efmdc', 'dup_same_ingredient', 'has_hepatic_risk_drug', 'has_high_risk_drug', 'has_renal_risk_drug', 'institution_count', 'long_term_drug_count', 'qt_risk_count', 'sex_m', 'triple_whammy'}  # unordered, sorted for display
serving/predictor.py:58 _INTENTIONAL_FEATURE_ALLOWLIST n=0 set()  # empty
hana_app/core/ml_runner.py:50 FEATURE_COLS n=22 ['drug_count', 'drug_count_7d', 'institution_count', 'ddi_contraindicated', 'ddi_major', 'ddi_moderate', 'ddi_minor', 'triple_whammy', 'qt_risk_count', 'dup_same_ingredient', 'dup_atc5', 'dup_atc4', 'dup_atc3', 'dup_efmdc', 'has_high_risk_drug', 'has_renal_risk_drug', 'has_hepatic_risk_drug', 'cyp_risk_score', 'cyp_max_enzyme_risk', 'cyp_high_risk_pairs', 'age', 'sex_m']
hana_app/core/ml_runner.py:75 RISK_LABEL_MAP n=4 {'Red': 3, 'Yellow': 2, 'Green': 1, 'Normal': 0}
hana_app/core/ml_runner.py:91 _SAFE_MISCLS_FEATURES n=20 ['drug_count', 'drug_count_7d', 'institution_count', 'ddi_contraindicated', 'ddi_major', 'ddi_moderate', 'ddi_minor', 'triple_whammy', 'qt_risk_count', 'dup_same_ingredient', 'dup_atc5', 'dup_atc4', 'dup_atc3', 'dup_efmdc', 'has_high_risk_drug', 'has_renal_risk_drug', 'has_hepatic_risk_drug', 'cyp_risk_score', 'cyp_max_enzyme_risk', 'cyp_high_risk_pairs']
scripts/features/feature_engineer.py:37 ETL_NUMERIC_COLS n=14 ['drug_count', 'drug_count_7d', 'institution_count', 'ddi_contraindicated', 'ddi_major', 'ddi_moderate', 'ddi_minor', 'triple_whammy', 'qt_risk_count', 'dup_same_ingredient', 'dup_atc5', 'dup_atc4', 'dup_atc3', 'age']
scripts/datasets/contracts.py:41 DL_BUNDLE_REQUIRED_FILES n=6 ('model.pt', 'model_config.json', 'drug_vocab.json', 'edge_index.pt', 'feature_normalizer.pkl', 'schema_version.json')
scripts/datasets/contracts.py:35 DL_DATASET_REQUIRED_COLUMNS n=3 ('patient_id', 'drug_code', 'prescription_date')
scripts/datasets/contracts.py:23 ML_DATASET_REQUIRED_COLUMNS n=9 ('patient_id', 'drug_count', 'drug_count_7d', 'institution_count', 'ddi_contraindicated', 'ddi_major', 'ddi_moderate', 'ddi_minor', 'risk_level')
hana_app/core/hierarchical_runner.py:31 STAGE2_LABELS n=7 ('Y_TRIPLE', 'Y_DOUBLE', 'Y_DDI_MAJOR', 'Y_DDI_MOD', 'Y_DUP', 'Y_FRAG', 'No_Alert')
hana_app/core/hierarchical_runner.py:28 YELLOW_SUBTYPE_LABELS n=6 ('Y_TRIPLE', 'Y_DOUBLE', 'Y_DDI_MAJOR', 'Y_DDI_MOD', 'Y_DUP', 'Y_FRAG')
scripts/etl/prescription_aggregator.py:209 DDI_FEATURE_SEMANTICS_VERSION n=- 'ddi.v2'
scripts/etl/prescription_aggregator.py:216 FEATURE_SEMANTICS_VERSION n=- 'rulefeat.v1'

== counts ==
B _BUILDER_KNOWN_COLS: 24
F FEATURE_COLS: 22
E ETL_NUMERIC_COLS: 14
D ML_DATASET_REQUIRED_COLUMNS: 9

== union (presence matrix rows) ==
age YYY-
avg_drug_duration Y---
cyp_high_risk_pairs YY--
cyp_max_enzyme_risk YY--
cyp_risk_score YY--
ddi_contraindicated YYYY
ddi_major YYYY
ddi_minor YYYY
ddi_moderate YYYY
drug_count YYYY
drug_count_7d YYYY
dup_atc3 YYY-
dup_atc4 YYY-
dup_atc5 YYY-
dup_efmdc YY--
dup_same_ingredient YYY-
has_hepatic_risk_drug YY--
has_high_risk_drug YY--
has_renal_risk_drug YY--
institution_count YYYY
long_term_drug_count Y---
patient_id ---Y
qt_risk_count YYY-
risk_level ---Y
sex_m YY--
triple_whammy YYY-

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

The extractor covers constants only. Behavioral facts in this report — defaults, thresholds, guards, fallbacks, reload/rollback, and DataFrame order — were read from the cited lines and are not derivable from this output.
