# Profile Contract Specifications (Phase 2A)

**Status:** Frozen. This document codifies current behavior. It changes nothing.
**Plan:** `docs/superpowers/plans/2026-07-12-phase2-contract-spec-and-characterization.md` Wave 1 Task 1
**Authority:** This frozen document is authoritative for the recorded profile contracts based on its cited source snapshot and Phase 0A/0B baseline evidence. `docs/superpowers/specs/2026-07-12-opencode-lo-contract-design.md` is cited upstream historical provenance only.
**Baseline evidence:** `docs/superpowers/reports/contract-baseline/` (Phase 0A profile contract map, Phase 0A feature dispersion table, Phase 0A bundle metadata record, Phase 0B dependency graph, Phase 0B baseline report)
**Source snapshot:** commit `3d8d64e` (`3d8d64e78601a3ff56dc38034a9da62853e6b656`), with `serving/`, `hana_app/`, and `scripts/` unmodified in the working tree.
**Policy baseline:** `RESEARCH_TRACK_FROZEN` (`AGENTS.md`, `CLAUDE.md`). No frozen holdout (Nov to Dec) was read, and no protected artifact (`packages_win/py312/`, `mlruns/`, generated `.parquet`, `out/`) was opened for this document.

## 0. How to read this document

Four profiles operate today: `tabular_binary`, `hierarchical`, `ui_experimental`, and `dl_history`. Each holds its own feature set, label space, thresholds, semantic versions, and validation rules. **Those differences are the contract, not a defect.** This specification records them. It does not flatten them, does not merge any feature list into another, and does not target zero diff between sources.

Where the plan's draft text, an older report, or a source comment disagreed with the code at `3d8d64e`, the code and the approved Phase 0A/0B baseline govern. Section 8 lists every such correction explicitly so that a reader who compares this spec against the plan draft can see what changed and why.

Rules for anyone touching this file or the code it describes:

- Do not change feature names, order, dtype or default semantics, labels, thresholds, semantic version constants, or artifact/reload contracts.
- Do not merge `FEATURE_COLS` into `_BUILDER_KNOWN_COLS`, or any other cross-profile flattening.
- Do not remove the `serving -> hana_app.core.hierarchical_runner` dependency. That removal is Phase 3, out of scope (Section 7).
- Do not copy or move `predict_risk`, `ACTION_BY_LABEL`, or `STAGE2_LABELS` into `serving/`.
- Do not migrate artifacts, retrain, or split `serving/predictor.py`.

---

## 1. Profile: tabular_binary

Single ML model inference (XGBoost, LightGBM, Ensemble, or EnsembleTrainer3Way).

### 1.1 Feature source

`serving/predictor.py` `_BUILDER_KNOWN_COLS`, a `frozenset[str]` of 24 names (`serving/predictor.py:45-53`). It is a **membership set, not an order contract**.

Sorted for inventory only:

```text
age, avg_drug_duration, cyp_high_risk_pairs, cyp_max_enzyme_risk,
cyp_risk_score, ddi_contraindicated, ddi_major, ddi_minor, ddi_moderate,
drug_count, drug_count_7d, dup_atc3, dup_atc4, dup_atc5, dup_efmdc,
dup_same_ingredient, has_hepatic_risk_drug, has_high_risk_drug,
has_renal_risk_drug, institution_count, long_term_drug_count,
qt_risk_count, sex_m, triple_whammy
```

- `_INTENTIONAL_FEATURE_ALLOWLIST`: empty `frozenset()` (`serving/predictor.py:58`).
- `_FEATURE_ALLOWED = _BUILDER_KNOWN_COLS | _INTENTIONAL_FEATURE_ALLOWLIST` (`serving/predictor.py:59`), therefore currently equal to `_BUILDER_KNOWN_COLS` (24 names).

### 1.2 Feature order

The **model bundle's `feature_names` list is the sole order authority**. `MLModel.load()` reads it from bundle state; `RequestFeatureBuilder.build()` then constructs an insertion-ordered dict in that sequence (`aligned = {name: feat.get(name, 0.0) for name in feature_names}`) and converts a one-row DataFrame to a float vector (`serving/predictor.py:371-418`, `serving/predictor.py:1112-1142`).

Because the final conversion is positional, a bundle carrying wrong or reordered names produces a silently misaligned vector. Name alignment protects only bundles whose metadata is correct.

### 1.3 dtype and default semantics

`feat` is declared `dict[str, float]`; counts and boolean-like flags are cast to float, and the vector is `astype(float)` (`serving/predictor.py:1029-1048`, `serving/predictor.py:1085-1110`, `serving/predictor.py:1141-1142`).

| Feature group | Current default / fallback | Source |
|---|---|---|
| `age` | `float(patient_age or 0)`, so `0.0` when absent | `serving/predictor.py:1038` |
| `sex_m` | `1.0` for `M`, `0.0` for a supplied non-`M` value, `0.5` when absent. Request domain is `M`/`F` only, enforced by Pydantic. | `serving/predictor.py:1039`; `serving/schemas.py:76` |
| `ddi_contraindicated`, `ddi_major`, `ddi_moderate`, `ddi_minor` | Float severity counts. All four are `0.0` when the standardizer, DDI matrix, or DrugMaster is unavailable, or when fewer than two mapped records exist. | `serving/predictor.py:869-885`, `serving/predictor.py:1041-1048` |
| `drug_count`, `drug_count_7d`, duplicate counts | EDI to WK/DrugMaster parity path when available; otherwise EDI/ATC-derived counts, with `dup_efmdc=0.0` and `drug_count_7d` falling back to `drug_count`. | `serving/predictor.py:904-953`, `serving/predictor.py:1029-1074`, `serving/predictor.py:1107-1110` |
| `institution_count` | Distinct non-empty institution IDs, as float. | `serving/predictor.py:1034-1037` |
| `avg_drug_duration`, `long_term_drug_count` | Mean duration, or `0.0` for no drugs; count of durations >= 30 days. | `serving/predictor.py:1050-1053` |
| `cyp_risk_score`, `cyp_max_enzyme_risk`, `cyp_high_risk_pairs` | Extracted when a CYP extractor and ATC codes exist; otherwise all three are `0.0`. | `serving/predictor.py:1076-1083` |
| `triple_whammy`, `has_high_risk_drug`, `has_renal_risk_drug`, `has_hepatic_risk_drug` | Component/DrugMaster path only when rule semantics are active (Section 2.5); otherwise a name/ATC fallback. Each cast to float. | `serving/predictor.py:1085-1105` |
| Unknown bundle feature name | `feat.get(name, 0.0)`. Schema validation normally prevents this; active lenient mode records the degraded load. | `serving/predictor.py:93-139`, `serving/predictor.py:1112-1116` |

### 1.4 Labels and threshold

Serving label space is `RiskLevel`: `Red`, `Yellow`, `Green`, `Normal`, ordered 3 to 0 (`serving/schemas.py:22-37`).

`MLModel._threshold` initializes to `0.5` and loads bundle `best_threshold` with the same `0.5` fallback. `MLModel.classify()` (`serving/predictor.py:611-619`) uses proportions of that threshold `t`:

| Condition | Level |
|---|---|
| `prob >= t` | RED |
| `prob >= t * 0.6` | YELLOW |
| `prob >= t * 0.3` | GREEN |
| otherwise | NORMAL |

Response `intervention` is selected from the **final hybrid level** via `INTERVENTION_MAP` (`serving/schemas.py:283-290`): Red `즉각 개입`, Yellow `복약 상담`, Green and Normal `관여 안 함`.

### 1.5 Semantic version guard

`DDI_FEATURE_SEMANTICS_VERSION = "ddi.v2"` (`scripts/etl/prescription_aggregator.py:209`), meaning WK to DrugMaster to DB-code severity over overlapping pairs.

The guard fires **only when at least one loaded feature name starts with `ddi_`**. In that case the bundle's `ddi_feature_semantics_version` must equal `"ddi.v2"`. Missing or mismatched rejects the load, clearing `_model` and `_feature_names` (`serving/predictor.py:389-406`). This is a hard load rejection, motivated by the commit `d201743` train/serve skew precedent.

### 1.6 Schema validation and the lenient escape hatch

`_validate_feature_schema()` (`serving/predictor.py:93-139`) checks `set(feature_names) - _FEATURE_ALLOWED`. Any unknown name rejects the load (strict default), preventing silent `0.0` drift.

The lenient escape hatch is doubly gated:

- `FEATURE_SCHEMA_LENIENT` must be `1`, `true`, or `yes` (case-insensitive, stripped).
- **and** today must be strictly before the sunset. `_FEATURE_SCHEMA_LENIENT_SUNSET_DEFAULT = date(2026, 8, 1)` (`serving/predictor.py:65`). `FEATURE_SCHEMA_LENIENT_SUNSET_DATE=YYYY-MM-DD` overrides it; an invalid date **fails closed to strict** (`serving/predictor.py:68-90`).

When lenient is active, unknown columns load with a `0.0` fallback and are recorded as `_schema_drift`. Once `today >= sunset`, lenient is ignored and the bundle is strictly rejected even with the variable set.

Observed on 2026-07-12 in the implementer's WSL environment: both variables unset, so lenient is inactive. That is not evidence about the Windows production environment.

### 1.7 Artifact format

Pickle `.pkl` with a mandatory SHA-256 sidecar `.pkl.sha256`. Integrity is **fail-closed**: a missing hash file rejects the load (`serving/predictor.py:322-342`). The file is read once and the same bytes are verified and deserialized, closing the TOCTOU window.

**There is no single fixed key list for every tabular bundle.** The payload depends on which trainer wrote it, and the loader reads a superset with per-key defaults. Both facts are recorded in the Phase 0A bundle metadata record (`docs/superpowers/reports/contract-baseline/phase0a-bundle-metadata-record.md`, Section 1.1).

Keys written by each trainer's `save()` (`scripts/train/trainer.py`):

| Writer | Top-level payload keys | Model sidecars written, excluding the mandatory top-level `.sha256` integrity sidecar |
|---|---|---|
| `BaseTrainer` (XGBoost, LightGBM single trainers) | `model`, `params`, `feature_importances`, `best_threshold`, `trainer_class`, plus `**_extra_meta` | none beyond the top-level bundle and its `.sha256` |
| `EnsembleTrainer` (`scripts/train/trainer.py:272-290`) | `trainer_class`, `weights`, `best_threshold`, `feature_importances`, plus `**_extra_meta`. **No top-level `model`, no `params`.** | `.xgb.pkl`, `.lgb.pkl`, each written through `BaseTrainer.save()` and therefore each carrying its own `model`/`params`/`feature_importances`/`best_threshold`/`trainer_class` payload and its own `.sha256` |
| `EnsembleTrainer3Way` (`scripts/train/trainer.py:449-475`) | `trainer_class`, `weights`, `best_threshold`, `feature_importances`, plus `**_extra_meta`. **No top-level `model`, no `params`.** | `.xgb.pkl`, `.lgb.pkl`, and `gat_model.pt` in the model's parent directory. `gat_model.pt` is **omitted with a warning when the GAT sub-model was never trained** (`scripts/train/trainer.py:456-462`), while the loader treats it as mandatory for this trainer class (`serving/predictor.py:491-500`) |

The ensemble top-level payloads carry no `model` because the estimators live in the two sidecars; serving reconstructs a local `_EnsembleWrapper` from `xgb_state["model"]` and `lgb_state["model"]` after both are unpickled (`serving/predictor.py:454-486`). `MLModel.load` reads `model` with `state.get`, so an ensemble payload legitimately yields `self._model is None`.

**`_extra_meta` stamping.** The feature-contract keys are not written by the trainer classes themselves. `scripts/train/pipeline.py:150-158` sets `trainer._extra_meta` immediately before `trainer.save()`, and every `save()` splats it into the payload (`scripts/train/trainer.py:89`, `:283`, `:468`). The stamped keys are:

```text
artifact_version = 2, feature_names, scaler_path, selector_path,
ddi_feature_semantics_version
```

`scaler_path`/`selector_path` are stamped as paths relative to the model directory. Sub-trainers invoked from an ensemble `save()` have no `_extra_meta`, so the `.xgb.pkl`/`.lgb.pkl` sidecars carry no feature-contract keys; only the top-level bundle does.

Keys read by `MLModel.load` (`serving/predictor.py:378-397`, `:424-469`), with defaults: `model` (None), `best_threshold` (`0.5`), `trainer_class` (`"unknown"`), `feature_names` (`[]`), `artifact_version` (`1`), `partition` (None), `ddi_feature_semantics_version` (only consulted when a `ddi_` feature name is present, Section 1.5), `scaler_path`, `selector_path`, `weights` (ensemble). **`partition` is read but is not written by any current training path**, so it is loader-tolerated legacy metadata, not a contract the trainers satisfy today.

Sidecar artifacts (scaler, selector) named in state are resolved relative to the model directory, defended against path traversal via `relative_to(model_dir.resolve())`, hash-verified, and applied to instance state only after every sidecar passes. Absence, hash mismatch, or traversal all reject the whole model load (`serving/predictor.py:344-450`).

`EnsembleTrainer`/`EnsembleTrainer3Way` bundles additionally load `.xgb.pkl` and `.lgb.pkl` sub-models, each hash-verified.

### 1.8 Reload and rollback

`HybridPredictor.reload_model(model_path)` constructs a **new** `MLModel`, calls `load()`, and swaps into `self._ml` under `_ml_lock` only when `load()` returns True (`serving/predictor.py:1286-1294`). A False return leaves the previous model in place: rollback by retention.

### 1.9 Production path

`RequestFeatureBuilder.build()` -> `MLModel.predict_proba()` -> `MLModel.classify()` (`serving/predictor.py:1405-1416`). The current model is snapshotted under the lock before use.

---

## 2. Profile: hierarchical

Stage 1 Red binary plus Stage 2 Yellow-subtype 7-class classifier.

### 2.1 Feature source and order

`stage_meta.json` `feature_cols` (`list[str]`) is the **ordered** contract, written from the training function's `feature_cols` argument (`hana_app/core/hierarchical_runner.py:463-485`, `serving/predictor.py:652-669`). The serving builder emits a float NumPy vector in exactly that order, and `predict_risk()` requires the training column order (`serving/predictor.py:1112-1142`; `hana_app/core/hierarchical_runner.py:744-766`).

dtype and default semantics are inherited from the online builder (Section 1.3): this profile shares `RequestFeatureBuilder`.

### 2.2 Label space

```python
YELLOW_SUBTYPE_LABELS = ("Y_TRIPLE", "Y_DOUBLE", "Y_DDI_MAJOR", "Y_DDI_MOD", "Y_DUP", "Y_FRAG")   # 6
STAGE2_LABELS         = YELLOW_SUBTYPE_LABELS + ("No_Alert",)                                       # 7
```

Source: `hana_app/core/hierarchical_runner.py:28-31`. Order is significant: `predict_risk()` maps local class indices to global label slots by position.

Serving maps `Red` to RED, any `Y_*` to YELLOW, and `No_Alert` to NORMAL (`serving/predictor.py:1346-1358`).

### 2.3 Label integrity guards

Three independent guards, all rejecting the load:

1. **Bundle metadata guard.** `stage_meta.json["stage2_labels"]` must be present and exactly equal to the current `STAGE2_LABELS`, order included. Checked before any joblib deserialization (fail fast). Missing or mismatched clears state and rejects (`serving/predictor.py:670-685`).
2. **Encoder guard.** After deserialization, `[str(c) for c in encoder.classes_]` must equal `list(STAGE2_LABELS)` exactly (`serving/predictor.py:736-757`).
3. **`classes_present` range guard.** Every index must satisfy `0 <= gi < len(STAGE2_LABELS)` (same block).

A 6-class legacy bundle (for example, one carrying `Y_MIX`) would misalign the local-to-global slot mapping and cause silent train/serve skew. These guards exist to reject it.

### 2.4 Thresholds and dispatch

`stage_meta.json["thresholds"]` supplies `tau_red` and `tau_review` (floats). `_dispatch_result()` (`hana_app/core/hierarchical_runner.py:705-741`):

| Condition | Result |
|---|---|
| `p_red >= tau_red` | `risk_level="Red"`, Stage 2 skipped, `stage2_probs=None`, `red_suspect=False`, `action=RED_ACTION` |
| `tau_review <= p_red < tau_red` | Stage 2 argmax label, `red_suspect=True` |
| `p_red < tau_review` | Stage 2 argmax label, `red_suspect=False` |

When Stage 2 runs, `stage2_probs` is returned as a **dict** `{label: prob}` over `STAGE2_LABELS`, not a raw array. `action` is `ACTION_BY_LABEL.get(label, "알림 없음")`.

**Degraded Stage 1.** If training data has fewer than 10 Red or fewer than 10 non-Red rows, Stage 1 is replaced by a constant non-Red dummy with fixed `tau_red=1.0`, `tau_review=0.5`, and `p_red` always 0, recorded as `stage_meta.stage1_trained=False` (`hana_app/core/hierarchical_runner.py:497-517`). Such a bundle never emits Red from the model. The invariant `tau_review < tau_red` holds.

### 2.5 Semantic versions

Two constants apply, and **they are enforced differently**:

| Constant | Value | Enforcement |
|---|---|---|
| `DDI_FEATURE_SEMANTICS_VERSION` | `ddi.v2` | **Hard load rejection.** `stage_meta.json["ddi_feature_semantics_version"]` must equal it; missing or mismatched clears state and rejects (`serving/predictor.py:686-700`). |
| `FEATURE_SEMANTICS_VERSION` | `rulefeat.v1` | **Runtime feature-path gate only. No load rejection exists in current code.** At prediction time, `HybridPredictor.predict()` compares the bundle's `feature_semantics_version` to the constant and sets `rule_features_active` on equality (`serving/predictor.py:1385-1396`, property at `serving/predictor.py:780-782`). |

When `rule_features_active` is True, `triple_whammy` and the three risk-drug flags are computed through the EDI to WK to DrugMaster component path. When it is False (an older bundle without the version), the builder uses the legacy name/ATC fallback path, which can produce nonzero values. The gate selects the feature-generation semantics rather than rejecting the bundle.

The source comment at `scripts/etl/prescription_aggregator.py:211-216` describes a reload guard that rejects missing or old `rulefeat` bundles. **No such guard exists in `serving/predictor.py` at this snapshot.** The runtime gate above is the authoritative observed behavior. This discrepancy is recorded, not fixed, in Phase 2.

### 2.6 Load validation chain (actual order)

`HierarchicalPredictor.load()` (`serving/predictor.py:652-760`) executes in this order. Every failure calls `_clear_state()` and returns False.

1. File existence: `stage_meta.json`, `stage1_red.joblib`, `stage2_yellow.joblib`.
2. Read `stage_meta.json`; take `thresholds` and `feature_cols`.
3. Stage 2 label space guard against `STAGE2_LABELS` (before joblib load).
4. DDI semantic version guard against `ddi.v2`.
5. `feature_cols` non-empty.
6. `_validate_feature_schema(feature_cols)` against `_FEATURE_ALLOWED`.
7. SHA-256 verification of both joblib files against `stage1_sha256` and `stage2_sha256`. A **missing hash in metadata is fail-closed** (rejected, not skipped).
8. `joblib.load()` of both stages.
9. Encoder `classes_` and `classes_present` integrity guard.

### 2.7 Intervention actions

`ACTION_BY_LABEL` (`hana_app/core/hierarchical_runner.py:690-702`):

| Label | Action |
|---|---|
| `Y_DDI_MAJOR` | `약사 전화` |
| `Y_TRIPLE` | `문자 안내` |
| `Y_DOUBLE` | `모니터링` |
| `Y_DDI_MOD` | `모니터링` |
| `Y_DUP` | `모니터링` |
| `Y_FRAG` | `모니터링` |
| `No_Alert` | `관여 안 함` |

`RED_ACTION = "즉각 개입"` is **not** a key of `ACTION_BY_LABEL`, because Red is dispatched by `risk_level`/Stage 1, not as a Stage 2 label (`hana_app/core/hierarchical_runner.py:685-688`).

The response's level-wide `intervention` still comes from the four-level `INTERVENTION_MAP`. Subtype `action` is an additional field, and for Yellow it carries the operational meaning.

### 2.8 Deterministic backstops

Both are model-independent and can only **escalate**, never downgrade:

- `red_triggers()` returns `RED_CONTRAINDICATED` for contraindicated DDI, forcing Red (`serving/predictor.py:987`, `serving/predictor.py:1438-1476`).
- `rule_floor()` guarantees a subtype floor: major DDI count >= 1 gives `Y_DDI_MAJOR`; a severe-immediate trigger (triple_whammy, 10 drugs plus high-risk, elderly plus long-term) gives `Y_TRIPLE` (`serving/predictor.py:998`).

### 2.9 Artifact format

**`stage_meta.json`.** Current training writes **14 keys** (`hana_app/core/hierarchical_runner.py:623-642`). The full list, in write order:

| # | Key | Read by serving? |
|---|---|---|
| 1 | `clinical_standards_version` | no |
| 2 | `ddi_feature_semantics_version` | yes, hard load guard (Section 2.5) |
| 3 | `feature_semantics_version` | yes, runtime feature-path gate only (Section 2.5) |
| 4 | `feature_cols` | yes, ordered feature contract (Section 2.1) |
| 5 | `thresholds` (`tau_red`, `tau_review`) | yes, dispatch (Section 2.4) |
| 6 | `stage2_labels` | yes, label-space guard (Section 2.3) |
| 7 | `stage2_label_counts` | no |
| 8 | `y_other_excluded_count` | no |
| 9 | `stage1_sha256` | yes, fail-closed hash check (Section 2.6) |
| 10 | `stage2_sha256` | yes, fail-closed hash check (Section 2.6) |
| 11 | `cost_sensitive` | no |
| 12 | `cost_ratio_by_class` | no |
| 13 | `stage1_trained` | no. Deployment review signal only: `False` marks the degraded constant Stage 1 (Section 2.4) |
| 14 | `stage1_red_count` | no |

Keys 1, 7, 8, 11, 12, 13, and 14 are written but never consulted by `HierarchicalPredictor.load()`. They are provenance and review metadata. Serving ignores them; do not treat their absence in an older bundle as a load-blocking condition, and do not treat `stage1_trained=False` as an automatic serving rejection, because no such check exists.

- `stage1_red.joblib`: Stage 1 estimator, or the module-level `_ConstantNegativeStage1` dummy when Stage 1 degrades (`hana_app/core/hierarchical_runner.py:40-59`, `:512-517`). That class path is a pickle compatibility obligation: see the Phase 0A bundle metadata record, Section 4.1.
- `stage2_yellow.joblib`: dict with **four** keys, `model`, `encoder`, `stage2_classes_global`, and `classes_present` (`hana_app/core/hierarchical_runner.py:605-613`). `stage2_classes_global` and `classes_present` currently receive the same `classes_present.tolist()` value; serving reads only `model`, `encoder`, and `classes_present` (`serving/predictor.py:737-757`).

### 2.10 Reload and rollback

`HybridPredictor.reload_hierarchical(model_dir)` (`serving/predictor.py:1296-1315`) constructs a new `HierarchicalPredictor`, calls `load()`, then **re-checks** non-empty `feature_cols` and re-runs `_validate_feature_schema` before swapping under `_hier_lock`. Any failure returns False with the previous object retained.

### 2.11 Production path and the serving to hana_app dependency

`RequestFeatureBuilder.build()` -> `HierarchicalPredictor.predict_risk_single()` -> `hana_app.core.hierarchical_runner.predict_risk()` (`serving/predictor.py:1385-1404`).

**`serving/` depends on `hana_app/` at runtime.** Three lazy import edges:

| Serving location | Symbol | Timing |
|---|---|---|
| `HierarchicalPredictor.load`, line 674 | `STAGE2_LABELS` | Bundle load |
| `HierarchicalPredictor.predict_risk_single`, line 788 | `predict_risk` | Request |
| `HybridPredictor.predict`, line 1470 | `ACTION_BY_LABEL` | Request, only when a subtype floor fires |

`hana_app/core/hierarchical_runner.py:21-25` also mutates `sys.path` at import time. The serving edge therefore crosses both a UI/domain ownership boundary and a process-global import-path boundary.

**This dependency is recorded here and NOT removed in Phase 2.** Removal is Phase 3 (Section 7).

---

## 3. Profile: ui_experimental

Page 3 Streamlit training path. **Not a production serving path.**

### 3.1 Feature source and order

`hana_app/core/ml_runner.py` `FEATURE_COLS`, an **ordered** `list[str]` of 22 names (`hana_app/core/ml_runner.py:50-73`). It is the training default and the Page 3 multiselect order.

Physical order as declared:

```text
drug_count, drug_count_7d, institution_count, ddi_contraindicated, ddi_major,
ddi_moderate, ddi_minor, triple_whammy, qt_risk_count, dup_same_ingredient,
dup_atc5, dup_atc4, dup_atc3, dup_efmdc, has_high_risk_drug,
has_renal_risk_drug, has_hepatic_risk_drug, cyp_risk_score,
cyp_max_enzyme_risk, cyp_high_risk_pairs, age, sex_m
```

The user may deselect features in the UI, supplying a different ordered subset. `df[_feature_cols]` is the selected training order.

### 3.2 dtype and default semantics (differ from serving)

`_patient_features_to_row()` (`hana_app/core/ml_runner.py:438-465`) preserves native numeric values, casts boolean flags to `int`, and:

| Item | ui_experimental | tabular_binary / serving |
|---|---|---|
| Missing `age` | `-1` | `0.0` |
| `sex_m` input domain | Raw HANA sex strings `"1"` and `"2"` | Request `M`/`F`, Pydantic-enforced |
| `sex_m` encoding | `1.0` if `"1"`, `0.0` if `"2"`, else `0.5` | `1.0` if `M`, `0.0` if a supplied non-`M`, else `0.5` |

Both producers emit the same `1.0`/`0.0`/`0.5` codes from **different input vocabularies**. Any code that equates the two domains without mapping will misencode. This is a contract distinction, recorded as such.

`sex_type` is an additional column carrying the raw HANA sex value. It is **metadata, not a `FEATURE_COLS` model input**, and must not be fed to a model as a feature.

### 3.3 Labels

`RISK_LABEL_MAP` (`hana_app/core/ml_runner.py:75-80`): `Red=3`, `Yellow=2`, `Green=1`, `Normal=0`.

`risk_binary` in the UI row is `1` for **both Red and Yellow**. This differs from `scripts/features/feature_engineer.py`'s `is_high_risk`, which is `1` for Red only (`scripts/features/feature_engineer.py:45-47`, `:120-123`). Recorded, not reconciled.

UI training supports `risk_binary` or four-class `risk_label`. Optional binary threshold optimization is **off by default** and uses `0.5` unless explicitly run (`hana_app/core/ml_runner.py:1676-1707`, `:2084-2096`, `:2417-2438`).

### 3.4 Training path

`ml_runner.py` -> `aggregate_patient_features()` -> `FeatureEngineer` -> trainer -> joblib save. Includes optional stratification, train/test split, and cross-validation with metrics returned to the UI (`hana_app/core/ml_runner.py:823-882`, `:1676-1717`, `:1863-1963`).

### 3.5 Safety guards

`MemoryGuard` is active in both feature construction and training. It escalates at 80/90/95 percent and raises `MemoryLimitExceeded` at the hard stop (`hana_app/core/memory_guard.py:68-99`, `:131-195`; used at `hana_app/core/ml_runner.py:832`, `:1050`, `:1729`).

`hana_app/core/page_guards.py` defines HANA validation helpers but **Page 3 does not import it** in current source; live/local mode handling is inline. It must not be represented as an active Page 3 dependency.

### 3.6 Operational separation

This path does not load or connect to production serving bundles. It has no serving intervention mapping and no hot-reload/rollback contract of its own; those live in the operational predictor and response schema.

`_SAFE_MISCLS_FEATURES` contains 20 of the 22 UI features, excluding `age` and `sex_m`, and is used to JSON-normalize present non-null values for misclassification case reporting (`hana_app/core/ml_runner.py:91-121`).

---

## 4. Profile: dl_history

Operational DL bundle (graph neural network). Auxiliary inference.

### 4.1 Bundle contract

`DL_BUNDLE_REQUIRED_FILES` (`scripts/datasets/contracts.py:41-48`), 6 files:

```text
model.pt, model_config.json, drug_vocab.json,
edge_index.pt, feature_normalizer.pkl, schema_version.json
```

`MANIFEST.json` (`DL_MANIFEST_FILE`) carries `track` (`"dl"`), `run_id`, `schema_version`, `created_at`, `hash_alg` (`"sha256"`), `lookback_days`, `drug_vocab_sha256`, `edge_index_sha256`, and `files`. `validate_dl_bundle_manifest()` checks track, run/schema IDs, hash algorithm, required files, and hashes before loading (`scripts/datasets/contracts.py:136-176`, `:201-258`).

### 4.2 Dataset contract

`DL_DATASET_REQUIRED_COLUMNS = ("patient_id", "drug_code", "prescription_date")` (`scripts/datasets/contracts.py:35-39`), an **event-level** contract. It is disjoint (apart from `patient_id`) from `ML_DATASET_REQUIRED_COLUMNS`, which is patient-level tabular.

### 4.3 Encoding, architecture, lookback

- `_SUPPORTED_ENCODING_STRATEGIES = {"multi_hot"}` (`serving/dl_predictor.py:31`). `"count"` was removed as dead infrastructure: it had no training path, so accepting it would silently admit misconfigured bundles. Input is a float list initialized to `0.0`; observed drug positions become `1.0`; OOV drugs use `_unk` when the vocabulary has it, otherwise they are dropped with a warning (legacy vocabularies).
- `_GRAPH_ARCHITECTURES = {"gat", "gcn"}` (`serving/dl_predictor.py:32`). These receive `edge_index`; other configured architectures use the single-tensor call.
- `LOOKBACK_DAYS_DEFAULT = 365`, `LOOKBACK_DAYS_MIN = 7`, `LOOKBACK_DAYS_MAX = 1825` (`scripts/datasets/contracts.py:53-55`). Artifact and runtime lookback must be **equal**, or `validate_lookback_consistency()` raises `LookbackMismatchError` (`scripts/datasets/contracts.py:100-125`).

### 4.4 Output and operational impact

Output labels come from `model_config.json`. Inference takes softmax/sigmoid probabilities and selects the maximum-probability label. **There is no decision threshold and no action/intervention mapping in this profile.**

**DL results do not affect the final `risk_level`.** The final level is computed from the rule/tabular or hierarchical path first; `dl_prediction` is then attached to the response as an auxiliary field, and `dl_error` captures any failure while the Rule/ML response is still returned (`serving/predictor.py:1375-1436`, `:1451-1517`).

### 4.5 Reload and rollback (weaker than the other two)

`HybridPredictor.reload_dl(bundle_dir)` (`serving/predictor.py:1317-1328`) constructs a new `DLModel` with the same `runtime_lookback_days`, calls `load()`, then **unconditionally** swaps under `_dl_lock` and returns `True`.

It does **not** branch on `load()`'s boolean return. Its rollback contract therefore depends entirely on `DLModel.load()` raising on every failure, which it currently does: `load()` validates manifest, hash, and lookback first and updates instance state only after every check passes, so an invalid bundle raises its original validation exception rather than returning False (`serving/dl_predictor.py:102-121`).

Recorded risk: a future `load()` that returned False without raising would be swapped in and reported successful. Do not rely on `reload_dl()`'s return value as a validation signal.

### 4.6 Production path

`HANAHistoryProvider.fetch_patient_history()` -> `DLModel.predict()`. DL inference is skipped unless both a validated bundle and a history provider are present; the primary final level proceeds regardless (`serving/predictor.py:1418-1436`).

---

## 5. Cross-profile feature set differences

These differences are **intentional and preserved**. This section records them. It does not remove or flatten them.

Four sources:

| Key | Symbol | Container | Count | Location |
|---|---|---|---|---|
| B | `_BUILDER_KNOWN_COLS` | unordered `frozenset` | 24 | `serving/predictor.py:45-53` |
| F | `FEATURE_COLS` | ordered `list` | 22 | `hana_app/core/ml_runner.py:50-73` |
| E | `ETL_NUMERIC_COLS` | ordered `list` | 14 | `scripts/features/feature_engineer.py:37-43` |
| D | `ML_DATASET_REQUIRED_COLUMNS` | ordered `tuple` | 9 | `scripts/datasets/contracts.py:23-33` |

### 5.1 Presence matrix

| Name | B | F | E | D |
|---|:---:|:---:|:---:|:---:|
| `age` | Y | Y | Y | - |
| `avg_drug_duration` | Y | - | - | - |
| `cyp_high_risk_pairs` | Y | Y | - | - |
| `cyp_max_enzyme_risk` | Y | Y | - | - |
| `cyp_risk_score` | Y | Y | - | - |
| `ddi_contraindicated` | Y | Y | Y | Y |
| `ddi_major` | Y | Y | Y | Y |
| `ddi_minor` | Y | Y | Y | Y |
| `ddi_moderate` | Y | Y | Y | Y |
| `drug_count` | Y | Y | Y | Y |
| `drug_count_7d` | Y | Y | Y | Y |
| `dup_atc3` | Y | Y | Y | - |
| `dup_atc4` | Y | Y | Y | - |
| `dup_atc5` | Y | Y | Y | - |
| `dup_efmdc` | Y | Y | - | - |
| `dup_same_ingredient` | Y | Y | Y | - |
| `has_hepatic_risk_drug` | Y | Y | - | - |
| `has_high_risk_drug` | Y | Y | - | - |
| `has_renal_risk_drug` | Y | Y | - | - |
| `institution_count` | Y | Y | Y | Y |
| `long_term_drug_count` | Y | - | - | - |
| `patient_id` | - | - | - | Y |
| `qt_risk_count` | Y | Y | Y | - |
| `risk_level` | - | - | - | Y |
| `sex_m` | Y | Y | - | - |
| `triple_whammy` | Y | Y | Y | - |

### 5.2 Exact set differences

| Comparison | Left-only names | Reading |
|---|---|---|
| B \ F | `avg_drug_duration`, `long_term_drug_count` | Builder capability exceeds the UI default list by two duration features. |
| F \ B | none | **`FEATURE_COLS` is a strict subset of `_BUILDER_KNOWN_COLS`** (22 of 24). |
| B \ E | `avg_drug_duration`, `cyp_high_risk_pairs`, `cyp_max_enzyme_risk`, `cyp_risk_score`, `dup_efmdc`, `has_hepatic_risk_drug`, `has_high_risk_drug`, `has_renal_risk_drug`, `long_term_drug_count`, `sex_m` | B covers online/resource-derived and duration/sex features E does not enumerate. |
| E \ B | none | E is a subset of B. |
| F \ E | `cyp_high_risk_pairs`, `cyp_max_enzyme_risk`, `cyp_risk_score`, `dup_efmdc`, `has_hepatic_risk_drug`, `has_high_risk_drug`, `has_renal_risk_drug`, `sex_m` | UI defaults include 8 names outside E. |
| E \ F | none | E is a subset of F (14 of 22). |
| D \ E | `patient_id`, `risk_level` | D adds an identifier and a label, which are not numeric model features. |
| E \ D | `age`, `dup_atc3`, `dup_atc4`, `dup_atc5`, `dup_same_ingredient`, `qt_risk_count`, `triple_whammy` | E holds 7 numeric names outside the minimum dataset presence contract. |

D and E therefore share exactly 7 numeric features; neither contains the other.

### 5.3 Semantics behind the notable differences

- **`dup_efmdc`** is in B and F, absent from E. Serving documents it as the EDI to EFMDC (HIRA classification) bridge, with `0.0` only as a degraded fallback when the bridge is missing.
- **`sex_m`** is in B and F, absent from E. `FeatureEngineer.run()` separately creates a column named **`sex_male`**, not `sex_m` (`scripts/features/feature_engineer.py:124-127`), so the E-side naming divergence is real. The input-domain difference between B and F is described in Section 3.2.
- **`avg_drug_duration`, `long_term_drug_count`** are B-only among these four, computed online from request duration values.
- **`patient_id`, `risk_level`** are D-only because D is a minimum dataset presence contract including metadata and a label. `validate_required_columns()` converts actual columns to a **set** and checks presence only; it does **not** enforce physical order (`scripts/datasets/contracts.py:92-97`).

### 5.4 Physical column order

Equal logical feature sets do **not** prove equal physical order.

| Surface | Order authority | Risk |
|---|---|---|
| Online tabular / hierarchical | Bundle `feature_names` / `feature_cols`. Dict-by-name alignment, then positional conversion. | Safe only while bundle metadata carries correct ordered names. The frozenset is not an order contract. |
| UI training | `FEATURE_COLS` declared order, or the user's ordered subset. | Divergence from bundle order is possible by design. |
| Feature-engineering Parquet | `FeatureEngineer.run()` starts from the physical ETL frame, appends CYP and temporal columns via left merges, then label/sex transforms, normalizer, and selector before writing `ml_features_{partition}.parquet` (`scripts/features/feature_engineer.py:80-144`). | Physical order is pipeline/merge/transform dependent, **not** defined by `ETL_NUMERIC_COLS` alone. |
| Dataset required tuple | Declared tuple order exists but is not enforced. | Presence-only validation. |

Any consumer that bypasses name selection and reads positionally is exposed to train/serve skew.

### 5.5 Policy

- Do **not** merge `FEATURE_COLS` into `_BUILDER_KNOWN_COLS`.
- Do **not** demand zero diff between B, F, E, and D.
- Phase 2B records these differences through the read-only `ProfileDiffReporter` adapter, which reports differences and never removes them.

### 5.6 Historical context: commit `d201743`

`F \ B` is empty today partly because of one alignment commit. `d201743` (2026-04-29) renamed the serving builder's `sex_male` to `sex_m` to match `FEATURE_COLS`, added `dup_atc3` and the three `has_*_risk_drug` flags, added a `0.0` fallback branch for `cyp_max_enzyme_risk`, and updated `_BUILDER_KNOWN_COLS`.

Three limits on what that commit supports:

1. It aligned **names**. It is not evidence that the two producers compute equal **values**, and it says nothing about physical order.
2. Its scope was `serving/predictor.py` against `hana_app/core/ml_runner.py`. It did not touch `scripts/features/feature_engineer.py`, which still creates `sex_male`.
3. Its `dup_efmdc = 0.0` fixed-value constraint has been superseded; `dup_efmdc` is now a bridge output with `0.0` as a degraded fallback only.

Where that commit message and current source disagree, **source at `3d8d64e` governs**. The commit is cited in current source as the precedent motivating the DDI semantic-version guards.

---

## 6. Serving dependency graph

Fourteen direct local-module edges from `serving.predictor` (`docs/superpowers/reports/contract-baseline/phase0b-dependency-graph.md`):

```text
serving.predictor
├── serving.schemas                  [eager: API/domain types + INTERVENTION_MAP]
├── serving.dl_predictor             [eager: DLModel]
├── serving.hana_history             [eager: HANAHistoryProvider]
├── rules.risk_drug_constants        [eager: HIGH_RISK_KEYWORDS, RENAL_RISK_KEYWORDS,
│                                     HEPATIC_RISK_KEYWORDS, and the three ATC prefix sets]
├── rules.safety_net                 [lazy/guarded: SafetyNet]
├── rules.duplicate_detector         [lazy/guarded: DuplicateDetector]
├── scripts.etl.code_standardizer    [lazy/guarded: CodeStandardizer]
├── scripts.features.cyp_features    [lazy/guarded: CYPFeatureExtractor]
├── scripts.etl.models               [lazy: PrescriptionRecord, PatientFeatures]
├── scripts.etl.overlap_calculator   [lazy: calculate_overlaps_for_patient,
│                                     get_concurrent_drug_count]
├── scripts.etl.prescription_aggregator
│                                    [lazy: count_ddi_severities, ddi_pair_severities,
│                                     _fill_dup_features, detect_triple_whammy,
│                                     detect_risk_drug, DDI_FEATURE_SEMANTICS_VERSION,
│                                     FEATURE_SEMANTICS_VERSION]
├── scripts.etl.clinical_rules       [lazy: collect_red_triggers,
│                                     collect_severe_immediate_triggers]
├── scripts.train.gat_trainer        [lazy/optional: GATTrainer, for an
│                                     EnsembleTrainer3Way bundle]
└── hana_app.core.hierarchical_runner
                                     [lazy: STAGE2_LABELS, predict_risk, ACTION_BY_LABEL]
```

Notes that correct commonly repeated but stale claims:

- **No circular dependency involves `serving.predictor`.** No direct dependency has a static path back to it, in either the import-time graph or the all-lexical-scope graph (115 modules parsed).
- The import-time closure reachable from `serving.predictor` is acyclic. **The graph as a whole is not a DAG**, however: the all-scope graph contains a lazy cycle in the optional GAT training branch, `base_graph_trainer -> trainer -> (method-local) gat_trainer -> base_graph_trainer`. It is not an import-time cycle, but it is real coupling and must not be described away.
- `serving.dl_predictor -> serving.hana_history` is a **sibling** edge, not a cycle.
- There is **no direct `DrugMaster` import**. `RequestFeatureBuilder._drug_master()` reaches the object through `CodeStandardizer.drug_master`.
- `GATTrainer` is the imported symbol. `EnsembleTrainer3Way` is the bundle/trainer-class **condition** that activates the branch, not an alias.
- `serving/dl_predictor.py:227` loads Torch with `importlib.import_module("torch")`, keeping Torch off the initial import path and moving DL failures to first runtime use.

The `serving -> hana_app.core.hierarchical_runner` edge (Section 2.11) is recorded and **NOT removed in Phase 2**.

---

## 7. Phase 3 future-work policy (out of scope)

Phase 3 and everything after it are **outside the scope of this plan**. This section records policy for reference only. Nothing here authorizes work.

**Phase 3 items:**

1. **predictor/domain extraction.** Split `serving/predictor.py` responsibilities. Remove the runtime `hana_app.core` dependency. Move the pure domain policy (`predict_risk`, `ACTION_BY_LABEL`, `STAGE2_LABELS`) to a neutral shared module, **preserving compatibility imports** as needed.
2. **Wide engine integration.** Merging `tabular_binary` and `hierarchical` into a single inference engine. **Fable 5 and Codex both returned NO-GO. Not pursued.**
3. `scripts/ops/` reorganization: separating reusable library code from one-shot command scripts.
4. Documentation alignment: `data_pipeline_architecture.md` names Spark, HDFS, S3, Feast, Great Expectations, and Grafana, while production is a single closed-network Windows Python 3.12 machine.
5. `FEATURE_SCHEMA_LENIENT` environment variable removal, as a separate PR **after** Phase 3.

Any label-space, train/serve schema, or HANA-query impact in Phase 3 requires the repository's critical cross-family review and the serving parity gates.

**What Phase 2 does NOT do.** It does not copy or move `predict_risk`, `ACTION_BY_LABEL`, or `STAGE2_LABELS`. It does not split `predictor.py`. It does not merge feature lists. It does not change labels, semantic versions, thresholds, or artifact formats. It does not migrate artifacts or retrain. It does not remove `sys.path` mutations or touch the lazy GAT cycle.

The `rulefeat.v1` comment-versus-implementation discrepancy (Section 2.5) and the `reload_dl()` return-value weakness (Section 4.5) are **recorded risks for future design review, not authorization to change them here**.

---

## 8. Source-backed corrections that override stale upstream statements

Two upstream documents describe these contracts and are both partly stale:

- **The plan draft:** `docs/superpowers/plans/2026-07-12-phase2-contract-spec-and-characterization.md`, Wave 1 Task 1 Step 1, the fenced draft of this document (plan lines 62 to 214).
- **The historical design source:** `docs/superpowers/specs/2026-07-12-opencode-lo-contract-design.md`, Sections 6.3.1 to 6.3.4 and 6.4.

Neither upstream source is being edited here. **This section is the override record:** where either upstream text disagrees with source at `3d8d64e`, the statements below govern, and this document, not the upstream text, is what Phase 2B characterizes. This profile contract together with current `AGENTS.md` and `CLAUDE.md` governs the recorded contract and safety scope; references to the old design are historical comparison only. Phase 0A/0B baseline reports corroborate each row.

Read the "Stale statement" column as: what the cited upstream text says or implies. Read "Source at `3d8d64e`" as: what the code does.

| # | Stale statement | Where it appears upstream | Source at `3d8d64e` | This spec |
|---|---|---|---|---|
| 1 | ui_experimental safety guards are `page_guards.py` and `memory_guard.py`. | Plan draft, section 3 "Validation"; Spec 6.3.3, row 안전장치 | **Page 3 does not import `page_guards.py`.** Only `memory_guard` is active. | 3.5 |
| 2 | (Both silent on UI dtype/default semantics.) | Plan draft, section 3; Spec 6.3.3 | UI missing `age` is **`-1`**, not the serving `0.0`; UI `sex_m` encodes from raw HANA `"1"`/`"2"`, not `M`/`F`; `sex_type` is metadata, not a model input. | 3.2 |
| 3 | `FEATURE_SEMANTICS_VERSION` is enforced like the DDI guard: bundle metadata mismatch rejects the load. | Plan draft, section 2 "Semantic versions"; Spec 6.3.2, row 시맨틱 버전; Spec 6.4 (blanket "가드가 현재 버전과 불일치/누락 번들을 거부한다") | `rulefeat.v1` has **no load rejection**. It is a prediction-time feature-path gate only. The `scripts/etl/prescription_aggregator.py:211-216` comment claiming a reload guard is contradicted by the implementation. Spec 6.4's blanket claim holds for `ddi.v2` and **not** for `rulefeat.v1`. | 2.5 |
| 4 | `reload_dl` performs eager validation, implying a rollback contract equal to the other two hot-swaps. | Plan draft, section 4 "Reload contract"; Spec 6.3.4, row 핫스왑 | `reload_dl()` **never branches on `load()`'s return**; it always swaps and returns `True`. Rollback depends entirely on `DLModel.load()` raising. | 4.5 |
| 5 | Hierarchical validation chain ordered as schema, labels, DDI, hash, encoder. | Plan draft, section 2 "Validation chain"; Spec 6.3.2, row 검증 (order unstated, listing implies schema first) | Actual order is files, meta, **labels**, **DDI**, non-empty `feature_cols`, schema, hash, joblib load, encoder. Labels and DDI are checked before joblib deserialization (fail fast). | 2.6 |
| 6 | (Both silent on missing-hash behavior.) | Plan draft, sections 1 and 2; Spec 6.3.2, 6.3.4 | Missing `stage1_sha256`/`stage2_sha256` in metadata is **fail-closed** (rejected), not skipped. Same for the tabular `.pkl.sha256` sidecar. | 1.7, 2.6 |
| 7 | (Both silent on degraded Stage 1.) | Plan draft, section 2; Spec 6.3.2 | Stage 1 degrades to a constant non-Red dummy below 10 Red or 10 non-Red rows, with fixed `tau_red=1.0`, `tau_review=0.5`. Such bundles never emit model Red, and serving does **not** reject them on `stage1_trained=False`. | 2.4, 2.9 |
| 8 | The DDI semantic guard applies to every tabular bundle. | Plan draft, section 1 "Semantic version"; Spec 6.3.1, row 시맨틱 버전 | It fires **only when some loaded feature name starts with `ddi_`**. | 1.5 |
| 9 | `_dispatch_result` returns `stage2_probs` as passed. | Plan draft, section 2 "Thresholds" | It returns a **dict** `{label: prob}` when Stage 2 runs, and `None` on confirmed Red. | 2.4 |
| 10 | (Draft graph.) Implied direct `DrugMaster` edge; omitted `rules.risk_drug_constants`; asserted "No circular dependencies" globally. | Plan draft, section 7 | No direct `DrugMaster` import. `rules.risk_drug_constants` is an eager edge. The all-scope graph holds a lazy GAT cycle, though nothing cycles back to `serving.predictor`, so the global no-cycle claim is wrong while the `serving.predictor` claim is right. | 6 |
| 11 | `ETL_NUMERIC_COLS` vs `ML_DATASET_REQUIRED_COLUMNS` overlap stated loosely. | Plan draft, section 5 | D \ E is exactly `patient_id`, `risk_level`; E \ D is exactly 7 names; they share exactly 7 numeric features. | 5.2 |
| 12 | One fixed tabular state-dict key list applies to every bundle, always including `model` and never including `params`/`feature_importances`. | Plan draft, section 1 "Artifact format"; Spec 6.3.1 | Keys are **per trainer**. Single trainers write `model`, `params`, `feature_importances`, `best_threshold`, `trainer_class`. **Ensemble top-level payloads omit `model` and `params`** and carry `weights` plus sidecars. The feature-contract keys (`artifact_version`, `feature_names`, `scaler_path`, `selector_path`, `ddi_feature_semantics_version`) reach the payload only through `pipeline.py`'s `_extra_meta` stamping. `partition` is read by the loader but written by no current trainer. | 1.7 |
| 13 | `stage_meta.json` holds the six or eight keys serving happens to read. | Plan draft, section 2 "Artifact format"; Spec 6.3.2 | Training writes **14 keys**. Seven of them are provenance/review metadata that serving never reads. | 2.9 |
| 14 | `stage2_yellow.joblib` is a dict of `model`, `encoder`, `classes_present`. | Plan draft, section 2 "Artifact format" | It has **four** keys; `stage2_classes_global` is also written. Serving reads three of the four. | 2.9 |

Corroboration: rows 12, 13, and 14 restate the Phase 0A bundle metadata record (Sections 1.1, 1.2, and 3), which already flagged the plan's fixed-key-list implication as stale. Row 10 restates the Phase 0B dependency graph.

---

## 9. Acceptance check for this document

The Wave 1 acceptance criteria live in the plan (`docs/superpowers/plans/2026-07-12-phase2-contract-spec-and-characterization.md`, "Acceptance Criteria", Wave 1) and in the Spec (Section 7.5, 인수 기준). They are reproduced here **split by what this document can and cannot establish**.

### 9.1 Established by this document

- [x] Covers all four profiles: `tabular_binary` (1), `hierarchical` (2), `ui_experimental` (3), `dl_history` (4).
- [x] Feature names, physical order, dtype/default semantics, labels, thresholds, semantic versions, and artifact/reload contracts recorded per profile.
- [x] Cross-profile differences recorded without flattening; no feature list merged (5).
- [x] `serving -> hana_app.core.hierarchical_runner` dependency recorded and not removed (2.11, 6).
- [x] Phase 3 future-work policy recorded as a section of this document, not a separate file (7).
- [x] No em dashes or en dashes.
- [x] No frozen holdout read; no protected artifact opened for this document.
- [x] No production code modified **by this document**. It adds one Markdown file and touches nothing under `serving/`, `hana_app/`, `scripts/`, `config/`, `rules/`, or `dags/`. Independently checkable: `git status --short -- serving hana_app scripts config rules dags` reports only the two untracked Phase 1 helper scripts (`scripts/ops/check_lenient_sunset.py`, `scripts/ops/check_py312_drift.py`), which pre-date this document and were not created by it.

### 9.2 Not established by this document: pre-existing tests

The Wave 1 criterion is "**No production code modified; all pre-existing tests pass**" (Spec 7.5: 기존 테스트 전체 통과). The first half is settled above. **The second half is open, and this document does not check it off.** No test run and no pytest collection was performed while writing this spec.

Two prior findings make an honest claim impossible right now, both from `docs/superpowers/reports/contract-baseline/phase0b-baseline-report.md`:

1. **No durable test baseline exists** (Risk 2 in that report). An earlier interactive pytest run left no durable node-ID or outcome evidence, so nothing here can assert which tests passed before this change.
2. **The worktree `.venv` cannot reproduce a run** (Risk 3 in that report). `pytest` is installed, while `pydantic` package metadata is `NOT_INSTALLED`, so the serving tests cannot even import. A reproducible baseline needs an environment with the full serving and training dependencies.

Because this document is Markdown only, it has no runtime surface and cannot itself change a test outcome: a passing suite before it is a passing suite after it. That is an argument about impact, **not evidence of a passing suite**, and it must not be recorded as one.

**What closes this criterion:** capture a durable node-ID and outcome baseline in a full-dependency Python 3.12 environment (`python -m pytest tests/ -q`), as Phase 1 was already directed to do, and compare the post-change run against it. Until that exists, treat the "all pre-existing tests pass" half of Wave 1 as unverified.
