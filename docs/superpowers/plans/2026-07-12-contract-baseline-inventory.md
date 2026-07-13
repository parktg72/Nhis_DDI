# Contract Baseline Inventory (Phase 0A/0B) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Map all policy/runtime contract state across four operational profiles and produce a reproducible read-only contract baseline report, with zero code changes, stopping before any tooling implementation.

**Architecture:** Phase 0A inventories feature lists, label spaces, thresholds, semantic versions, validation rules, dtypes, defaults, resource semantics, reload/rollback behavior, physical column order, pickle/joblib module-path risks, and FEATURE_SCHEMA_LENIENT runtime/deadline state for each profile (tabular_binary, hierarchical, ui_experimental, dl_history). Phase 0B consumes the 0A mapping to produce a reproducible baseline report using only AST/source/grep analysis (no runtime imports of serving or hana_app modules), plus a dependency graph of serving/predictor.py. No new scripts are written. No code is changed. No protected artifacts are loaded or unpickled.

**Tech Stack:** Python 3.12 (.venv), grep/sed for text search, python ast module for source-level extraction (no runtime imports), pickletools for non-executing bundle metadata (LO approval required), Markdown for report output.

**Authority sources:**
- Spec: `docs/superpowers/specs/2026-07-12-opencode-lo-contract-design.md` (sections 7.2, 7.3)
- `AGENTS.md` hard gates and protected paths
- `CLAUDE.md` configured paths and environment
- OpenCode is final LO. All worker results return evidence to OpenCode for verification.

**Freeze-safe declaration:** All tasks are read-only. No Nov->Dec holdout tuning, no feature/label/version changes, no artifact migration, no retraining, no Gate 5A/5B activation, no 2025-01 data acquisition. `RESEARCH_TRACK_FROZEN`.

---

## File Structure

All output files are new documentation/report artifacts. No existing source files are modified.

| File | Phase | Responsibility |
|---|---|---|
| `docs/superpowers/reports/contract-baseline/phase0a-profile-contract-map.md` | 0A | Profile-by-profile contract state mapping |
| `docs/superpowers/reports/contract-baseline/phase0a-feature-dispersion-table.md` | 0A | Cross-source feature list dispersion table |
| `docs/superpowers/reports/contract-baseline/phase0a-bundle-metadata-record.md` | 0A | Bundle metadata and pickle/joblib module-path risks |
| `docs/superpowers/reports/contract-baseline/phase0b-dependency-graph.md` | 0B | serving/predictor.py dependency graph with circular dependency check |
| `docs/superpowers/reports/contract-baseline/phase0b-baseline-report.md` | 0B | Reproducible contract baseline report |
| `docs/superpowers/reports/contract-baseline/README.md` | 0B | Index and reproduction instructions |

Protected path safety: none of these files touch `packages_win/py312/`, `mlruns/`, generated `.parquet`, or `out/`.

---

## Pre-flight Checks

- [ ] **Pre-flight 1: Verify Python 3.12 runtime**

Run:
```bash
source .venv/bin/activate && python --version
```
Expected: `Python 3.12.x`. If not 3.12: STOP (BLOCK trigger, AGENTS.md Python 3.12 runtime lock).

- [ ] **Pre-flight 2: Record git working tree state**

Run:
```bash
git status --short
```
Record the output. Existing unrelated changes must NOT be included in future commits. All commits from this plan stage ONLY files under `docs/superpowers/reports/contract-baseline/`. If protected paths appear in the working tree: note them but do NOT touch them.

- [ ] **Pre-flight 3: Create report output directory**

Run:
```bash
mkdir -p docs/superpowers/reports/contract-baseline
```

---

## Phase 0A: Policy/Runtime State Mapping

### Task 1: Extract All Feature Lists and Label Constants via AST

This task extracts every feature list, label tuple, and version constant from source code using AST parsing only. No runtime imports of serving, hana_app, scripts, or rules modules. This ensures the baseline works even when optional dependencies (torch, sklearn, etc.) are absent.

**Files:**
- Create: `docs/superpowers/reports/contract-baseline/phase0a-profile-contract-map.md`
- Create: `docs/superpowers/reports/contract-baseline/phase0a-feature-dispersion-table.md`
- Read: `serving/predictor.py`, `hana_app/core/ml_runner.py`, `hana_app/core/hierarchical_runner.py`, `scripts/features/feature_engineer.py`, `scripts/datasets/contracts.py`, `scripts/etl/prescription_aggregator.py`, `serving/schemas.py`

- [ ] **Step 1: Extract all constants via single AST script**

Run:
```bash
source .venv/bin/activate
python3 << 'PYEOF'
import ast, json, os
from datetime import date

results = {}

def extract_assigns(filepath, names):
    """Extract top-level assignments by name from a Python file via AST."""
    with open(filepath) as f:
        tree = ast.parse(f.read())
    out = {}
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in names:
                    try:
                        out[target.id] = ast.literal_eval(node.value)
                    except Exception:
                        out[target.id] = f"<non-literal: {ast.dump(node.value)[:100]}>"
    return out

def extract_imports(filepath):
    """Extract all import module paths from a Python file via AST."""
    with open(filepath) as f:
        tree = ast.parse(f.read())
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                imports.append((node.lineno, a.name, "top-level"))
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            imports.append((node.lineno, mod, f"from line {node.lineno}"))
    return imports

# 1. serving/predictor.py constants
pred = extract_assigns("serving/predictor.py", {
    "_BUILDER_KNOWN_COLS", "_INTENTIONAL_FEATURE_ALLOWLIST",
    "_FEATURE_SCHEMA_LENIENT_SUNSET_DEFAULT",
})
results["predictor"] = pred

# 2. hana_app/core/ml_runner.py constants
ml = extract_assigns("hana_app/core/ml_runner.py", {
    "FEATURE_COLS", "RISK_LABEL_MAP", "_SAFE_MISCLS_FEATURES",
})
results["ml_runner"] = ml

# 3. hana_app/core/hierarchical_runner.py constants
hier = extract_assigns("hana_app/core/hierarchical_runner.py", {
    "YELLOW_SUBTYPE_LABELS", "STAGE2_LABELS", "ACTION_BY_LABEL",
})
results["hierarchical_runner"] = hier

# 4. scripts/features/feature_engineer.py constants
fe = extract_assigns("scripts/features/feature_engineer.py", {
    "ETL_NUMERIC_COLS", "LABEL_COL", "BINARY_LABEL_COL",
    "ML_FEATURE_OUTPUT",
})
results["feature_engineer"] = fe

# 5. scripts/datasets/contracts.py constants
dc = extract_assigns("scripts/datasets/contracts.py", {
    "ML_DATASET_REQUIRED_COLUMNS", "DL_DATASET_REQUIRED_COLUMNS",
    "DL_BUNDLE_REQUIRED_FILES", "DL_MANIFEST_FILE", "HASH_ALG_SHA256",
    "LOOKBACK_DAYS_DEFAULT", "LOOKBACK_DAYS_MIN", "LOOKBACK_DAYS_MAX",
})
results["contracts"] = dc

# 6. scripts/etl/prescription_aggregator.py constants
pa = extract_assigns("scripts/etl/prescription_aggregator.py", {
    "DDI_FEATURE_SEMANTICS_VERSION", "FEATURE_SEMANTICS_VERSION",
})
results["prescription_aggregator"] = pa

# 7. serving/schemas.py INTERVENTION_MAP
schemas = extract_assigns("serving/schemas.py", {"INTERVENTION_MAP"})
results["schemas"] = schemas

# 8. FEATURE_SCHEMA_LENIENT runtime state
env_lenient = os.environ.get("FEATURE_SCHEMA_LENIENT", "")
env_sunset = os.environ.get("FEATURE_SCHEMA_LENIENT_SUNSET_DATE", "")
sunset_default = pred.get("_FEATURE_SCHEMA_LENIENT_SUNSET_DEFAULT", "???")
today = date.today()
results["lenient_state"] = {
    "env_FEATURE_SCHEMA_LENIENT": env_lenient,
    "env_FEATURE_SCHEMA_LENIENT_SUNSET_DATE": env_sunset,
    "code_default_sunset": str(sunset_default),
    "today": str(today),
    "today_before_sunset": str(today < sunset_default) if isinstance(sunset_default, date) else "???",
}

print(json.dumps(results, indent=2, default=str, ensure_ascii=False))
PYEOF
```
Expected: JSON output with all constants extracted. Record this output; it feeds every subsequent task. The executor should save this output to a temporary variable or file for reference while writing the report sections.

- [ ] **Step 2: Write the profile contract map**

Write `docs/superpowers/reports/contract-baseline/phase0a-profile-contract-map.md` with the following sections. For each section, run the indicated command and insert its output in a fenced block, plus insert the relevant values from the Task 1 Step 1 AST JSON output.

**Section 1: tabular_binary Profile**
- 1.1 Feature Source: `_BUILDER_KNOWN_COLS` (`serving/predictor.py:45-53`, frozenset). Insert sorted list, `_INTENTIONAL_FEATURE_ALLOWLIST`, and `_FEATURE_ALLOWED` from AST output.
- 1.2 Feature dtype/defaults: Run `sed -n '1029,1074p' serving/predictor.py` and insert. Note each `feat[...] = ...` line: name, dtype (all float or bool-as-float), default expression, source line.
- 1.3 Threshold: `MLModel._threshold` default `0.5` (`serving/predictor.py:310`), loaded from `state.get("best_threshold", 0.5)` (line 383). `classify()` uses threshold for Red/Yellow/Green/Normal.
- 1.4 Semantic Version Guard: Insert `DDI_FEATURE_SEMANTICS_VERSION` from AST output. Guard: if any `feature_names` starts with `ddi_`, bundle meta must match, else load rejected (`serving/predictor.py:393-406`).
- 1.5 Validation: `_validate_feature_schema()` (`serving/predictor.py:93-139`) checks `feature_names` subset of `_FEATURE_ALLOWED`. Strict by default. Lenient escape: `FEATURE_SCHEMA_LENIENT=1` env + sunset check. Insert sunset default from AST output.
- 1.6 Reload/Rollback: Run `sed -n '1286,1294p' serving/predictor.py` and insert. Thread-safe via `_ml_lock`. On failure: `ok=False`, old model retained. On success: atomic swap.
- 1.7 Production Path: `RequestFeatureBuilder.build()` -> `MLModel.predict_proba()` -> `MLModel.classify()`

**Section 2: hierarchical Profile**
- 2.1 Feature Source: Bundle meta `stage_meta.json` `feature_cols`. Runtime validation: non-empty after load (`serving/predictor.py:1301-1303`).
- 2.2 Label Space: Insert `YELLOW_SUBTYPE_LABELS` and `STAGE2_LABELS` from AST output. Guard: bundle meta `stage2_labels` must exactly match current `STAGE2_LABELS`, else load rejected (`serving/predictor.py:674-685`).
- 2.3 ACTION_BY_LABEL: Insert dict from AST output.
- 2.4 INTERVENTION_MAP: Insert dict from AST output (from `serving/schemas.py`).
- 2.5 Thresholds: From `stage_meta.json` `thresholds`: `tau_red`, `tau_review`. 2-stage branching: `p_red >= tau_red` -> Red, `tau_review <= p_red < tau_red` -> Stage 2 + red_suspect, `p_red < tau_review` -> Stage 2 only.
- 2.6 Semantic Versions: Insert both version constants from AST output. Both must match bundle meta. Runtime: `_rf_active` gates rule feature path (`serving/predictor.py:1390-1391`).
- 2.7 Reload/Rollback: Run `sed -n '1296,1315p' serving/predictor.py` and insert. Thread-safe via `_hier_lock`. On success: validates `feature_cols` non-empty, runs `_validate_feature_schema()`, atomic swap. On failure: `ok=False`, old retained.
- 2.8 Serving -> hana_app Dependency: `predict_risk_single` imports `predict_risk` (`serving/predictor.py:788`), `predict` imports `ACTION_BY_LABEL` (`serving/predictor.py:1470`), label guard imports `STAGE2_LABELS` (`serving/predictor.py:674-676`).
- 2.9 Backstop: `red_triggers()` (contraindications, `RED_CONTRAINDICATED`), `rule_floor()` (Y_DDI_MAJOR/Y_TRIPLE minimum subtype guarantee).
- 2.10 Production Path: `RequestFeatureBuilder.build()` -> `HierarchicalPredictor.predict_risk_single()` -> `predict_risk()` (from hana_app)

**Section 3: ui_experimental Profile**
- 3.1 Feature Source: `FEATURE_COLS` (`hana_app/core/ml_runner.py:50-73`, list, ordered). Insert ordered list from AST output.
- 3.2 Label: Insert `RISK_LABEL_MAP` from AST output.
- 3.3 Training Path: `ml_runner.py` -> `aggregate_patient_features()` -> `FeatureEngineer` -> trainer. UI internal stratified sample, cross-validation, metrics display.
- 3.4 Safety Guards: `page_guards.py`, `memory_guard.py` (memory/time limits).
- 3.5 Operational Separation: Not directly connected to operational serving bundles. Separate path.
- 3.6 Risk: FEATURE_COLS vs _BUILDER_KNOWN_COLS mismatch. Compare the two sets from AST output. Record which features are in one but not the other. Design-intentional, NOT a zero-diff target. Phase 2A will specify; Phase 2B will record in characterization tests.
- 3.7 _SAFE_MISCLS_FEATURES: Insert list from AST output. Subset of FEATURE_COLS (excludes `age`, `sex_m`).

**Section 4: dl_history Profile**
- 4.1 Bundle Required Files: Insert `DL_BUNDLE_REQUIRED_FILES` from AST output.
- 4.2 Dataset Required Columns: Insert `DL_DATASET_REQUIRED_COLUMNS` from AST output.
- 4.3 Manifest: Insert `DL_MANIFEST_FILE` and `HASH_ALG_SHA256` from AST output.
- 4.4 Lookback: Insert `LOOKBACK_DAYS_DEFAULT`, `LOOKBACK_DAYS_MIN`, `LOOKBACK_DAYS_MAX` from AST output. `validate_lookback_consistency()`: runtime must match artifact, else `LookbackMismatchError`.
- 4.5 Encoding Strategy: `multi_hot` only (supported), `count` removed (dead infra).
- 4.6 Graph Architecture: `gat`, `gcn`.
- 4.7 Reload/Rollback: Run `sed -n '1317,1328p' serving/predictor.py` and insert. Thread-safe via `_dl_lock`. Risk: `reload_dl` does not check `load()` return value before swap (unlike `reload_model`/`reload_hierarchical`). Always returns `True`.
- 4.8 Production Path: `HANAHistoryProvider.fetch_patient_history()` -> `DLModel.predict()`
- 4.9 Operational Impact: Currently not reflected in final `risk_level` determination. Auxiliary result only.

**Section 5: Semantic Version Constants and FEATURE_SCHEMA_LENIENT State**
- 5.1 DDI_FEATURE_SEMANTICS_VERSION: Insert value from AST output. Source: `scripts/etl/prescription_aggregator.py:209`. Meaning: DDI count semantic version. v2 = WK->DrugMaster->DB-code overlap path. Guard: bundle meta must match, else load rejected (tabular_binary: `serving/predictor.py:393-406` if `ddi_*` in features; hierarchical: `serving/predictor.py:689-700`).
- 5.2 FEATURE_SEMANTICS_VERSION: Insert value from AST output. Source: `scripts/etl/prescription_aggregator.py:216`. Meaning: Rule-derived feature semantic version. v1 = component keyword path. Guard: runtime `_rf_active` gates rule feature path (`serving/predictor.py:1390-1391`).
- 5.3 FEATURE_SCHEMA_LENIENT Runtime/Deadline State: Insert the `lenient_state` object from AST JSON output in a fenced block. Record: code default sunset, today, today before sunset. Risk: if deployed model uses unknown columns and lenient is blocked after sunset, server startup will fail.
- 5.4 Lenient Logic: `_is_feature_schema_lenient_allowed(today=None)` (`serving/predictor.py:68-90`): if `FEATURE_SCHEMA_LENIENT_SUNSET_DATE` env set and valid, use that date; if invalid, return False; if not set, use code default; returns `today < sunset`. `_validate_feature_schema()` (`serving/predictor.py:93-139`): strict rejects unknown columns; lenient (if allowed) warns + 0.0 fallback; lenient env set but sunset passed: strict enforced with error log.

**Section 6: Required Resource Semantics**
- Run `grep -n "ddi_matrix\|cyp_extractor\|code_standardizer\|CodeStandardizer\|CYPFeatureExtractor\|SafetyNet\|DuplicateDetector\|DrugMaster\|_ddi_matrix\|_cyp\b\|_std\b\|_safety_net\|_dup_detector\|_drug_master" serving/predictor.py | head -40` and insert output.
- 6.1 DDI Matrix: `pd.DataFrame` (optional). Fallback: if None, DDI alert enrichment skipped. DDI count features still computed via `count_ddi_severities`.
- 6.2 CYP Extractor: `CYPFeatureExtractor` (optional). Fallback: if None or no ATC codes, CYP features default to 0.0 (`serving/predictor.py:1081-1083`).
- 6.3 CodeStandardizer: (optional). Fallback: if None, EDI->ATC enrichment skipped.
- 6.4 SafetyNet: (optional). Fallback: `ImportError` -> `RiskLevel.NORMAL` (`serving/predictor.py:252-255`). Runtime error with instance: propagated (`serving/predictor.py:257-260`).
- 6.5 DuplicateDetector: (optional). Fallback: `ImportError` -> `(0, [])` (`serving/predictor.py:288-291`). Runtime error with instance: propagated (`serving/predictor.py:293-296`).
- 6.6 DrugMaster: lazy load via `RequestFeatureBuilder._drug_master()` (only when `rule_features_active=True`). Fallback: if None and `rule_features_active=False`, ATC/name-based fallback path used.

**Section 7: Physical DataFrame/Parquet Column Order**
- 7.1 Serving Feature Alignment: Run `sed -n '1118,1130p' serving/predictor.py` and insert. Serving aligns to `feature_names` from model bundle (dict-based, order-safe by name). Missing features default to `0.0`. `_BUILDER_KNOWN_COLS` is a `frozenset` (unordered), does NOT define physical order.
- 7.2 Training Feature Parquet: Output path `data/features/ml_features_{partition}.parquet` (from `ML_FEATURE_OUTPUT` in AST output). Column order: `ETL_NUMERIC_COLS` (from AST output) + CYP features + temporal features, merged via `df.merge()`. Physical order depends on merge order in `FeatureEngineer.run()`.
- 7.3 Skew Risk: Logical feature name set equality does NOT guarantee physical order equality. Serving uses dict-based alignment (order-safe). Training Parquet column order determined by ETL_NUMERIC_COLS + merge order. Risk: any code that reads Parquet columns by position (df.values without column names) is vulnerable. Phase 2B will record exact physical column order in characterization tests.

- [ ] **Step 3: Write the feature dispersion table**

Write to `docs/superpowers/reports/contract-baseline/phase0a-feature-dispersion-table.md`:

```markdown
# Phase 0A: Cross-Source Feature Dispersion Table

**Purpose:** Record the current state of feature list dispersion across four sources. This is NOT a zero-diff target. Differences are design-intentional and must not be flattened.

**Sources:**
1. `_BUILDER_KNOWN_COLS` - `serving/predictor.py:45-53` (serving baseline, frozenset)
2. `FEATURE_COLS` - `hana_app/core/ml_runner.py:50-73` (Page 3 UI training, list)
3. `ETL_NUMERIC_COLS` - `scripts/features/feature_engineer.py:37-43` (Airflow feature engineering, list)
4. `ML_DATASET_REQUIRED_COLUMNS` - `scripts/datasets/contracts.py:23-33` (dataset contract, tuple)

## Cross-Source Presence Matrix

Run the following command and insert the output under this heading in a fenced block:

```bash
source .venv/bin/activate
python3 << 'PYEOF'
import ast

def extract_assign(filepath, name):
    with open(filepath) as f:
        tree = ast.parse(f.read())
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    try:
                        return ast.literal_eval(node.value)
                    except Exception:
                        return None
    return None

builder = extract_assign("serving/predictor.py", "_BUILDER_KNOWN_COLS")
feature_cols = extract_assign("hana_app/core/ml_runner.py", "FEATURE_COLS")
etl_cols = extract_assign("scripts/features/feature_engineer.py", "ETL_NUMERIC_COLS")
dataset_cols = extract_assign("scripts/datasets/contracts.py", "ML_DATASET_REQUIRED_COLUMNS")

builder_set = set(builder) if builder else set()
feature_set = set(feature_cols) if feature_cols else set()
etl_set = set(etl_cols) if etl_cols else set()
dataset_set = set(dataset_cols) if dataset_cols else set()
all_features = builder_set | feature_set | etl_set | dataset_set

print(f"{'feature':<30} {'BUILDER':>8} {'FEATURE':>8} {'ETL':>8} {'DATASET':>8}")
for f in sorted(all_features):
    print(f"{f:<30} {'Y' if f in builder_set else '-':>8} {'Y' if f in feature_set else '-':>8} {'Y' if f in etl_set else '-':>8} {'Y' if f in dataset_set else '-':>8}")

print()
print("=== Set Differences ===")
print(f"BUILDER \\ FEATURE_COLS: {sorted(builder_set - feature_set)}")
print(f"FEATURE_COLS \\ BUILDER: {sorted(feature_set - builder_set)}")
print(f"BUILDER \\ ETL: {sorted(builder_set - etl_set)}")
print(f"ETL \\ BUILDER: {sorted(etl_set - builder_set)}")
print(f"FEATURE_COLS \\ ETL: {sorted(feature_set - etl_set)}")
print(f"ETL \\ FEATURE_COLS: {sorted(etl_set - feature_set)}")
PYEOF
```

## Known Historical Context
- `dup_efmdc` in `_BUILDER_KNOWN_COLS`/`FEATURE_COLS` but NOT in `ETL_NUMERIC_COLS`: edi->efmdc (HIRA classification) bridge feature. P2 addition.
- `sex_m` in `_BUILDER_KNOWN_COLS`/`FEATURE_COLS` but NOT in `ETL_NUMERIC_COLS`: sex feature added at serving/UI level, not in ETL numeric pipeline.
- `avg_drug_duration`/`long_term_drug_count` in `_BUILDER_KNOWN_COLS` but NOT in `FEATURE_COLS`: serving-only features, not in UI training path.
- `patient_id`/`risk_level` in `ML_DATASET_REQUIRED_COLUMNS` only: metadata/label columns, not ML features.
- commit `d201743` train/serve regression root cause: feature list dispersion across these sources.

## Design Intent
These differences are NOT bugs to fix. Each profile has its own contract. Phase 2A will codify these differences. Phase 2B will record them in characterization tests. No flattening.
```

- [ ] **Step 10: Verify no code was changed**

Run:
```bash
git diff --stat
```
Expected: no `.py` file changes. Only new untracked `.md` files under `docs/superpowers/reports/contract-baseline/`.

---

### Task 2: Pickle/Joblib Module-Path Risk Inspection

**Files:**
- Create: `docs/superpowers/reports/contract-baseline/phase0a-bundle-metadata-record.md`
- Read: `serving/predictor.py`, `serving/dl_predictor.py`, `hana_app/core/hierarchical_runner.py`

**CRITICAL SAFETY RULE:** This task does NOT load, unpickle, or execute any model bundle. It only records configured paths from code and documents pickle/joblib module-path risks from static analysis. Any bundle byte inspection requires explicit OpenCode LO approval and uses `pickletools` only. No `pickle.load`, `pickle.loads`, or `joblib.load`.

- [ ] **Step 1: Record configured model paths and deserialization sites**

Run:
```bash
grep -n "MODELS_DIR\|models/\|model_dir\|model_path\|ddi_model_\|stage_meta\|gat_model\|\.pkl\|\.joblib\|pickle\.load\|pickle\.loads\|joblib\.load" serving/predictor.py | head -30
```
Insert the output in the report.

- [ ] **Step 2: Record classes that may be referenced in pickle payloads**

Run:
```bash
grep -n "class.*Predictor\|class.*MLModel\|class.*Wrapper\|class.*Constant" serving/predictor.py hana_app/core/hierarchical_runner.py | head -10
```
Insert the output in the report.

- [ ] **Step 3: Write the bundle metadata record**

Write to `docs/superpowers/reports/contract-baseline/phase0a-bundle-metadata-record.md`:

```markdown
# Phase 0A: Deployed Bundle Metadata Record

**CRITICAL SAFETY:** This record does NOT load, unpickle, or execute any model bundle. All information is from static code analysis only.

## 1. Configured Model Paths (from code, no file access)

### 1.1 tabular_binary
- Path pattern: `models/ddi_model_{partition}.pkl`
- Format: pickle dict with keys: `model`, `best_threshold`, `trainer_class`, `feature_names`, `artifact_version`, `partition`, `scaler_path`, `selector_path`, `ddi_feature_semantics_version`, `weights`
- Sub-models (EnsembleTrainer): `*.xgb.pkl`, `*.lgb.pkl`
- GAT sub-model (EnsembleTrainer3Way): `gat_model.pt`, `gat_graph_meta.json`

### 1.2 hierarchical
- Path pattern: model directory with `stage_meta.json`, `stage1_red.joblib`, `stage2_yellow.joblib`
- `stage_meta.json` keys: `feature_cols`, `stage2_labels`, `thresholds` (tau_red, tau_review), `ddi_feature_semantics_version`, `feature_semantics_version`

### 1.3 dl_history
- Path pattern: `models/dl/` with bundle directory
- Required files: insert `DL_BUNDLE_REQUIRED_FILES` from Task 1 AST output, plus `MANIFEST.json`

## 2. Pickle/Joblib Deserialization Sites

Insert the grep output from Step 1 here in a fenced block.

## 3. Module-Path Risks

### 3.1 _EnsembleWrapper (CRITICAL)
- Defined at: `serving/predictor.py:471-486` (INSIDE `MLModel.load` method, local class)
- Pickle reference: `serving.predictor.MLModel.load.<locals>._EnsembleWrapper`
- Risk: If class is moved to module level or method is refactored, old ensemble bundles will fail to unpickle with `ModuleNotFoundError` or `AttributeError`.
- Mitigation: Phase 2B characterization test should record this risk. Phase 3 refactor must preserve compatibility import or class location.

### 3.2 _ConstantNegativeStage1
- Defined at: `hana_app/core/hierarchical_runner.py:40-59` (module level)
- Pickle reference: `hana_app.core.hierarchical_runner._ConstantNegativeStage1`
- Risk: If module path changes (e.g., moved to serving domain), old stage1 bundles will fail.
- Mitigation: Phase 3 must preserve compatibility import.

### 3.3 Dev vs Prod Module Path
- Dev: `sys.path` includes project root, modules resolved as `serving.predictor`, `hana_app.core.hierarchical_runner`, `scripts.etl.prescription_aggregator`
- Prod (Windows closed network): same structure, but `sys.path.insert(0, str(Path(__file__).parent.parent))` in multiple locations (`serving/predictor.py:173,215`) may cause different resolution order
- Risk: If `sys.path` order differs, pickle may resolve classes to different module names, causing unpickle failure.
- Mitigation: Record current `sys.path` in baseline report (Phase 0B Task 3).

## 4. Bundle Byte Inspection (REQUIRES LO APPROVAL)

**DEFAULT: Do NOT inspect bundle bytes.** The plan records configured paths and static code analysis only.

If OpenCode LO explicitly approves byte inspection of a specific bundle:
- Use `pickletools.dis(bytes)` or `pickletools.genops(bytes)` (non-executing, safe)
- Do NOT use `pickle.load`, `pickle.loads`, `joblib.load`, or any executing deserialization
- Record the LO approval reference in the report
- Example safe command (DO NOT RUN without LO approval):
```bash
# pickletools.dis is non-executing: it disassembles without importing classes
python -c "import pickletools; pickletools.dis(open('models/ddi_model_202407.pkl','rb'))"  # REQUIRES LO APPROVAL
```

## 5. JSON/Text Metadata Extraction (Safe, No Unpickling)

JSON metadata files can be safely read without unpickling risk. The following commands are safe to run with LO approval but are NOT run by default in this plan:

```bash
# cat models/<hierarchical_dir>/stage_meta.json  # REQUIRES LO APPROVAL (protected path)
# cat models/dl/<bundle>/model_config.json  # REQUIRES LO APPROVAL (protected path)
# cat models/dl/<bundle>/schema_version.json  # REQUIRES LO APPROVAL (protected path)
```
```

- [ ] **Step 4: Verify no code was changed and no bundles were loaded**

Run:
```bash
git diff --stat
```
Expected: no `.py` file changes, no changes to `models/`, `mlruns/`, or any protected path.

---

### Task 3: Phase 0A Review Gate and Commit

**Files:**
- Read: all three Phase 0A report files

- [ ] **Step 1: Verify all 0A acceptance criteria from spec section 7.2**

Check each criterion against the produced documents:
1. 4 profile contract specifications exist in `phase0a-profile-contract-map.md` (sections 1-4)
2. Feature list dispersion table exists in `phase0a-feature-dispersion-table.md`
3. Deployed bundle feature names, dtype, defaults recorded in `phase0a-profile-contract-map.md` and `phase0a-bundle-metadata-record.md`
4. FEATURE_SCHEMA_LENIENT runtime/deadline state recorded in `phase0a-profile-contract-map.md` (section 5)
5. pickle/joblib module path state recorded in `phase0a-bundle-metadata-record.md`
6. Required resource semantics recorded in `phase0a-profile-contract-map.md` (section 6)
7. reload/rollback behavior recorded in `phase0a-profile-contract-map.md` (sections 1.6, 2.7, 4.7)
8. Physical DataFrame/Parquet column order recorded in `phase0a-profile-contract-map.md` (section 7)
9. No code changed (verify with `git diff --stat`)

If any criterion is not fully met, record the gap and flag as WARN. Route to OpenCode LO.

- [ ] **Step 2: Commit Phase 0A outputs**

Run:
```bash
git add docs/superpowers/reports/contract-baseline/phase0a-profile-contract-map.md \
       docs/superpowers/reports/contract-baseline/phase0a-feature-dispersion-table.md \
       docs/superpowers/reports/contract-baseline/phase0a-bundle-metadata-record.md
git commit -m "docs: Phase 0A contract baseline inventory

Read-only inventory of policy/runtime state across 4 profiles. No code
changes. No protected artifact loading. Maps feature lists, label spaces,
thresholds, semantic versions, validation rules, dtypes, defaults,
resource semantics, reload/rollback, physical column order, pickle/joblib
module-path risks, and FEATURE_SCHEMA_LENIENT runtime/deadline state.

Spec: docs/superpowers/specs/2026-07-12-opencode-lo-contract-design.md (section 7.2)"
```
Expected: commit created with only the 3 new report files.

- [ ] **Step 3: OpenCode LO Phase 0A gate**

Phase 0A is complete when OpenCode LO verifies all acceptance criteria are met. OpenCode LO must explicitly approve before Phase 0B begins.

---

## Phase 0B: Reproducible Contract Baseline Report

### Task 4: serving/predictor.py Dependency Graph

**Files:**
- Create: `docs/superpowers/reports/contract-baseline/phase0b-dependency-graph.md`
- Read: `serving/predictor.py`

- [ ] **Step 1: Extract all imports from serving/predictor.py via AST**

Run:
```bash
source .venv/bin/activate
python3 << 'PYEOF'
import ast

with open("serving/predictor.py") as f:
    tree = ast.parse(f.read())

top_level = []
dynamic = []

for node in ast.walk(tree):
    if isinstance(node, ast.Import):
        for a in node.names:
            entry = (node.lineno, a.name)
            if node.col_offset == 0:
                top_level.append(entry)
            else:
                dynamic.append(entry)
    elif isinstance(node, ast.ImportFrom):
        mod = node.module or ""
        names = [a.name for a in node.names]
        entry = (node.lineno, mod, names)
        if node.col_offset == 0:
            top_level.append(entry)
        else:
            dynamic.append(entry)

print("=== TOP-LEVEL IMPORTS ===")
for entry in top_level:
    print(entry)
print()
print("=== DYNAMIC IMPORTS (in-function) ===")
for entry in dynamic:
    print(entry)
PYEOF
```
Insert the output in the report.

- [ ] **Step 2: Check for circular dependencies via AST**

Run:
```bash
source .venv/bin/activate
python3 << 'PYEOF'
import ast, os

# Check if any module that serving/predictor.py imports also imports serving.*
# Check: hana_app.core.hierarchical_runner, scripts.etl.prescription_aggregator,
#        scripts.etl.models, scripts.etl.overlap_calculator, scripts.etl.clinical_rules,
#        scripts.etl.code_standardizer, scripts.features.cyp_features,
#        scripts.train.gat_trainer, rules.safety_net, rules.duplicate_detector

files_to_check = [
    "hana_app/core/hierarchical_runner.py",
    "scripts/etl/prescription_aggregator.py",
    "scripts/etl/models.py",
    "scripts/etl/overlap_calculator.py",
    "scripts/etl/clinical_rules.py",
    "scripts/etl/code_standardizer.py",
    "scripts/features/cyp_features.py",
    "scripts/train/gat_trainer.py",
    "rules/safety_net.py",
    "rules/duplicate_detector.py",
]

for filepath in files_to_check:
    if not os.path.exists(filepath):
        print(f"SKIP (not found): {filepath}")
        continue
    with open(filepath) as f:
        tree = ast.parse(f.read())
    serving_imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if "serving" in a.name:
                    serving_imports.append((node.lineno, a.name))
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if "serving" in mod:
                serving_imports.append((node.lineno, mod))
    if serving_imports:
        print(f"CIRCULAR DEPENDENCY FOUND in {filepath}:")
        for line, mod in serving_imports:
            print(f"  line {line}: imports {mod}")
    else:
        print(f"OK (no serving.* imports): {filepath}")
PYEOF
```
Insert the output in the report. Expected: all files OK (no circular dependencies).

- [ ] **Step 3: Write the dependency graph document**

Write to `docs/superpowers/reports/contract-baseline/phase0b-dependency-graph.md`:

```markdown
# Phase 0B: serving/predictor.py Dependency Graph

## 1. Import Inventory

### 1.1 Top-Level Imports (module scope)
Insert the top-level imports from Step 1 AST output here in a fenced block.

### 1.2 Dynamic Imports (function scope)
Insert the dynamic imports from Step 1 AST output here in a fenced block.

## 2. Dependency Graph (ASCII)

```
serving/predictor.py
|
+-- serving.schemas
|   +-- DDIAlert, DLPredictionResult, DrugItem, PredictRequest, PredictResponse
|   +-- RiskLevel, Severity, INTERVENTION_MAP
|
+-- serving.dl_predictor
|   +-- DLModel
|
+-- serving.hana_history
|   +-- HANAHistoryProvider
|
+-- rules.risk_drug_constants
|   +-- HIGH_RISK_KEYWORDS, HIGH_RISK_ATC_PREFIXES
|   +-- RENAL_RISK_KEYWORDS, RENAL_RISK_ATC_PREFIXES
|   +-- HEPATIC_RISK_KEYWORDS, HEPATIC_RISK_ATC_PREFIXES
|
+-- [dynamic] scripts.etl.prescription_aggregator
|   +-- count_ddi_severities, ddi_pair_severities
|   +-- _fill_dup_features, detect_triple_whammy, detect_risk_drug
|   +-- DDI_FEATURE_SEMANTICS_VERSION, FEATURE_SEMANTICS_VERSION
|   +-- _RENAL_RISK_KEYWORDS, _RENAL_RISK_ATC_PREFIXES
|   +-- _HEPATIC_RISK_KEYWORDS, _HEPATIC_RISK_ATC_PREFIXES
|
+-- [dynamic] scripts.etl.overlap_calculator
|   +-- calculate_overlaps_for_patient, get_concurrent_drug_count
|
+-- [dynamic] scripts.etl.clinical_rules
|   +-- collect_red_triggers, collect_severe_immediate_triggers
|
+-- [dynamic] scripts.etl.code_standardizer
|   +-- CodeStandardizer
|
+-- [dynamic] scripts.etl.models
|   +-- PrescriptionRecord, PatientFeatures
|
+-- [dynamic] scripts.features.cyp_features
|   +-- CYPFeatureExtractor
|
+-- [dynamic] scripts.train.gat_trainer
|   +-- GATTrainer (EnsembleTrainer3Way)
|
+-- [dynamic] hana_app.core.hierarchical_runner
|   +-- predict_risk, ACTION_BY_LABEL, STAGE2_LABELS
|
+-- [dynamic] rules.safety_net
|   +-- SafetyNet
|
+-- [dynamic] rules.duplicate_detector
    +-- DuplicateDetector
```

## 3. Serving -> hana_app Dependency (Explicit)

This is the P1 dependency from the spec.

| Import | Location | Symbol | Type |
|---|---|---|---|
| `from hana_app.core.hierarchical_runner import STAGE2_LABELS` | serving/predictor.py:674-676 | `STAGE2_LABELS` | Label space guard (load time) |
| `from hana_app.core.hierarchical_runner import predict_risk` | serving/predictor.py:788 | `predict_risk` | Inference (call time) |
| `from hana_app.core.hierarchical_runner import ACTION_BY_LABEL` | serving/predictor.py:1470 | `ACTION_BY_LABEL` | Action mapping (call time) |

This dependency is NOT removed in Phase 0A/0B. It is recorded for Phase 3 (out of scope).

## 4. Circular Dependency Check

Insert the output from Step 2 here in a fenced block.

Result: No circular dependencies detected. The dependency graph is a DAG with `serving/predictor.py` as a consumer of `hana_app.core.hierarchical_runner`, `scripts.etl.*`, `scripts.features.*`, and `rules.*`.
```

- [ ] **Step 4: Verify no code was changed**

Run:
```bash
git diff --stat
```
Expected: no `.py` file changes.

---

### Task 5: Reproducible Baseline Report

**Files:**
- Create: `docs/superpowers/reports/contract-baseline/phase0b-baseline-report.md`
- Create: `docs/superpowers/reports/contract-baseline/README.md`

- [ ] **Step 1: Record Python runtime and git state**

Run:
```bash
source .venv/bin/activate
python --version
python -c "import sys; print(f'executable: {sys.executable}'); print(f'path[:5]: {sys.path[:5]}')"
git log --oneline -5
git rev-parse HEAD
git status --short
```
Record all output for the report header.

- [ ] **Step 2: Record all contract state via AST (reproducible)**

Run the same AST extraction script from Task 1 Step 1 again. This is the reproducible baseline: the same command should produce the same output if the code has not changed. Insert the full JSON output in the baseline report.

- [ ] **Step 3: Record serving/predictor.py import inventory**

Run the same AST import extraction from Task 4 Step 1 again. Insert the output in the baseline report.

- [ ] **Step 4: Write the baseline report**

Write to `docs/superpowers/reports/contract-baseline/phase0b-baseline-report.md`:

```markdown
# Phase 0B: Reproducible Contract Baseline Report

**Generated:** insert date from Step 1
**Git commit:** insert from Step 1
**Python:** insert from Step 1
**Working tree:** insert from Step 1

## Reproduction Instructions

1. Activate Python 3.12: `source .venv/bin/activate && python --version`
2. Run the AST extraction script from Task 1 Step 1
3. Run the AST import extraction from Task 4 Step 1
4. Compare output to the recorded output in this report
5. If output differs, the contract state has changed since baseline

## 1. Contract State (AST Extraction)

Insert the full JSON output from Step 2 here in a fenced block.

## 2. serving/predictor.py Import Inventory

Insert the full output from Step 3 here in a fenced block.

## 3. Cross-Source Feature Dispersion

Reference: see `phase0a-feature-dispersion-table.md` for the full dispersion table.

## 4. Dependency Graph

Reference: see `phase0b-dependency-graph.md` for the full dependency graph.

## 5. Pickle/Joblib Module-Path Risks

Reference: see `phase0a-bundle-metadata-record.md` for the full risk record.

## 6. Physical Column Order

Reference: see `phase0a-profile-contract-map.md` section 7.

## 7. Reload/Rollback Behavior

Reference: see `phase0a-profile-contract-map.md` sections 1.6, 2.7, 4.7.
```

- [ ] **Step 5: Write the README index**

Write to `docs/superpowers/reports/contract-baseline/README.md`:

```markdown
# Contract Baseline Inventory Report

This directory contains the Phase 0A/0B contract baseline inventory for the MODE_11_hana project.

## Files

| File | Phase | Content |
|---|---|---|
| `phase0a-profile-contract-map.md` | 0A | Profile-by-profile contract state mapping (4 profiles) |
| `phase0a-feature-dispersion-table.md` | 0A | Cross-source feature list dispersion table |
| `phase0a-bundle-metadata-record.md` | 0A | Deployed bundle metadata and pickle/joblib module-path risks |
| `phase0b-dependency-graph.md` | 0B | serving/predictor.py dependency graph and circular dependency check |
| `phase0b-baseline-report.md` | 0B | Reproducible contract baseline report with frozen command outputs |

## Reproduction

1. Activate Python 3.12: `source .venv/bin/activate`
2. Verify: `python --version` (must be 3.12.x)
3. Follow the commands in `phase0b-baseline-report.md`
4. Compare outputs to the recorded baseline

## Safety

- No code was changed in producing this baseline
- No protected artifacts were loaded or unpickled
- No Nov->Dec holdout data was accessed
- All work is freeze-safe (RESEARCH_TRACK_FROZEN)

## Spec Reference

`docs/superpowers/specs/2026-07-12-opencode-lo-contract-design.md` (sections 7.2, 7.3)

## Handoff

This baseline is frozen and handed off to Phase 1 (minimal tooling). Phase 1 consumes this baseline to set up pytest baseline, Ruff check-only, dependency drift check, and sunset monitor.
```

- [ ] **Step 6: Verify no code was changed**

Run:
```bash
git diff --stat
```
Expected: no `.py` file changes.

---

### Task 6: Phase 0B Review Gate and Commit

**Files:**
- Read: all three Phase 0B report files

- [ ] **Step 1: Verify all 0B acceptance criteria from spec section 7.3**

Check each criterion:
1. Dependency graph document exists in `phase0b-dependency-graph.md`
2. `serving -> hana_app` dependency explicitly recorded (section 3 of dependency graph)
3. Circular dependency check completed (section 4 of dependency graph)
4. Profile state baseline report is reproducible (all commands recorded in `phase0b-baseline-report.md`)
5. No code changed, no new scripts written, no commits of code

If any criterion is not fully met, record the gap and flag as WARN. Route to OpenCode LO.

- [ ] **Step 2: Commit Phase 0B outputs**

Run:
```bash
git add docs/superpowers/reports/contract-baseline/phase0b-dependency-graph.md \
       docs/superpowers/reports/contract-baseline/phase0b-baseline-report.md \
       docs/superpowers/reports/contract-baseline/README.md
git commit -m "docs: Phase 0B contract baseline report

Read-only reproducible baseline report and serving/predictor.py dependency
graph. No code changes. No new scripts. No protected artifact loading.
Records all profile contract state via AST extraction for reproducibility.
Dependency graph confirms serving -> hana_app dependency and no circular
dependencies.

Spec: docs/superpowers/specs/2026-07-12-opencode-lo-contract-design.md (section 7.3)"
```
Expected: commit created with only the 3 new report files.

- [ ] **Step 3: OpenCode LO Phase 0B gate**

Phase 0B is complete when OpenCode LO verifies all acceptance criteria are met. The frozen baseline is handed off to Phase 1.

---

## Rollback

Since Phase 0A/0B produces only documentation files and makes zero code changes, rollback is trivial:

- [ ] **Rollback Step 1: Delete the report directory**

Run:
```bash
rm -rf docs/superpowers/reports/contract-baseline/
```

- [ ] **Rollback Step 2: Revert commits (if committed)**

Run:
```bash
git log --oneline -5
# Identify the Phase 0A and 0B commits by their messages, then revert each by hash
# Example: git revert abc1234  (where abc1234 is the Phase 0A commit hash)
# Run git revert once per commit, substituting the actual hash from the log output above
```

- [ ] **Rollback Step 3: Verify no code was affected**

Run:
```bash
git diff HEAD~2 --stat
```
Expected: only documentation files under `docs/superpowers/reports/contract-baseline/` differ. No `.py` files changed.

### Rollback Triggers

- Any `.py` file was modified during Phase 0A/0B execution
- Any protected artifact (`mlruns/`, `packages_win/py312/`, generated `.parquet`, `out/`) was touched
- Any bundle was loaded or unpickled without explicit OpenCode LO approval
- Any Nov->Dec holdout data was accessed

If any trigger fires: STOP, abort the current step, route to OpenCode LO immediately.

---

## Constraints Summary

1. **No code changes:** No `.py` files are modified. Only new `.md` files under `docs/superpowers/reports/contract-baseline/` are created.
2. **No new scripts:** No new Python scripts are written or committed. Only inline `python3 << 'PYEOF'` AST extraction and existing read-only commands (`grep`, `sed`) are used.
3. **No runtime imports of serving/hana_app:** All constant extraction uses `ast.literal_eval` on source files, not `import`. This ensures the baseline works even when optional dependencies (torch, sklearn, etc.) are absent.
4. **No protected artifact loading:** `mlruns/`, `packages_win/py312/`, generated `.parquet`, `out/` are not loaded, unpickled, or modified. Bundle byte inspection requires explicit OpenCode LO approval and uses `pickletools` only (non-executing). No `pickle.load`, `pickle.loads`, or `joblib.load`.
5. **No Nov->Dec holdout access:** No holdout data is accessed. `RESEARCH_TRACK_FROZEN`.
6. **No Phase 1+ work:** This plan stops at Phase 0B. No tooling implementation, no pytest baseline, no Ruff config, no drift check.
7. **OpenCode is final LO:** All worker results return evidence to OpenCode. OpenCode verifies before reporting success. Workers do not make user-facing decisions.
8. **Phase 0A precedes 0B:** 0B consumes 0A's mapping. 0B does not start until 0A's review gate passes.
9. **Existing working-tree changes:** Any pre-existing working-tree changes are NOT included in commits from this plan. Commits stage ONLY files under `docs/superpowers/reports/contract-baseline/`.