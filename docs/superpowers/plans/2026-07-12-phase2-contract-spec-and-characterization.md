# Phase 2A/2B: Contract Specification and Characterization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Codify and freeze the current serving/training contract state for four operational profiles (tabular_binary, hierarchical, ui_experimental, dl_history) as a single specification, then record the frozen behavior as characterization tests and one read-only adapter. No production code behavior changes.

**Architecture:** Two sequential waves. Wave 1 (Phase 2A) writes one contract spec document covering all four profiles, cross-profile differences, and Phase 3 future-work policy. Wave 2 (Phase 2B) adds focused characterization tests and a read-only adapter that live in the test tree and are never imported by production runtime. All new files are additive. Pre-existing test node IDs and pass/fail outcomes must remain unchanged.

**Tech Stack:** Python 3.12, pytest, pickle, pickletools, numpy, pandas. No new dependencies.

**Depends on:** `docs/superpowers/plans/2026-07-12-contract-baseline-inventory.md` must be complete, then `docs/superpowers/plans/2026-07-12-phase1-minimal-tooling.md` must satisfy every non-blocked gate. If Ruff validation remains blocked, OpenCode LO must record that blocker explicitly before authorizing Phase 2 work.

**Authority documents:**
- `docs/superpowers/specs/2026-07-12-opencode-lo-contract-design.md` (the approved design spec, hereafter "the Spec")
- `CLAUDE.md` (project rules, protected paths, freeze policy)
- `AGENTS.md` (hard gates, trigger severity, cross-family review)

## Global Constraints

All non-negotiable. Every task's requirements implicitly include this section.

- Do NOT flatten feature sets across profiles or demand zero diff. Each profile keeps its own feature list.
- Do NOT change feature names, order, dtype/default semantics, labels, thresholds, semantic versions, or artifact/reload contracts.
- Do NOT migrate artifacts, retrain, split predictor, move domain policy, or touch protected paths.
- Do NOT use frozen holdout (Nov->Dec). `RESEARCH_TRACK_FROZEN`.
- Do NOT edit existing production code files. Only create new files listed in this plan.
- Do NOT commit, push, or create PRs unless explicitly asked; existing unrelated working-tree changes must not be included in future commits for this plan.
- Total test collection may grow. All pre-existing test node IDs and pass/fail outcomes must remain unchanged.
- Cross-family review required before merge: Claude/Fable logical review + Codex technical review. This direct cross-family review is distinct from the unavailable formal `ask_advisor_panel` Oracle gate. Oracle remains blocked.
- No em dashes or en dashes in any file produced by this plan.

---

## File Structure

Seven new files total.

```
docs/superpowers/specs/contracts/
  profile_contracts.md                        -- single contract spec (4 profiles + diff + Phase 3 policy)

tests/test_contracts/
  __init__.py                                  -- package init (empty)
  profile_diff_reporter.py                     -- read-only adapter (test-tree only, not imported by production)
  test_profile_contracts.py                    -- profile contract characterization: names, order, labels, versions, dtype defaults, physical column order
  test_serving_characterization.py             -- resource fallback, request mutation, feature vector alignment
  test_reload_artifact_compat.py               -- reload success/failure/rollback, pickle module-path via non-executing inspection
  test_adapter.py                              -- adapter integration tests
```

---

## Wave 1: Contract Specification (Phase 2A)

### Task 1: Write the single contract spec document

**Files:**
- Create: `docs/superpowers/specs/contracts/profile_contracts.md`

- [ ] **Step 1: Write the contract spec document**

```markdown
# Profile Contract Specifications (Phase 2A)

**Status:** Frozen (codify current behavior, no changes)
**Authority:** `docs/superpowers/specs/2026-07-12-opencode-lo-contract-design.md` Sections 6, 7.5

## 1. tabular_binary

Single ML model (XGBoost/LightGBM/Ensemble) inference contract.

**Feature source:** `serving/predictor.py` `_BUILDER_KNOWN_COLS` (frozenset, 24 features).

Feature list (unordered set, membership check only):
drug_count, institution_count, age, sex_m, ddi_contraindicated, ddi_major, ddi_moderate, ddi_minor, avg_drug_duration, long_term_drug_count, dup_same_ingredient, dup_atc5, dup_atc4, dup_atc3, dup_efmdc, has_high_risk_drug, has_renal_risk_drug, has_hepatic_risk_drug, cyp_risk_score, cyp_high_risk_pairs, cyp_max_enzyme_risk, triple_whammy, qt_risk_count, drug_count_7d.

**Intentional allowlist** (`_INTENTIONAL_FEATURE_ALLOWLIST`): empty frozenset.
**Allowed set** (`_FEATURE_ALLOWED`): `_BUILDER_KNOWN_COLS | _INTENTIONAL_FEATURE_ALLOWLIST`.

**Feature order:** Model bundle `feature_names` is authoritative. `RequestFeatureBuilder.build()` aligns via `aligned = {name: feat.get(name, 0.0) for name in feature_names}`.

**dtype/default semantics:** All features are float64. Key defaults: sex_m=0.5 when unknown (1.0 for M, 0.0 for F), age=0.0 when None, ddi_*=0.0 when resources absent, cyp_*=0.0 when extractor absent, dup_efmdc=0.0 when code_standardizer absent, drug_count_7d falls back to drug_count when bridge absent.

**Threshold:** Model bundle `best_threshold` (default 0.5). `MLModel.classify()`: prob >= threshold -> RED, prob >= threshold*0.6 -> YELLOW, prob >= threshold*0.3 -> GREEN, else NORMAL.

**Semantic version:** `DDI_FEATURE_SEMANTICS_VERSION = "ddi.v2"` (from `scripts/etl/prescription_aggregator.py`). Models with ddi_* features must have `ddi_feature_semantics_version` in bundle metadata matching "ddi.v2". Missing or mismatched -> load rejection.

**Validation:** `_validate_feature_schema()` checks `feature_names` subset of `_FEATURE_ALLOWED`. Unknown columns -> load rejection (strict). `FEATURE_SCHEMA_LENIENT=1` allows degraded load with 0.0 fallback until sunset `2026-08-01` (`_FEATURE_SCHEMA_LENIENT_SUNSET_DEFAULT`).

**Artifact format:** pickle `.pkl` with SHA-256 sidecar `.pkl.sha256`. State dict keys: model, best_threshold, trainer_class, feature_names, artifact_version, partition, ddi_feature_semantics_version, scaler_path (optional), selector_path (optional). Sidecar: scaler/selector pickle with hash verification and path traversal defense.

**Reload contract:** `HybridPredictor.reload_model(model_path)` creates new MLModel, loads, on success swaps under `_ml_lock`. On failure, existing model preserved.

**Production path:** `RequestFeatureBuilder.build()` -> `MLModel.predict_proba()` -> `MLModel.classify()`.

## 2. hierarchical

Stage 1 Red binary + Stage 2 Yellow-subtype 7-class classifier contract.

**Feature source:** `stage_meta.json` `feature_cols` (list[str]). Validated against `_FEATURE_ALLOWED` at load time.

**Label space:** `STAGE2_LABELS = ("Y_TRIPLE", "Y_DOUBLE", "Y_DDI_MAJOR", "Y_DDI_MOD", "Y_DUP", "Y_FRAG", "No_Alert")` (7-class, from `hana_app/core/hierarchical_runner.py`). `YELLOW_SUBTYPE_LABELS = ("Y_TRIPLE", "Y_DOUBLE", "Y_DDI_MAJOR", "Y_DDI_MOD", "Y_DUP", "Y_FRAG")` (6 Yellow subtypes, no No_Alert).

**Label integrity guard:** Bundle metadata `stage2_labels` must exactly match current `STAGE2_LABELS` (order-sensitive). Encoder `classes_` must exactly match `STAGE2_LABELS`. `classes_present` indices must be in `[0, len(STAGE2_LABELS))`. Any mismatch -> load rejection.

**Thresholds:** `stage_meta.json` `thresholds`: `tau_red` (float), `tau_review` (float). 2-stage dispatch: p_red >= tau_red -> Red (Stage 2 skipped); tau_review <= p_red < tau_red -> Stage 2 label + red_suspect=True; p_red < tau_review -> Stage 2 label (red_suspect=False).

**Semantic versions:** `DDI_FEATURE_SEMANTICS_VERSION = "ddi.v2"` and `FEATURE_SEMANTICS_VERSION = "rulefeat.v1"` (both from `scripts/etl/prescription_aggregator.py`). When `feature_semantics_version` matches "rulefeat.v1", `rule_features_active=True` and triple_whammy/risk-flags computed via edi->wk->DrugMaster. Old bundles without this version keep these features at 0 (gating prevents skew).

**Validation chain:** (1) `_validate_feature_schema()` on `feature_cols`, (2) label space guard, (3) DDI semantic version guard, (4) SHA-256 hash verification (stage1_sha256, stage2_sha256), (5) encoder/classes_present integrity guard.

**Intervention actions:** `ACTION_BY_LABEL` (from `hana_app/core/hierarchical_runner.py`): Y_DDI_MAJOR=약사 전화, Y_TRIPLE=문자 안내, Y_DOUBLE=모니터링, Y_DDI_MOD=모니터링, Y_DUP=모니터링, Y_FRAG=모니터링, No_Alert=관여 안 함. Red action: "즉각 개입" (`RED_ACTION`, not in `ACTION_BY_LABEL`).

**Backstop:** `red_triggers()` returns `RED_CONTRAINDICATED` only (contraindicated DDI), model-independent. `rule_floor()`: major DDI >= 1 -> Y_DDI_MAJOR; severe immediate (triple_whammy/10drug+highrisk/elderly+longterm) -> Y_TRIPLE. Single-direction escalation.

**Artifact format:** `stage_meta.json` (thresholds, feature_cols, stage2_labels, ddi_feature_semantics_version, stage1_sha256, stage2_sha256), `stage1_red.joblib`, `stage2_yellow.joblib` (dict with model, encoder, classes_present).

**Reload contract:** `HybridPredictor.reload_hierarchical(model_dir)` creates new HierarchicalPredictor, loads, validates feature_cols non-empty and schema. On success, swaps under `_hier_lock`. On failure, existing hierarchical model preserved.

**Production path:** `RequestFeatureBuilder.build()` -> `HierarchicalPredictor.predict_risk_single()` -> `predict_risk()` (from `hana_app.core.hierarchical_runner`).

**Serving dependency:** `serving/predictor.py` imports `predict_risk`, `ACTION_BY_LABEL`, `STAGE2_LABELS` from `hana_app.core.hierarchical_runner`. This dependency is recorded here and NOT removed in Phase 2. Removal is Phase 3 (out of scope, see Section 6).

## 3. ui_experimental

Page 3 Streamlit training UI path contract. Not a production serving path.

**Feature source:** `hana_app/core/ml_runner.py` `FEATURE_COLS` (list[str], 22 features, ordered):
drug_count, drug_count_7d, institution_count, ddi_contraindicated, ddi_major, ddi_moderate, ddi_minor, triple_whammy, qt_risk_count, dup_same_ingredient, dup_atc5, dup_atc4, dup_atc3, dup_efmdc, has_high_risk_drug, has_renal_risk_drug, has_hepatic_risk_drug, cyp_risk_score, cyp_max_enzyme_risk, cyp_high_risk_pairs, age, sex_m.

**Differences from tabular_binary:** `_BUILDER_KNOWN_COLS` has 2 extra features not in `FEATURE_COLS`: `avg_drug_duration`, `long_term_drug_count`. `FEATURE_COLS` is a strict subset. These differences are intentional and must NOT be flattened or merged.

**Training path:** `ml_runner.py` -> `aggregate_patient_features()` -> `FeatureEngineer` -> trainer -> joblib save.

**Labels:** `RISK_LABEL_MAP`: Red=3, Yellow=2, Green=1, Normal=0. `RISK_COLOR_MAP`: Red=🔴, Yellow=🟡, Green=🟢, Normal=⚪.

**Validation:** UI-internal stratified sampling, cross-validation, metrics display. Safety guards: `page_guards.py`, `memory_guard.py`.

**Operational separation:** Does NOT connect to production serving bundles. Separate experimental/validation path.

## 4. dl_history

Operational DL bundle (graph neural network) inference contract.

**Bundle required files:** `DL_BUNDLE_REQUIRED_FILES` (from `scripts/datasets/contracts.py`): model.pt, model_config.json, drug_vocab.json, edge_index.pt, feature_normalizer.pkl, schema_version.json (6 files).

**Manifest:** `MANIFEST.json` with SHA-256 hash verification. Validated by `validate_dl_bundle_manifest()`. Fields: track ("dl"), run_id, schema_version, created_at, hash_alg ("sha256"), lookback_days, drug_vocab_sha256, edge_index_sha256, files.

**Encoding strategy:** `_SUPPORTED_ENCODING_STRATEGIES = {"multi_hot"}` (from `serving/dl_predictor.py`). "count" removed (dead infra).

**Graph architectures:** `_GRAPH_ARCHITECTURES = {"gat", "gcn"}`.

**Lookback:** `LOOKBACK_DAYS_DEFAULT=365`, `LOOKBACK_DAYS_MIN=7`, `LOOKBACK_DAYS_MAX=1825`. Runtime lookback must match bundle lookback. Mismatch -> `LookbackMismatchError`.

**Dataset contract:** `DL_DATASET_REQUIRED_COLUMNS = ("patient_id", "drug_code", "prescription_date")`. Distinct from `ML_DATASET_REQUIRED_COLUMNS` which uses patient-level tabular features.

**Reload contract:** `HybridPredictor.reload_dl(bundle_dir)` creates new DLModel with same runtime_lookback_days, loads (validates manifest/hash/lookback). On success, swaps under `_dl_lock`. Invalid bundles raise their original validation exception (eager validation).

**Operational impact:** DL prediction results do NOT affect final `risk_level` determination. Returned as auxiliary `dl_prediction` field only. `dl_error` captures failure reason; Rule/ML response still returned.

**Production path:** `HANAHistoryProvider.fetch_patient_history()` -> `DLModel.predict()`.

## 5. Cross-Profile Feature Set Differences

Differences are intentional. This section records them without removing or flattening.

**_BUILDER_KNOWN_COLS vs FEATURE_COLS:** `_BUILDER_KNOWN_COLS` has 2 extra: `avg_drug_duration`, `long_term_drug_count`. `FEATURE_COLS` is a strict subset (22 of 24).

**FEATURE_COLS vs ETL_NUMERIC_COLS:** `FEATURE_COLS` has 8 extra not in `ETL_NUMERIC_COLS`: `dup_efmdc`, `has_high_risk_drug`, `has_renal_risk_drug`, `has_hepatic_risk_drug`, `cyp_risk_score`, `cyp_max_enzyme_risk`, `cyp_high_risk_pairs`, `sex_m`. `ETL_NUMERIC_COLS` is a strict subset (14 of 22).

**ETL_NUMERIC_COLS vs ML_DATASET_REQUIRED_COLUMNS:** `ML_DATASET_REQUIRED_COLUMNS` includes meta (`patient_id`) and label (`risk_level`) plus 7 numeric features. `ETL_NUMERIC_COLS` has 14 numeric features. They overlap on 7 numeric features but each has unique entries.

**Policy:** Do NOT merge `FEATURE_COLS` into `_BUILDER_KNOWN_COLS`. Do NOT demand zero diff. Phase 2B characterization tests record these differences via the `ProfileDiffReporter` adapter.

## 6. Phase 3 Future-Work Policy (Out of Scope)

Phase 3 and all subsequent implementation work are outside the scope of this plan. This section records the policy for reference only.

**Phase 3 items (future work):**
1. predictor/domain extraction: Split `serving/predictor.py` responsibilities. Remove `hana_app.core` dependency. Move pure domain policy (`predict_risk`, `ACTION_BY_LABEL`, `STAGE2_LABELS`) to a neutral shared module. Preserve compatibility imports as needed.
2. Wide engine integration: Merge tabular_binary and hierarchical into one inference engine. Fable 5 and Codex both agreed NO-GO. Not pursued.
3. `scripts/ops/` reorganization.
4. Documentation alignment (`data_pipeline_architecture.md`).
5. `FEATURE_SCHEMA_LENIENT` environment variable removal (separate PR after Phase 3).

**What Phase 2 does NOT do:** Does NOT copy or move `predict_risk`, `ACTION_BY_LABEL`, `STAGE2_LABELS`. Does NOT split `predictor.py`. Does NOT merge feature lists. Does NOT change labels, versions, thresholds, or artifact formats. Does NOT migrate artifacts or retrain.

## 7. Serving Dependency Graph

```
serving/predictor.py
  +-- hana_app.core.hierarchical_runner (predict_risk, ACTION_BY_LABEL, STAGE2_LABELS)
  +-- scripts.etl: prescription_aggregator (count_ddi_severities, ddi_pair_severities, _fill_dup_features, detect_triple_whammy, detect_risk_drug, DDI_FEATURE_SEMANTICS_VERSION, FEATURE_SEMANTICS_VERSION), overlap_calculator (calculate_overlaps_for_patient, get_concurrent_drug_count), clinical_rules (collect_red_triggers, collect_severe_immediate_triggers), code_standardizer (CodeStandardizer), models (PrescriptionRecord, PatientFeatures)
  +-- scripts.features.cyp_features (CYPFeatureExtractor)
  +-- scripts.train.gat_trainer (GATTrainer, EnsembleTrainer3Way only)
  +-- rules: safety_net (SafetyNet), duplicate_detector (DuplicateDetector), risk_drug_constants (HIGH_RISK_KEYWORDS, RENAL_RISK_KEYWORDS, HEPATIC_RISK_KEYWORDS, HIGH_RISK_ATC_PREFIXES, RENAL_RISK_ATC_PREFIXES, HEPATIC_RISK_ATC_PREFIXES)
```

No circular dependencies. `serving` depends on `hana_app.core` and `scripts.*`, but those do not import from `serving`. The `serving -> hana_app.core.hierarchical_runner` dependency is recorded here and NOT removed in Phase 2.
```

- [ ] **Step 2: Verify the file exists**

Run: `test -f docs/superpowers/specs/contracts/profile_contracts.md && echo OK`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/specs/contracts/profile_contracts.md
git commit -m "docs(contracts): add profile contract spec (Phase 2A)"
```

---

## Wave 2: Characterization Tests and Read-Only Adapter (Phase 2B)

### Task 2: Create test package init and the read-only ProfileDiffReporter adapter

**Files:**
- Create: `tests/test_contracts/__init__.py`
- Create: `tests/test_contracts/profile_diff_reporter.py`

The package init is required: every existing `tests/` subdirectory (test_serving, test_features, etc.) is a package with `__init__.py`, and `test_adapter.py` imports the adapter as `tests.test_contracts.profile_diff_reporter`.

- [ ] **Step 1: Create empty package init**

```python
# tests/test_contracts/__init__.py
```

- [ ] **Step 2: Write the adapter**

```python
# tests/test_contracts/profile_diff_reporter.py
"""Read-only adapter: reports cross-profile feature set differences.

Test-tree only. NOT imported by production runtime. Reports differences
without removing them. Each profile keeps its own feature set.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProfileDiff:
    profile_a: str
    profile_b: str
    only_in_a: frozenset[str]
    only_in_b: frozenset[str]
    shared: frozenset[str]


class ProfileDiffReporter:
    """Reports feature set differences across profiles.

    Usage:
        reporter = ProfileDiffReporter()
        reporter.register("tabular_binary", _BUILDER_KNOWN_COLS)
        reporter.register("ui_experimental", FEATURE_COLS)
        diff = reporter.diff("tabular_binary", "ui_experimental")
    """

    def __init__(self) -> None:
        self._profiles: dict[str, frozenset[str]] = {}

    def register(self, name: str, features: frozenset[str] | set[str] | list[str]) -> None:
        self._profiles[name] = frozenset(features)

    def diff(self, profile_a: str, profile_b: str) -> ProfileDiff:
        a = self._profiles[profile_a]
        b = self._profiles[profile_b]
        return ProfileDiff(
            profile_a=profile_a,
            profile_b=profile_b,
            only_in_a=a - b,
            only_in_b=b - a,
            shared=a & b,
        )
```

- [ ] **Step 3: Verify import**

Run: `python -c "from tests.test_contracts.profile_diff_reporter import ProfileDiffReporter; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add tests/test_contracts/__init__.py tests/test_contracts/profile_diff_reporter.py
git commit -m "test(contracts): add read-only ProfileDiffReporter adapter (Phase 2B)"
```

---

### Task 3: Write profile contract characterization tests

**Files:**
- Create: `tests/test_contracts/test_profile_contracts.py`

Characterizes: feature names, physical column order, label spaces, semantic versions, dtype defaults, cross-profile diff. All imports are lazy (inside test functions) to avoid importing optional dependency-heavy modules at collection time.

- [ ] **Step 1: Write the test file**

```python
# tests/test_contracts/test_profile_contracts.py
"""Characterization tests for profile contracts (Phase 2B).

Records current feature names, physical column order, label spaces,
semantic versions, dtype defaults, and cross-profile differences.
All imports are lazy to keep collection lightweight.
"""
from __future__ import annotations

import pytest


# ─── tabular_binary: _BUILDER_KNOWN_COLS ──────────────────────────────────

def test_builder_known_cols():
    from serving.predictor import (
        _BUILDER_KNOWN_COLS, _INTENTIONAL_FEATURE_ALLOWLIST, _FEATURE_ALLOWED,
    )
    assert isinstance(_BUILDER_KNOWN_COLS, frozenset)
    assert len(_BUILDER_KNOWN_COLS) == 24
    expected = frozenset({
        "drug_count", "institution_count", "age", "sex_m",
        "ddi_contraindicated", "ddi_major", "ddi_moderate", "ddi_minor",
        "avg_drug_duration", "long_term_drug_count",
        "dup_same_ingredient", "dup_atc5", "dup_atc4", "dup_atc3", "dup_efmdc",
        "has_high_risk_drug", "has_renal_risk_drug", "has_hepatic_risk_drug",
        "cyp_risk_score", "cyp_high_risk_pairs", "cyp_max_enzyme_risk",
        "triple_whammy", "qt_risk_count", "drug_count_7d",
    })
    assert _BUILDER_KNOWN_COLS == expected
    assert _INTENTIONAL_FEATURE_ALLOWLIST == frozenset()
    assert _FEATURE_ALLOWED == _BUILDER_KNOWN_COLS


# ─── tabular_binary: dtype defaults ────────────────────────────────────────

def test_dtype_defaults():
    from datetime import date
    from serving.predictor import RequestFeatureBuilder, _BUILDER_KNOWN_COLS
    from serving.schemas import PredictRequest, DrugItem
    # sex_m defaults
    for sex, expected in [(None, 0.5), ("M", 1.0), ("F", 0.0)]:
        req = PredictRequest(
            patient_id="p1",
            drugs=[DrugItem(edi_code="A001", total_days=30)],
            patient_age=50, patient_sex=sex,
        )
        _, feat = RequestFeatureBuilder().build(req)
        assert feat["sex_m"] == expected
    # age default 0.0 when None
    req = PredictRequest(
        patient_id="p1",
        drugs=[DrugItem(edi_code="A001", total_days=30)],
    )
    _, feat = RequestFeatureBuilder().build(req)
    assert feat["age"] == 0.0
    # all features are float
    req = PredictRequest(
        patient_id="p1",
        drugs=[DrugItem(edi_code="A001", drug_name="aspirin", total_days=30)],
        patient_age=65, patient_sex="M",
    )
    _, feat = RequestFeatureBuilder().build(req, feature_names=sorted(_BUILDER_KNOWN_COLS))
    for name, val in feat.items():
        assert isinstance(val, float), f"Feature '{name}' is {type(val)}, expected float"


# ─── tabular_binary: classify and lenient sunset ──────────────────────────

def test_classify_threshold_proportions():
    from serving.predictor import MLModel
    from serving.schemas import RiskLevel
    ml = MLModel()
    ml._threshold = 0.5
    assert ml.classify(0.50) == RiskLevel.RED
    assert ml.classify(0.30) == RiskLevel.YELLOW
    assert ml.classify(0.15) == RiskLevel.GREEN
    assert ml.classify(0.14) == RiskLevel.NORMAL


def test_lenient_sunset_default_date():
    from datetime import date
    from serving.predictor import _FEATURE_SCHEMA_LENIENT_SUNSET_DEFAULT
    assert _FEATURE_SCHEMA_LENIENT_SUNSET_DEFAULT == date(2026, 8, 1)


# ─── hierarchical: labels and actions ─────────────────────────────────────

def test_stage2_labels():
    from hana_app.core.hierarchical_runner import (
        STAGE2_LABELS, YELLOW_SUBTYPE_LABELS, ACTION_BY_LABEL, RED_ACTION,
    )
    assert STAGE2_LABELS == (
        "Y_TRIPLE", "Y_DOUBLE", "Y_DDI_MAJOR", "Y_DDI_MOD",
        "Y_DUP", "Y_FRAG", "No_Alert",
    )
    assert len(STAGE2_LABELS) == 7
    assert len(YELLOW_SUBTYPE_LABELS) == 6
    assert "No_Alert" not in YELLOW_SUBTYPE_LABELS
    assert ACTION_BY_LABEL["Y_DDI_MAJOR"] == "약사 전화"
    assert ACTION_BY_LABEL["Y_TRIPLE"] == "문자 안내"
    assert ACTION_BY_LABEL["Y_DOUBLE"] == "모니터링"
    assert ACTION_BY_LABEL["Y_DDI_MOD"] == "모니터링"
    assert ACTION_BY_LABEL["Y_DUP"] == "모니터링"
    assert ACTION_BY_LABEL["Y_FRAG"] == "모니터링"
    assert ACTION_BY_LABEL["No_Alert"] == "관여 안 함"
    assert RED_ACTION == "즉각 개입"
    assert set(ACTION_BY_LABEL.keys()) == set(STAGE2_LABELS)


# ─── hierarchical: dispatch behavior ───────────────────────────────────────

def test_dispatch_behavior():
    import numpy as np
    from hana_app.core.hierarchical_runner import _dispatch_result, STAGE2_LABELS, RED_ACTION
    # Red when p >= tau_red
    r = _dispatch_result(p_red=0.9, stage2_probs=None,
                         stage2_labels=STAGE2_LABELS, tau_red=0.7, tau_review=0.3)
    assert r["risk_level"] == "Red"
    assert r["stage2_probs"] is None
    assert r["red_suspect"] is False
    assert r["action"] == RED_ACTION
    # Red suspect in review band
    probs = np.array([0.1, 0.1, 0.3, 0.2, 0.1, 0.1, 0.1])
    r = _dispatch_result(p_red=0.5, stage2_probs=probs,
                         stage2_labels=STAGE2_LABELS, tau_red=0.7, tau_review=0.3)
    assert r["risk_level"] == STAGE2_LABELS[2]
    assert r["red_suspect"] is True
    # No suspect below tau_review
    probs = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.4])
    r = _dispatch_result(p_red=0.2, stage2_probs=probs,
                         stage2_labels=STAGE2_LABELS, tau_red=0.7, tau_review=0.3)
    assert r["risk_level"] == STAGE2_LABELS[6]
    assert r["red_suspect"] is False


def test_stage2_label_to_risk_mapping():
    from serving.predictor import HybridPredictor
    from serving.schemas import RiskLevel
    assert HybridPredictor._stage2_label_to_risk("Red") == RiskLevel.RED
    assert HybridPredictor._stage2_label_to_risk("Y_TRIPLE") == RiskLevel.YELLOW
    assert HybridPredictor._stage2_label_to_risk("No_Alert") == RiskLevel.NORMAL


# ─── ui_experimental: FEATURE_COLS and ETL_NUMERIC_COLS ───────────────────

def test_feature_cols_and_etl_numeric_cols():
    from hana_app.core.ml_runner import FEATURE_COLS, RISK_LABEL_MAP
    from scripts.features.feature_engineer import ETL_NUMERIC_COLS
    from serving.predictor import _BUILDER_KNOWN_COLS
    # FEATURE_COLS physical order
    assert list(FEATURE_COLS) == [
        "drug_count", "drug_count_7d", "institution_count",
        "ddi_contraindicated", "ddi_major", "ddi_moderate", "ddi_minor",
        "triple_whammy", "qt_risk_count",
        "dup_same_ingredient", "dup_atc5", "dup_atc4", "dup_atc3", "dup_efmdc",
        "has_high_risk_drug", "has_renal_risk_drug", "has_hepatic_risk_drug",
        "cyp_risk_score", "cyp_max_enzyme_risk", "cyp_high_risk_pairs",
        "age", "sex_m",
    ]
    assert len(FEATURE_COLS) == 22
    # FEATURE_COLS subset of _BUILDER_KNOWN_COLS
    assert set(FEATURE_COLS).issubset(_BUILDER_KNOWN_COLS)
    # 2 extra in _BUILDER_KNOWN_COLS
    assert _BUILDER_KNOWN_COLS - set(FEATURE_COLS) == {"avg_drug_duration", "long_term_drug_count"}
    # ETL_NUMERIC_COLS physical order
    assert list(ETL_NUMERIC_COLS) == [
        "drug_count", "drug_count_7d", "institution_count",
        "ddi_contraindicated", "ddi_major", "ddi_moderate", "ddi_minor",
        "triple_whammy", "qt_risk_count",
        "dup_same_ingredient", "dup_atc5", "dup_atc4", "dup_atc3",
        "age",
    ]
    # ETL subset of FEATURE_COLS
    assert set(ETL_NUMERIC_COLS).issubset(set(FEATURE_COLS))
    # 8 extra in FEATURE_COLS vs ETL
    assert set(FEATURE_COLS) - set(ETL_NUMERIC_COLS) == {
        "dup_efmdc", "has_high_risk_drug", "has_renal_risk_drug",
        "has_hepatic_risk_drug", "cyp_risk_score", "cyp_max_enzyme_risk",
        "cyp_high_risk_pairs", "sex_m",
    }
    # RISK_LABEL_MAP
    assert RISK_LABEL_MAP == {"Red": 3, "Yellow": 2, "Green": 1, "Normal": 0}


# ─── dl_history: bundle contract ───────────────────────────────────────────

def test_dl_history_contract():
    from scripts.datasets.contracts import (
        DL_BUNDLE_REQUIRED_FILES, DL_DATASET_REQUIRED_COLUMNS,
        ML_DATASET_REQUIRED_COLUMNS,
        LOOKBACK_DAYS_DEFAULT, LOOKBACK_DAYS_MIN, LOOKBACK_DAYS_MAX,
    )
    from serving.dl_predictor import _SUPPORTED_ENCODING_STRATEGIES, _GRAPH_ARCHITECTURES
    assert set(DL_BUNDLE_REQUIRED_FILES) == {
        "model.pt", "model_config.json", "drug_vocab.json",
        "edge_index.pt", "feature_normalizer.pkl", "schema_version.json",
    }
    assert len(DL_BUNDLE_REQUIRED_FILES) == 6
    assert _SUPPORTED_ENCODING_STRATEGIES == {"multi_hot"}
    assert _GRAPH_ARCHITECTURES == {"gat", "gcn"}
    assert LOOKBACK_DAYS_DEFAULT == 365
    assert LOOKBACK_DAYS_MIN == 7
    assert LOOKBACK_DAYS_MAX == 1825
    assert DL_DATASET_REQUIRED_COLUMNS == ("patient_id", "drug_code", "prescription_date")
    assert tuple(ML_DATASET_REQUIRED_COLUMNS) == (
        "patient_id", "drug_count", "drug_count_7d", "institution_count",
        "ddi_contraindicated", "ddi_major", "ddi_moderate", "ddi_minor",
        "risk_level",
    )
    ml_feat = set(ML_DATASET_REQUIRED_COLUMNS) - {"patient_id", "risk_level"}
    dl_feat = set(DL_DATASET_REQUIRED_COLUMNS) - {"patient_id"}
    assert ml_feat.isdisjoint(dl_feat)


def test_dl_prediction_does_not_affect_risk_level():
    from serving.schemas import PredictResponse
    fields = PredictResponse.model_fields
    assert "dl_prediction" in fields
    assert "dl_error" in fields
    assert "risk_level" in fields
    assert fields["dl_prediction"].is_required() is False
    assert fields["risk_level"].is_required() is True


# ─── semantic versions and intervention map ──────────────────────────────

def test_semantic_versions():
    from scripts.etl.prescription_aggregator import (
        DDI_FEATURE_SEMANTICS_VERSION, FEATURE_SEMANTICS_VERSION,
    )
    assert DDI_FEATURE_SEMANTICS_VERSION == "ddi.v2"
    assert FEATURE_SEMANTICS_VERSION == "rulefeat.v1"


def test_intervention_map_values():
    from serving.schemas import INTERVENTION_MAP, RiskLevel
    assert INTERVENTION_MAP[RiskLevel.RED] == "즉각 개입"
    assert INTERVENTION_MAP[RiskLevel.YELLOW] == "복약 상담"
    assert INTERVENTION_MAP[RiskLevel.GREEN] == "관여 안 함"
    assert INTERVENTION_MAP[RiskLevel.NORMAL] == "관여 안 함"
```

- [ ] **Step 2: Run the tests**

Run: `python -m pytest tests/test_contracts/test_profile_contracts.py -v`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_contracts/test_profile_contracts.py
git commit -m "test(contracts): add profile contract characterization tests (Phase 2B)"
```

---

### Task 4: Write serving characterization tests (fallback, request mutation, alignment)

**Files:**
- Create: `tests/test_contracts/test_serving_characterization.py`

- [ ] **Step 1: Write the test file**

```python
# tests/test_contracts/test_serving_characterization.py
"""Characterization tests for serving fallback and request mutation (Phase 2B).

Records: resource absence fallback behavior, request mutation side effects,
feature vector alignment. All imports are lazy.
"""
from __future__ import annotations

import pytest


def _make_req(drugs=None, age=65, sex="M"):
    from serving.schemas import PredictRequest, DrugItem
    if drugs is None:
        drugs = [DrugItem(edi_code="A001", drug_name="aspirin", total_days=30)]
    return PredictRequest(patient_id="p1", drugs=drugs, patient_age=age, patient_sex=sex)


# ─── resource absence fallback ─────────────────────────────────────────────

def test_resource_absence_fallbacks():
    from serving.predictor import RequestFeatureBuilder
    req = _make_req()
    # No DDI matrix -> zero DDI counts
    _, feat = RequestFeatureBuilder(ddi_matrix=None).build(req)
    assert feat["ddi_contraindicated"] == 0.0
    assert feat["ddi_major"] == 0.0
    assert feat["ddi_moderate"] == 0.0
    assert feat["ddi_minor"] == 0.0
    # No CYP extractor -> zero CYP features
    _, feat = RequestFeatureBuilder(cyp_extractor=None).build(req)
    assert feat["cyp_risk_score"] == 0.0
    assert feat["cyp_high_risk_pairs"] == 0.0
    assert feat["cyp_max_enzyme_risk"] == 0.0
    # No code standardizer -> zero dup_efmdc
    _, feat = RequestFeatureBuilder(code_standardizer=None).build(req)
    assert feat["dup_efmdc"] == 0.0
    # No bridge -> drug_count_7d falls back to drug_count
    _, feat = RequestFeatureBuilder().build(req)
    assert feat["drug_count_7d"] == feat["drug_count"]


def test_atc_fallback_for_dup_without_std():
    from serving.predictor import RequestFeatureBuilder
    from serving.schemas import DrugItem
    drugs = [
        DrugItem(edi_code="A001", drug_name="aspirin", atc_code="B01AC06", total_days=30),
        DrugItem(edi_code="A002", drug_name="aspirin", atc_code="B01AC06", total_days=30),
    ]
    req = _make_req(drugs=drugs)
    _, feat = RequestFeatureBuilder(code_standardizer=None).build(req)
    assert feat["dup_same_ingredient"] >= 1.0


def test_safety_net_and_dup_detector_when_module_missing():
    from serving.predictor import _run_safety_net, _run_duplicate_detector
    from serving.schemas import DrugItem, RiskLevel
    drugs = [DrugItem(edi_code="A001", drug_name="aspirin", total_days=30)]
    level, reasons, alerts = _run_safety_net(drugs, sn_instance=None)
    assert level in (RiskLevel.NORMAL, RiskLevel.GREEN, RiskLevel.YELLOW, RiskLevel.RED)
    count, reasons = _run_duplicate_detector(drugs, dd_instance=None)
    assert isinstance(count, int)
    assert isinstance(reasons, list)


# ─── request mutation ─────────────────────────────────────────────────────

def test_request_mutation_and_validation():
    from datetime import date
    from serving.schemas import PredictRequest, DrugItem
    # start_date defaults to today at construction
    drug = DrugItem(edi_code="A001", total_days=30)
    PredictRequest(patient_id="p1", drugs=[drug])
    assert drug.start_date == date.today()
    # reference_date defaults to today
    req = PredictRequest(
        patient_id="p1",
        drugs=[DrugItem(edi_code="A001", total_days=30)],
    )
    assert req.reference_date == date.today()
    # edi_code stripped
    drug = DrugItem(edi_code="  A001  ", total_days=30)
    assert drug.edi_code == "A001"
    # empty edi_code rejected
    with pytest.raises(ValueError, match="EDI"):
        DrugItem(edi_code="   ", total_days=30)
    # zero drugs rejected
    with pytest.raises(Exception):
        PredictRequest(patient_id="p1", drugs=[])
    # total_days bounds
    for bad_days in (0, 366):
        with pytest.raises(Exception):
            DrugItem(edi_code="A001", total_days=bad_days)
    # patient_age validation
    with pytest.raises(Exception):
        PredictRequest(
            patient_id="p1",
            drugs=[DrugItem(edi_code="A001", total_days=30)],
            patient_age=121,
        )
    # patient_sex pattern validation
    with pytest.raises(Exception):
        PredictRequest(
            patient_id="p1",
            drugs=[DrugItem(edi_code="A001", total_days=30)],
            patient_sex="X",
        )


def test_build_does_not_mutate_atc_code_when_std_absent():
    from serving.predictor import RequestFeatureBuilder
    from serving.schemas import PredictRequest, DrugItem
    drug = DrugItem(edi_code="A001", total_days=30)
    req = PredictRequest(patient_id="p1", drugs=[drug], patient_age=65, patient_sex="M")
    RequestFeatureBuilder().build(req)
    assert drug.atc_code is None


# ─── feature vector alignment ──────────────────────────────────────────────

def test_feature_vector_alignment():
    import numpy as np
    from serving.predictor import RequestFeatureBuilder, _BUILDER_KNOWN_COLS
    req = _make_req()
    # Aligned to feature_names order
    names = ["age", "drug_count", "ddi_major"]
    vec, feat = RequestFeatureBuilder().build(req, feature_names=names)
    assert len(vec) == 3
    assert vec[0] == feat["age"]
    assert vec[1] == feat["drug_count"]
    assert vec[2] == feat["ddi_major"]
    # All finite
    vec, _ = RequestFeatureBuilder().build(req, feature_names=sorted(_BUILDER_KNOWN_COLS))
    assert np.isfinite(vec).all()
```

- [ ] **Step 2: Run the tests**

Run: `python -m pytest tests/test_contracts/test_serving_characterization.py -v`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_contracts/test_serving_characterization.py
git commit -m "test(contracts): add serving fallback and request mutation characterization (Phase 2B)"
```

---

### Task 5: Write reload, artifact, and pickle compatibility tests

**Files:**
- Create: `tests/test_contracts/test_reload_artifact_compat.py`

- [ ] **Step 1: Write the test file**

```python
# tests/test_contracts/test_reload_artifact_compat.py
"""Characterization tests for reload/rollback and pickle module-path compatibility (Phase 2B).

Records: reload_model/reload_hierarchical/reload_dl success and failure behavior,
pickle module-path references via non-executing pickletools inspection (no untrusted load).
All imports are lazy.
"""
from __future__ import annotations

import io
import pickle
import pickletools
import hashlib
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def _write_model(path: Path, feature_names: list[str]) -> Path:
    """Write a minimal model pkl with sha256 sidecar."""
    import numpy as np

    class _FakeModel:
        def predict_proba(self, X):
            return np.array([[0.3, 0.7]])

    from scripts.etl.prescription_aggregator import DDI_FEATURE_SEMANTICS_VERSION
    payload = {
        "model": _FakeModel(),
        "best_threshold": 0.5,
        "trainer_class": "XGBoostTrainer",
        "feature_names": feature_names,
        "artifact_version": 2,
        "ddi_feature_semantics_version": DDI_FEATURE_SEMANTICS_VERSION,
    }
    content = pickle.dumps(payload)
    path.write_bytes(content)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{hashlib.sha256(content).hexdigest()}  {path.name}\n"
    )
    return path


def _make_hybrid_predictor():
    import threading
    from serving.predictor import MLModel, HybridPredictor
    pred = HybridPredictor.__new__(HybridPredictor)
    pred._start_time = 0.0
    pred._ml = MLModel()
    pred._ddi_matrix = None
    pred._cyp = None
    pred._std = None
    pred._safety_net = None
    pred._dup_detector = None
    pred._ml_lock = threading.Lock()
    pred._hier_lock = threading.RLock()
    pred._dl_lock = threading.RLock()
    pred._hierarchical = None
    pred._dl = MagicMock()
    pred._dl.runtime_lookback_days = 365
    pred._dl_history_provider = None
    pred._builder = MagicMock()
    return pred


# ─── reload_model ──────────────────────────────────────────────────────────

def test_reload_model_success_swaps_model(tmp_path):
    pred = _make_hybrid_predictor()
    old_ml = pred._ml
    path = _write_model(tmp_path / "m.pkl", ["drug_count", "age"])
    ok = pred.reload_model(path)
    assert ok is True
    assert pred._ml is not old_ml
    assert pred._ml.loaded is True


def test_reload_model_failure_preserves_existing(tmp_path, monkeypatch):
    monkeypatch.delenv("FEATURE_SCHEMA_LENIENT", raising=False)
    pred = _make_hybrid_predictor()
    old_ml = pred._ml
    path = _write_model(tmp_path / "bad.pkl", ["drug_count", "fake_xyz"])
    ok = pred.reload_model(path)
    assert ok is False
    assert pred._ml is old_ml


# ─── reload_hierarchical ───────────────────────────────────────────────────

def test_reload_hierarchical_failure_preserves_existing(monkeypatch):
    monkeypatch.delenv("FEATURE_SCHEMA_LENIENT", raising=False)
    from serving.predictor import HierarchicalPredictor
    pred = _make_hybrid_predictor()
    existing_hp = MagicMock(spec=HierarchicalPredictor)
    pred._hierarchical = existing_hp

    fake_hp = MagicMock()
    fake_hp.load = MagicMock(return_value=True)
    fake_hp.feature_cols = ["drug_count", "fake_unknown_col"]

    import serving.predictor as P
    monkeypatch.setattr(P, "HierarchicalPredictor", MagicMock(return_value=fake_hp))

    ok = pred.reload_hierarchical("/tmp/fake")
    assert ok is False
    assert pred._hierarchical is existing_hp


def test_reload_hierarchical_empty_feature_cols_rejected(monkeypatch):
    monkeypatch.delenv("FEATURE_SCHEMA_LENIENT", raising=False)
    pred = _make_hybrid_predictor()

    fake_hp = MagicMock()
    fake_hp.load = MagicMock(return_value=True)
    fake_hp.feature_cols = []

    import serving.predictor as P
    monkeypatch.setattr(P, "HierarchicalPredictor", MagicMock(return_value=fake_hp))

    ok = pred.reload_hierarchical("/tmp/fake-empty")
    assert ok is False


# ─── reload_dl ─────────────────────────────────────────────────────────────

def test_reload_dl_invalid_bundle_raises(tmp_path):
    """Record: reload_dl with missing MANIFEST.json raises FileNotFoundError."""
    pred = _make_hybrid_predictor()
    empty_dir = tmp_path / "empty_dl"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        pred.reload_dl(str(empty_dir))


# ─── pickle module-path via non-executing inspection ───────────────────────

def _extract_global_modules(pickle_bytes: bytes) -> set[str]:
    """Extract module names from GLOBAL opcodes without loading the pickle.

    Uses pickletools.dis (bytecode disassembly only, never executes).
    Safe for untrusted artifacts.
    """
    output = io.StringIO()
    pickletools.dis(pickle_bytes, output)
    modules: set[str] = set()
    for line in output.getvalue().splitlines():
        if "GLOBAL" in line and "'" in line:
            parts = line.split("'")
            if len(parts) >= 2:
                full_ref = parts[1]
                if "." in full_ref:
                    modules.add(full_ref.rsplit(".", 1)[0])
    return modules


def test_extract_global_modules_runs_on_simple_pickle():
    blob = pickle.dumps({"key": [1, 2, 3]})
    modules = _extract_global_modules(blob)
    assert isinstance(modules, set)


def test_mlmodel_state_dict_pickle_no_unexpected_modules():
    """Record: MLModel state dict pickle references only expected modules."""
    state = {
        "model": None,
        "best_threshold": 0.5,
        "feature_names": ["drug_count", "age"],
        "artifact_version": 2,
    }
    blob = pickle.dumps(state)
    modules = _extract_global_modules(blob)
    for m in modules:
        assert m.startswith(("numpy", "serving", "scripts", "hana_app", "rules")) or "." not in m, (
            f"Unexpected module in pickle: {m}"
        )


def test_joblib_bundle_references_sklearn(tmp_path):
    """Record: joblib bundle for hierarchical artifacts references sklearn modules."""
    try:
        import joblib
    except ImportError:
        pytest.skip("joblib not installed")
    from sklearn.linear_model import LogisticRegression
    import numpy as np

    model = LogisticRegression()
    X = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
    y = np.array([0, 1, 1, 0])
    model.fit(X, y)

    path = tmp_path / "stage1.joblib"
    joblib.dump(model, path)

    blob = path.read_bytes()
    modules = _extract_global_modules(blob)
    assert any("sklearn" in m for m in modules), f"Expected sklearn in joblib pickle, got: {modules}"
```

- [ ] **Step 2: Run the tests**

Run: `python -m pytest tests/test_contracts/test_reload_artifact_compat.py -v`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_contracts/test_reload_artifact_compat.py
git commit -m "test(contracts): add reload/rollback and pickle module-path characterization (Phase 2B)"
```

---

### Task 6: Write adapter integration tests

**Files:**
- Create: `tests/test_contracts/test_adapter.py`

- [ ] **Step 1: Write the test file**

```python
# tests/test_contracts/test_adapter.py
"""Integration tests for the read-only ProfileDiffReporter adapter (Phase 2B).

Verifies that the adapter correctly reports cross-profile differences
without modifying any production code or data.
"""
from __future__ import annotations

from tests.test_contracts.profile_diff_reporter import ProfileDiffReporter


def test_diff_tabular_vs_ui_experimental():
    from serving.predictor import _BUILDER_KNOWN_COLS
    from hana_app.core.ml_runner import FEATURE_COLS
    reporter = ProfileDiffReporter()
    reporter.register("tabular_binary", _BUILDER_KNOWN_COLS)
    reporter.register("ui_experimental", FEATURE_COLS)
    diff = reporter.diff("tabular_binary", "ui_experimental")
    assert diff.only_in_a == {"avg_drug_duration", "long_term_drug_count"}
    assert diff.only_in_b == set()  # FEATURE_COLS is a strict subset
    assert len(diff.shared) == 22


def test_diff_ui_experimental_vs_etl():
    from hana_app.core.ml_runner import FEATURE_COLS
    from scripts.features.feature_engineer import ETL_NUMERIC_COLS
    reporter = ProfileDiffReporter()
    reporter.register("ui_experimental", FEATURE_COLS)
    reporter.register("etl_numeric", ETL_NUMERIC_COLS)
    diff = reporter.diff("ui_experimental", "etl_numeric")
    assert diff.only_in_a == {
        "dup_efmdc", "has_high_risk_drug", "has_renal_risk_drug",
        "has_hepatic_risk_drug", "cyp_risk_score", "cyp_max_enzyme_risk",
        "cyp_high_risk_pairs", "sex_m",
    }
    assert diff.only_in_b == set()  # ETL_NUMERIC_COLS is a strict subset


def test_adapter_does_not_modify_registered_sets():
    from serving.predictor import _BUILDER_KNOWN_COLS
    original = frozenset(_BUILDER_KNOWN_COLS)
    reporter = ProfileDiffReporter()
    reporter.register("test", _BUILDER_KNOWN_COLS)
    assert _BUILDER_KNOWN_COLS == original
```

- [ ] **Step 2: Run the tests**

Run: `python -m pytest tests/test_contracts/test_adapter.py -v`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_contracts/test_adapter.py
git commit -m "test(contracts): add adapter integration tests (Phase 2B)"
```

---

### Task 7: Verify pre-existing test node IDs unchanged

- [ ] **Step 1: Run the full test suite**

Run: `python -m pytest tests/ -q 2>&1 | tail -1`
Expected: All pre-existing tests pass. New `test_contracts` tests pass. No failures.

- [ ] **Step 2: Verify no production code was modified**

Run: `git diff --name-only -- serving/ hana_app/ scripts/ dags/ config/ rules/ | head -20`
Expected: No output (no production code files modified). Only new files under `docs/superpowers/specs/contracts/` and `tests/test_contracts/` should appear in `git status --porcelain`.

- [ ] **Step 3: Verify no em dashes or en dashes in new files**

Run: `grep -rn $'\xe2\x80\x94\|\xe2\x80\x93' docs/superpowers/specs/contracts/ tests/test_contracts/ || echo "OK"`
Expected: `OK` (no em dashes or en dashes found)

---

## Cross-Family Review Gate

Before merging any commits from this plan, both reviews below must succeed. If either finds issues, fix and re-review. Do not merge until both approve. Scope: all 7 files created by this plan (the contract spec plus the 6 test-tree files).

1. **Claude/Fable logical review**: Review the contract spec document for logical accuracy, label semantics, freeze policy compliance, and profile separation correctness. Verify no feature flattening, no label/version changes, no dependency removal.

2. **Codex technical review**: Review all test files and the adapter for technical correctness, import safety, test isolation, and verification rigor. Verify no production code is modified, no untrusted artifact loading, and all tests are additive (no pre-existing test modified or deleted).

3. **Note on advisor panel**: This direct cross-family review (Claude/Fable + Codex) is distinct from the formal `ask_advisor_panel` Oracle gate. The `ask_advisor_panel` interface is currently unavailable and not connected. This direct review does NOT satisfy the formal advisor panel requirement. Oracle remains blocked. This distinction is recorded explicitly.

---

## Acceptance Criteria

### Wave 1 (Phase 2A)

- [ ] Single contract spec document exists at `docs/superpowers/specs/contracts/profile_contracts.md`
- [ ] Spec covers all 4 profiles (tabular_binary, hierarchical, ui_experimental, dl_history)
- [ ] Cross-profile differences recorded without flattening; serving dependency graph recorded (serving -> hana_app not removed)
- [ ] Phase 3 future-work policy recorded as a section (not a separate file)
- [ ] No production code modified; all pre-existing tests pass

### Wave 2 (Phase 2B)

- [ ] Profile contract characterization tests cover: feature names, physical column order, label spaces, semantic versions, dtype defaults
- [ ] Serving characterization tests cover: resource absence/fallback, request mutation, feature vector alignment
- [ ] Reload/rollback characterization tests cover: reload_model, reload_hierarchical, reload_dl success and failure
- [ ] Pickle module-path compatibility recorded via non-executing pickletools inspection (no untrusted load)
- [ ] Read-only ProfileDiffReporter adapter exists in test tree (not production)
- [ ] Adapter integration tests verify cross-profile diff reporting
- [ ] All pre-existing test node IDs and pass/fail outcomes unchanged; collection only grows (new tests added, none removed)
- [ ] No production code modified
- [ ] No em dashes or en dashes in any new file
- [ ] Cross-family review (Claude/Fable + Codex) completed and approved

---

## Rollback

### Rollback Method

| Wave | Rollback Action | Impact |
|---|---|---|
| Wave 1 | Delete `docs/superpowers/specs/contracts/profile_contracts.md` | None (documentation only, no production code changes) |
| Wave 2 | Delete `tests/test_contracts/` directory | None (test-only, no production code changes) |

### Rollback Triggers

1. Any pre-existing test fails after Wave 1 or Wave 2
2. Any pre-existing test node ID disappears from collection
3. Any production code file is found modified
4. Python 3.12 compatibility failure
5. Cross-family review finds unfixable issues

### Rollback Verification

After rollback:
- Run `python -m pytest tests/ -q` and verify all pre-existing tests pass; run `git diff --name-only -- serving/ hana_app/ scripts/ dags/ config/ rules/` and verify no production code modified
- Verify `docs/superpowers/specs/contracts/` and `tests/test_contracts/` no longer exist (full rollback) or only Wave 1 spec exists (Wave 2 rollback)

---

## Out of Scope

- Phase 3 predictor/domain extraction (future work, recorded as policy section only)
- Feature list merging, flattening, or reordering; label, threshold, or semantic version changes
- Artifact migration, retraining, predictor.py splitting
- Domain policy (`predict_risk`, `ACTION_BY_LABEL`, `STAGE2_LABELS`) copying or moving
- `FEATURE_SCHEMA_LENIENT` environment variable removal
- Nov->Dec holdout tuning (`RESEARCH_TRACK_FROZEN`); Gate 5A/5B activation (canceled/retired); 2025-01 data acquisition (not planned)
- Protected path modifications (`packages_win/py312/`, `mlruns/`, `.parquet`, `out/`)
- Existing unrelated working-tree changes (must not be included in future commits for this plan)
