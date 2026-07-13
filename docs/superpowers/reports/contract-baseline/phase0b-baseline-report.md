# Phase 0B Contract Baseline Report

**Captured:** `2026-07-12T21:37:33+09:00`<br>
**Status:** Static, reproducible source baseline with known WARNs<br>
**Baseline source snapshot:** `BASELINE_SOURCE_SHA=3d8d64e78601a3ff56dc38034a9da62853e6b656`

## 1. Purpose and safety boundary

This report consolidates the four approved Phase 0A/0B source reports into one baseline for later contract work. It records source-visible contracts, feature dispersion, loader behavior, and static dependency structure. It does **not** assert the state or contents of any deployed model, bundle, sidecar, JSON metadata file, Windows production environment, HANA system, or generated dataset.

All evidence was obtained from Git, source text, Python 3.12 standard-library metadata/AST parsing, and the four approved Markdown inputs. No repository module was imported. No test suite or pytest collection was run for Task 5. No model, pickle, joblib, Torch artifact, model-side JSON, Parquet, generated output, protected path, HANA data, or frozen holdout was opened. Nothing under `packages_win/py312/`, `mlruns/`, generated Parquet, or `out/` was changed. The `RESEARCH_TRACK_FROZEN` boundary remains in force: Nov→Dec/future-onset holdout use for model, feature, ablation, or hyperparameter work is prohibited; Gate 5A/5B and 2025-01 acquisition are retired.

## 2. Snapshot provenance

The four approved inputs and their SHA-256 digests are:

| Input | SHA-256 |
|---|---|
| [`phase0a-profile-contract-map.md`](./phase0a-profile-contract-map.md) | `790ebe0d49cdc410d23a72725ae4959e909e51afaa90cf539df8bb1096b3db79` |
| [`phase0a-feature-dispersion-table.md`](./phase0a-feature-dispersion-table.md) | `d348d6b734004e30c73d09452182bff4773e0b7fa526f765a7458854c96fac15` |
| [`phase0a-bundle-metadata-record.md`](./phase0a-bundle-metadata-record.md) | `50273bf53c06ab6f277261141e2de1b5df7f4cd67778e91ba61b8fb9168a0da6` |
| [`phase0b-dependency-graph.md`](./phase0b-dependency-graph.md) | `6ed500c16c697a518a0754de8aa918f450bb7eaacd527dae5f3c4a9aa85dd243` |

`BASELINE_SOURCE_SHA=3d8d64e78601a3ff56dc38034a9da62853e6b656` identifies the analyzed source snapshot. It is independent of the later commit that publishes these reports: current `HEAD` need not equal `BASELINE_SOURCE_SHA`, and the report files need not exist at that source commit. The analyzed Python paths under `serving`, `hana_app`, `scripts`, and `rules` matched the paths and blobs at `BASELINE_SOURCE_SHA` (`analyzed_source_diff_exit=0`). Selected baseline Git blobs, in command order, were:

| Source | Git blob |
|---|---|
| `serving/predictor.py` | `3903066df459885b44a48530aa6e1911801d9fb8` |
| `hana_app/core/ml_runner.py` | `29bf7afcb634ce3630ec24942bb39905434a0886` |
| `hana_app/core/hierarchical_runner.py` | `2f5b30b2558664ffbfc97a994ae1a38b7bac9201` |
| `scripts/features/feature_engineer.py` | `407222615d2d6981c47b914656271bd07cd99f73` |
| `scripts/datasets/contracts.py` | `7418e21229b7a25e2c7257aabe0ed6f356e96485` |
| `scripts/etl/prescription_aggregator.py` | `771ea7f2dbb1ff1ad23c676fd4e0a0aeb319dc4f` |
| `serving/schemas.py` | `9e2cc58d2ae801de2a0681a8af407ca35042fea7` |
| `serving/dl_predictor.py` | `76c87f97e02e2c8ae96fb3141ffa802975e409b9` |
| `serving/hana_history.py` | `9b9d025b5240a07d1983c0b35ef630db533d25ed` |

The capture ran in a WSL worktree, not Windows production, with Python `3.12.3` at `.venv/bin/python`. The executable and first five `sys.path` entries are reproduced in Appendix A.

## 3. Profile contract summary

### 3.1 `tabular_binary`

- The online builder capability contract is `_BUILDER_KNOWN_COLS`, an **unordered set of 24 names**. It is not vector order. Bundle metadata `feature_names` owns serving order, and `RequestFeatureBuilder` emits values in that order.
- The default probability threshold is `0.5`. Classification bands are Red at `t`, Yellow at `0.6t`, Green at `0.3t`, then Normal. Response intervention follows Red/Yellow/Green/Normal.
- `DDI_FEATURE_SEMANTICS_VERSION == "ddi.v2"` is a hard load guard for bundles using `ddi_*` features.
- Feature values become floats. Missing resource fallbacks and name-based alignment remain contract behavior; they must not be confused with proof that deployed resources are present.
- `reload_model()` loads into a new object and swaps it under `_ml_lock` only when `load()` returns true; a false return retains the previous model (rollback-by-retention).

### 3.2 `hierarchical`

- Ordered input features come from `stage_meta.json["feature_cols"]`; this report did not read a real `stage_meta.json`.
- The ordered Stage-2 label contract has seven labels: `Y_TRIPLE`, `Y_DOUBLE`, `Y_DDI_MAJOR`, `Y_DDI_MOD`, `Y_DUP`, `Y_FRAG`, and `No_Alert`.
- `tau_red` and `tau_review` control Stage-1 dispatch. Contraindication and subtype rule backstops can only escalate outcomes.
- `ddi.v2` is a hard load guard. `rulefeat.v1` is currently a runtime rule-feature gate, not a hard load rejection.
- `reload_hierarchical()` swaps under `_hier_lock` only after load, non-empty feature, and schema checks pass; failures retain the previous predictor. Startup prefers a valid hierarchical directory and otherwise falls back to the single model path.
- Serving directly depends on `hana_app.core.hierarchical_runner` for label validation, inference, and subtype action mapping. This is a source-level ownership boundary that later extraction work must preserve behavior across.

### 3.3 `ui_experimental`

- `FEATURE_COLS` is an **ordered 22-feature** UI-training default and is a separate path from operational serving.
- Raw HANA sex values are `"1"`/`"2"` in the UI row builder, while serving request validation accepts `M`/`F`. Both produce `sex_m`, but the input vocabularies differ and must not be equated without mapping.
- UI `risk_binary` marks Red and Yellow positive. ETL `is_high_risk` marks only Red positive. These labels are not interchangeable.
- The UI training flow does not itself define the serving hot-reload, rollback, or intervention contract.

### 3.4 `dl_history`

- Required history columns are ordered as `patient_id`, `drug_code`, `prescription_date`.
- Six declared bundle files are required: `model.pt`, `model_config.json`, `drug_vocab.json`, `edge_index.pt`, `feature_normalizer.pkl`, and `schema_version.json`.
- Lookback defaults to 365 days and is constrained to 7..1825; artifact and runtime lookbacks must match.
- The DL result is auxiliary. The primary final risk level is established before the DL prediction is attached.
- `reload_dl()` has a narrower caveat than the other reload paths: its rollback behavior relies on `DLModel.load()` continuing to raise on every failure rather than returning false.

## 4. Cross-profile differences are baseline facts

The four source lists have different roles and container semantics. They are not a zero-diff target and must not be normalized as part of this baseline.

| Key | Contract | Count |
|---|---|---:|
| B | Online builder capability set | 24 |
| F | UI `FEATURE_COLS` ordered list | 22 |
| E | ETL numeric inventory | 14 |
| D | Minimum ML dataset required tuple | 9 |

Exact left-only differences:

| Comparison | Names |
|---|---|
| B ∖ F | `avg_drug_duration`, `long_term_drug_count` |
| F ∖ B | none |
| B ∖ E | `avg_drug_duration`, `cyp_high_risk_pairs`, `cyp_max_enzyme_risk`, `cyp_risk_score`, `dup_efmdc`, `has_hepatic_risk_drug`, `has_high_risk_drug`, `has_renal_risk_drug`, `long_term_drug_count`, `sex_m` |
| E ∖ B | none |
| F ∖ E | `cyp_high_risk_pairs`, `cyp_max_enzyme_risk`, `cyp_risk_score`, `dup_efmdc`, `has_hepatic_risk_drug`, `has_high_risk_drug`, `has_renal_risk_drug`, `sex_m` |
| E ∖ F | none |
| D ∖ E | `patient_id`, `risk_level` |
| E ∖ D | `age`, `dup_atc3`, `dup_atc4`, `dup_atc5`, `dup_same_ingredient`, `qt_risk_count`, `triple_whammy` |

Equal sets would still not prove positional parity. B is unordered, F is an ordered UI default, E is an ordered inventory embedded in a wider DataFrame pipeline, and D's validator checks presence rather than physical order. Serving order comes from bundle metadata.

## 5. `FEATURE_SCHEMA_LENIENT` state

At capture time, both `FEATURE_SCHEMA_LENIENT` and `FEATURE_SCHEMA_LENIENT_SUNSET_DATE` were unset (`None`). The observation date, 2026-07-12, was before the code-default sunset of 2026-08-01, but lenient mode was **inactive** because the enable variable was unset. Accepted enable values are `1`, `true`, and `yes`, case-insensitive after stripping, and remain subject to the sunset. Invalid sunset overrides fail closed.

This is evidence about the Task 5 WSL process only, not the Windows production environment. Before sunset, active lenient mode permits unknown bundle features to fall back to `0.0` while recording drift; at or after sunset, strict rejection applies even if the enable variable is set. Removing lenient mode is a separate post-Phase-3 change.

## 6. Bundle and deserialization risks

The approved bundle report inspected source-visible loader contracts only. **Artifact bytes and model-side JSON were not inspected**, so no deployed feature order, threshold, label list, schema version, run ID, lookback, estimator version, hash, dtype, or pickle global is established here.

- `_ConstantNegativeStage1` is a confirmed project-local compatibility path for degraded hierarchical Stage-1 joblibs. Its module/class path must remain importable or receive a compatibility symbol when refactored.
- The stale `_EnsembleWrapper` assumption is corrected: current source creates it as a local runtime adapter after sidecars are unpickled. It is not proven to be serialized in current bundles.
- External XGBoost, LightGBM, scikit-learn, PyTorch, and PyTorch Geometric estimator/module/version compatibility remains a high deployment risk, especially in closed-network Windows production.
- Adjacent SHA-256 checks establish integrity relative to the recorded digest, not safe deserialization or trusted provenance. Pickle/joblib and the GAT graph's `torch.load(..., weights_only=False)` can execute reconstruction/import behavior.
- `DDI_DL_BUNDLE_DIR` is configured, but current startup source does not wire it into automatic DL loading. It must not be documented as active startup behavior.

Only trusted, provenance-controlled artifacts may reach executing loaders. Any byte-level or model-JSON inspection needs a separately approved, bounded procedure.

## 7. Dependency and cycle baseline

`serving.predictor` has 14 direct local-module edges in the static inventory, including eager serving/schema/risk-constant edges, lazy ETL/rule/feature edges, an optional training edge to `scripts.train.gat_trainer`, and the explicit `serving -> hana_app.core.hierarchical_runner` edge. There is no direct `DrugMaster` import; the object is reached through `CodeStandardizer.drug_master`.

The scoped cycle rerun parsed 115 local Python modules:

- Import-time paths back to `serving.predictor`: none.
- Import-time reachable cycles: none.
- All-lexical-scope paths back to `serving.predictor`: none.
- All-lexical-scope reachable cycle: `scripts.train.base_graph_trainer -> scripts.train.trainer -> scripts.train.gat_trainer -> scripts.train.base_graph_trainer`.

Therefore, no static cycle involves `serving.predictor`, and its reachable import-time closure is acyclic. The broader all-scope graph is **not** a DAG because the optional GAT training branch contains a lazy structural cycle. The fixed-file cycle list and global-DAG claim in the stale plan were both incomplete.

## 8. Known WARNs

1. **Deployed artifacts uninspected.** This report cannot prove deployed bundle fields, byte-level globals, hashes, versions, or production startup selection.
2. **No durable test baseline.** An earlier interactive pytest run lacks durable baseline evidence, so this report makes no exact node, outcome, or failure-cause claim from it. Phase 1 must capture its own node/outcome baseline.
3. **Worktree `.venv` environment gaps.** `pytest==9.1.1` is installed, while `pydantic` and `ruff` package metadata are `NOT_INSTALLED` in this `.venv` (Appendix A). A Phase 1 baseline is not reproducible from this `.venv` as captured; running it requires an environment with the full serving/training dependencies. These gaps are environment WARNs, not product contract failures.
4. **Environment boundary.** The capture is WSL, not closed-network Windows production. Python 3.12 parity is present for this static run only; Windows wheel/package compatibility was not tested.
5. **Lazy dependency risks.** Broad lazy-import exception handling, process-global `sys.path` mutation, and optional training dependencies can defer failures until a selected model or request path executes.

## 9. Corrected stale plan assumptions

The approved evidence requires these corrections:

1. A literal-eval-only Task 1 extractor is insufficient. The source includes `AnnAssign`, wrapper calls such as `frozenset(...)`, name references, and date expressions; the safe extractor must handle the supported syntax explicitly and fail on missing targets.
2. `col_offset` does not identify lexical import scope. Scope-aware AST traversal is required for module, function, method, nested class, and `try` context.
3. The global DAG claim is false. Only the reachable import-time closure is acyclic; the all-scope optional GAT closure has the cycle recorded above.
4. A fixed-file cycle scan is incomplete. The approved scan covers all Python modules under `serving/`, `hana_app/`, `scripts/`, and `rules/`.
5. There is no direct `DrugMaster` import from `serving.predictor`; the instance is reached through `CodeStandardizer`.
6. The serving startup default is `/app/models/current/model_prod.pkl`, subject to configuration override, not the training output pattern `ddi_model_{partition}.pkl`.
7. `_EnsembleWrapper` is a current runtime adapter, not a proven serialized compatibility global.
8. No deployed values may be inferred from source defaults, writer shapes, configured paths, or this worktree's environment.

## 10. Reproduction instructions

Run the commands in Appendices A-E from the repository root with commit `BASELINE_SOURCE_SHA=3d8d64e78601a3ff56dc38034a9da62853e6b656` available and `.venv/bin/python` reporting Python 3.12. The publication commit may differ. Do not import repository modules. Do not point any command at model directories, protected paths, generated Parquet, HANA, or the frozen holdout.

Appendix A verifies source-snapshot provenance, environment variables, package metadata, input hashes, and selected source blobs without constraining current `HEAD`. Appendix B performs constant-only AST extraction and prints the complete deterministic JSON used by this report. Appendix C inventories imports with lexical scope. Appendix D performs the scoped static cycle scan. Appendix E asserts the exact report-directory Markdown allowlist, balanced fences, resolvable local links, source-path/blob parity with `BASELINE_SOURCE_SHA`, and nonzero failure on drift; it ignores unrelated worktree paths and does not run product tests.

## Appendix A: environment, Git, and provenance

Command:

```bash
.venv/bin/python - <<'PY'
import importlib.metadata
import os
import sys
print(f"python={sys.version.split()[0]}")
print(f"executable={sys.executable}")
print(f"path[:5]={sys.path[:5]!r}")
print(f"FEATURE_SCHEMA_LENIENT={os.environ.get('FEATURE_SCHEMA_LENIENT')!r}")
print(f"FEATURE_SCHEMA_LENIENT_SUNSET_DATE={os.environ.get('FEATURE_SCHEMA_LENIENT_SUNSET_DATE')!r}")
for name in ('pytest', 'pydantic', 'ruff'):
    try:
        value = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        value = 'NOT_INSTALLED'
    print(f"{name}={value}")
PY
BASELINE_SOURCE_SHA=3d8d64e78601a3ff56dc38034a9da62853e6b656
git cat-file -e "${BASELINE_SOURCE_SHA}^{commit}"
printf 'BASELINE_SOURCE_SHA=%s\n' "$BASELINE_SOURCE_SHA"
git diff --quiet "$BASELINE_SOURCE_SHA" -- serving hana_app scripts rules
source_diff_exit=$?
printf 'analyzed_source_diff_exit=%s\n' "$source_diff_exit"
[ "$source_diff_exit" -eq 0 ] || exit "$source_diff_exit"
sha256sum docs/superpowers/reports/contract-baseline/phase0a-profile-contract-map.md docs/superpowers/reports/contract-baseline/phase0a-feature-dispersion-table.md docs/superpowers/reports/contract-baseline/phase0a-bundle-metadata-record.md docs/superpowers/reports/contract-baseline/phase0b-dependency-graph.md
git rev-parse "$BASELINE_SOURCE_SHA:serving/predictor.py" "$BASELINE_SOURCE_SHA:hana_app/core/ml_runner.py" "$BASELINE_SOURCE_SHA:hana_app/core/hierarchical_runner.py" "$BASELINE_SOURCE_SHA:scripts/features/feature_engineer.py" "$BASELINE_SOURCE_SHA:scripts/datasets/contracts.py" "$BASELINE_SOURCE_SHA:scripts/etl/prescription_aggregator.py" "$BASELINE_SOURCE_SHA:serving/schemas.py" "$BASELINE_SOURCE_SHA:serving/dl_predictor.py" "$BASELINE_SOURCE_SHA:serving/hana_history.py"
```

Captured output:

```text
python=3.12.3
executable=/mnt/c/model/mode_11_hana/.worktrees/opencode-contract-phases/.venv/bin/python
path[:5]=['', '/usr/lib/python312.zip', '/usr/lib/python3.12', '/usr/lib/python3.12/lib-dynload', '/mnt/c/model/mode_11_hana/.worktrees/opencode-contract-phases/.venv/lib/python3.12/site-packages']
FEATURE_SCHEMA_LENIENT=None
FEATURE_SCHEMA_LENIENT_SUNSET_DATE=None
pytest=9.1.1
pydantic=NOT_INSTALLED
ruff=NOT_INSTALLED
BASELINE_SOURCE_SHA=3d8d64e78601a3ff56dc38034a9da62853e6b656
analyzed_source_diff_exit=0
790ebe0d49cdc410d23a72725ae4959e909e51afaa90cf539df8bb1096b3db79  docs/superpowers/reports/contract-baseline/phase0a-profile-contract-map.md
d348d6b734004e30c73d09452182bff4773e0b7fa526f765a7458854c96fac15  docs/superpowers/reports/contract-baseline/phase0a-feature-dispersion-table.md
50273bf53c06ab6f277261141e2de1b5df7f4cd67778e91ba61b8fb9168a0da6  docs/superpowers/reports/contract-baseline/phase0a-bundle-metadata-record.md
6ed500c16c697a518a0754de8aa918f450bb7eaacd527dae5f3c4a9aa85dd243  docs/superpowers/reports/contract-baseline/phase0b-dependency-graph.md
3903066df459885b44a48530aa6e1911801d9fb8
29bf7afcb634ce3630ec24942bb39905434a0886
2f5b30b2558664ffbfc97a994ae1a38b7bac9201
407222615d2d6981c47b914656271bd07cd99f73
7418e21229b7a25e2c7257aabe0ed6f356e96485
771ea7f2dbb1ff1ad23c676fd4e0a0aeb319dc4f
9e2cc58d2ae801de2a0681a8af407ca35042fea7
76c87f97e02e2c8ae96fb3141ffa802975e409b9
9b9d025b5240a07d1983c0b35ef630db533d25ed
```

## Appendix B: safe AST extraction

This command reads Python source as text and uses only the standard library. It does not import a repository module.

Command:

```bash
.venv/bin/python - <<'PY'
import ast
import json
from pathlib import Path

TARGETS = {
    "serving/predictor.py": ["_BUILDER_KNOWN_COLS", "_INTENTIONAL_FEATURE_ALLOWLIST", "_FEATURE_ALLOWED"],
    "hana_app/core/ml_runner.py": ["FEATURE_COLS", "RISK_LABEL_MAP", "_SAFE_MISCLS_FEATURES"],
    "scripts/features/feature_engineer.py": ["ETL_NUMERIC_COLS"],
    "scripts/datasets/contracts.py": ["ML_DATASET_REQUIRED_COLUMNS", "DL_DATASET_REQUIRED_COLUMNS", "DL_BUNDLE_REQUIRED_FILES"],
    "hana_app/core/hierarchical_runner.py": ["YELLOW_SUBTYPE_LABELS", "STAGE2_LABELS"],
    "scripts/etl/prescription_aggregator.py": ["DDI_FEATURE_SEMANTICS_VERSION", "FEATURE_SEMANTICS_VERSION"],
}
WRAPPERS = {"frozenset": frozenset, "set": set, "list": list, "tuple": tuple}

def evaluate(node, env):
    if isinstance(node, ast.Name):
        return env[node.id]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in WRAPPERS:
        return WRAPPERS[node.func.id](evaluate(node.args[0], env) if node.args else ())
    if isinstance(node, ast.BinOp):
        left, right = evaluate(node.left, env), evaluate(node.right, env)
        if isinstance(node.op, ast.BitOr): return left | right
        if isinstance(node.op, ast.Add): return left + right
    return ast.literal_eval(node)

def normalize(value):
    if isinstance(value, (set, frozenset)): return sorted(value)
    if isinstance(value, tuple): return list(value)
    return value

result = {"constants": {}}
values = {}
for rel, wanted in TARGETS.items():
    env = {}
    tree = ast.parse(Path(rel).read_text(encoding="utf-8"), filename=rel)
    found = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            value_node = node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None and isinstance(node.target, ast.Name):
            names, value_node = [node.target.id], node.value
        else:
            continue
        for name in names:
            try: value = evaluate(value_node, env)
            except (KeyError, ValueError, TypeError): continue
            env[name] = value
            if name in wanted:
                found[name] = {"line": node.lineno, "value": normalize(value)}
                values[name] = value
    missing = sorted(set(wanted) - set(found))
    if missing: raise SystemExit(f"missing {rel}: {missing}")
    result["constants"][rel] = found
B, F = set(values["_BUILDER_KNOWN_COLS"]), set(values["FEATURE_COLS"])
E, D = set(values["ETL_NUMERIC_COLS"]), set(values["ML_DATASET_REQUIRED_COLUMNS"])
result["counts"] = {"B": len(B), "F": len(F), "E": len(E), "D": len(D)}
result["differences"] = {
    "B\\F": sorted(B-F), "F\\B": sorted(F-B), "B\\E": sorted(B-E), "E\\B": sorted(E-B),
    "F\\E": sorted(F-E), "E\\F": sorted(E-F), "D\\E": sorted(D-E), "E\\D": sorted(E-D),
}
print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
PY
```

Captured output (complete):

```json
{
  "constants": {
    "hana_app/core/hierarchical_runner.py": {
      "STAGE2_LABELS": {
        "line": 31,
        "value": [
          "Y_TRIPLE",
          "Y_DOUBLE",
          "Y_DDI_MAJOR",
          "Y_DDI_MOD",
          "Y_DUP",
          "Y_FRAG",
          "No_Alert"
        ]
      },
      "YELLOW_SUBTYPE_LABELS": {
        "line": 28,
        "value": [
          "Y_TRIPLE",
          "Y_DOUBLE",
          "Y_DDI_MAJOR",
          "Y_DDI_MOD",
          "Y_DUP",
          "Y_FRAG"
        ]
      }
    },
    "hana_app/core/ml_runner.py": {
      "FEATURE_COLS": {
        "line": 50,
        "value": [
          "drug_count",
          "drug_count_7d",
          "institution_count",
          "ddi_contraindicated",
          "ddi_major",
          "ddi_moderate",
          "ddi_minor",
          "triple_whammy",
          "qt_risk_count",
          "dup_same_ingredient",
          "dup_atc5",
          "dup_atc4",
          "dup_atc3",
          "dup_efmdc",
          "has_high_risk_drug",
          "has_renal_risk_drug",
          "has_hepatic_risk_drug",
          "cyp_risk_score",
          "cyp_max_enzyme_risk",
          "cyp_high_risk_pairs",
          "age",
          "sex_m"
        ]
      },
      "RISK_LABEL_MAP": {
        "line": 75,
        "value": {
          "Green": 1,
          "Normal": 0,
          "Red": 3,
          "Yellow": 2
        }
      },
      "_SAFE_MISCLS_FEATURES": {
        "line": 91,
        "value": [
          "drug_count",
          "drug_count_7d",
          "institution_count",
          "ddi_contraindicated",
          "ddi_major",
          "ddi_moderate",
          "ddi_minor",
          "triple_whammy",
          "qt_risk_count",
          "dup_same_ingredient",
          "dup_atc5",
          "dup_atc4",
          "dup_atc3",
          "dup_efmdc",
          "has_high_risk_drug",
          "has_renal_risk_drug",
          "has_hepatic_risk_drug",
          "cyp_risk_score",
          "cyp_max_enzyme_risk",
          "cyp_high_risk_pairs"
        ]
      }
    },
    "scripts/datasets/contracts.py": {
      "DL_BUNDLE_REQUIRED_FILES": {
        "line": 41,
        "value": [
          "model.pt",
          "model_config.json",
          "drug_vocab.json",
          "edge_index.pt",
          "feature_normalizer.pkl",
          "schema_version.json"
        ]
      },
      "DL_DATASET_REQUIRED_COLUMNS": {
        "line": 35,
        "value": [
          "patient_id",
          "drug_code",
          "prescription_date"
        ]
      },
      "ML_DATASET_REQUIRED_COLUMNS": {
        "line": 23,
        "value": [
          "patient_id",
          "drug_count",
          "drug_count_7d",
          "institution_count",
          "ddi_contraindicated",
          "ddi_major",
          "ddi_moderate",
          "ddi_minor",
          "risk_level"
        ]
      }
    },
    "scripts/etl/prescription_aggregator.py": {
      "DDI_FEATURE_SEMANTICS_VERSION": {
        "line": 209,
        "value": "ddi.v2"
      },
      "FEATURE_SEMANTICS_VERSION": {
        "line": 216,
        "value": "rulefeat.v1"
      }
    },
    "scripts/features/feature_engineer.py": {
      "ETL_NUMERIC_COLS": {
        "line": 37,
        "value": [
          "drug_count",
          "drug_count_7d",
          "institution_count",
          "ddi_contraindicated",
          "ddi_major",
          "ddi_moderate",
          "ddi_minor",
          "triple_whammy",
          "qt_risk_count",
          "dup_same_ingredient",
          "dup_atc5",
          "dup_atc4",
          "dup_atc3",
          "age"
        ]
      }
    },
    "serving/predictor.py": {
      "_BUILDER_KNOWN_COLS": {
        "line": 45,
        "value": [
          "age",
          "avg_drug_duration",
          "cyp_high_risk_pairs",
          "cyp_max_enzyme_risk",
          "cyp_risk_score",
          "ddi_contraindicated",
          "ddi_major",
          "ddi_minor",
          "ddi_moderate",
          "drug_count",
          "drug_count_7d",
          "dup_atc3",
          "dup_atc4",
          "dup_atc5",
          "dup_efmdc",
          "dup_same_ingredient",
          "has_hepatic_risk_drug",
          "has_high_risk_drug",
          "has_renal_risk_drug",
          "institution_count",
          "long_term_drug_count",
          "qt_risk_count",
          "sex_m",
          "triple_whammy"
        ]
      },
      "_FEATURE_ALLOWED": {
        "line": 59,
        "value": [
          "age",
          "avg_drug_duration",
          "cyp_high_risk_pairs",
          "cyp_max_enzyme_risk",
          "cyp_risk_score",
          "ddi_contraindicated",
          "ddi_major",
          "ddi_minor",
          "ddi_moderate",
          "drug_count",
          "drug_count_7d",
          "dup_atc3",
          "dup_atc4",
          "dup_atc5",
          "dup_efmdc",
          "dup_same_ingredient",
          "has_hepatic_risk_drug",
          "has_high_risk_drug",
          "has_renal_risk_drug",
          "institution_count",
          "long_term_drug_count",
          "qt_risk_count",
          "sex_m",
          "triple_whammy"
        ]
      },
      "_INTENTIONAL_FEATURE_ALLOWLIST": {
        "line": 58,
        "value": []
      }
    }
  },
  "counts": {
    "B": 24,
    "D": 9,
    "E": 14,
    "F": 22
  },
  "differences": {
    "B\\E": [
      "avg_drug_duration",
      "cyp_high_risk_pairs",
      "cyp_max_enzyme_risk",
      "cyp_risk_score",
      "dup_efmdc",
      "has_hepatic_risk_drug",
      "has_high_risk_drug",
      "has_renal_risk_drug",
      "long_term_drug_count",
      "sex_m"
    ],
    "B\\F": [
      "avg_drug_duration",
      "long_term_drug_count"
    ],
    "D\\E": [
      "patient_id",
      "risk_level"
    ],
    "E\\B": [],
    "E\\D": [
      "age",
      "dup_atc3",
      "dup_atc4",
      "dup_atc5",
      "dup_same_ingredient",
      "qt_risk_count",
      "triple_whammy"
    ],
    "E\\F": [],
    "F\\B": [],
    "F\\E": [
      "cyp_high_risk_pairs",
      "cyp_max_enzyme_risk",
      "cyp_risk_score",
      "dup_efmdc",
      "has_hepatic_risk_drug",
      "has_high_risk_drug",
      "has_renal_risk_drug",
      "sex_m"
    ]
  }
}
```

## Appendix C: lexical import inventory

Command:

```bash
.venv/bin/python - <<'PY'
import ast
from pathlib import Path

p = Path("serving/predictor.py")
tree = ast.parse(p.read_text(encoding="utf-8"), filename=str(p))

class Imports(ast.NodeVisitor):
    def __init__(self):
        self.scope, self.try_lines, self.rows = [], [], []
    def visit_ClassDef(self, node):
        self.scope.append(node.name); self.generic_visit(node); self.scope.pop()
    def visit_FunctionDef(self, node):
        self.scope.append(node.name); self.generic_visit(node); self.scope.pop()
    visit_AsyncFunctionDef = visit_FunctionDef
    def visit_Try(self, node):
        self.try_lines.append(node.lineno); self.generic_visit(node); self.try_lines.pop()
    def visit_Import(self, node):
        for a in node.names:
            self.rows.append((node.lineno, a.name, a.asname or "-",
                              ".".join(self.scope) or "<module>",
                              self.try_lines[-1] if self.try_lines else "-"))
    def visit_ImportFrom(self, node):
        module = "." * node.level + (node.module or "")
        names = ", ".join(a.name + (f" as {a.asname}" if a.asname else "")
                          for a in node.names)
        self.rows.append((node.lineno, module, names,
                          ".".join(self.scope) or "<module>",
                          self.try_lines[-1] if self.try_lines else "-"))

v = Imports(); v.visit(tree)
print("line | module | names/alias | lexical scope | guarded by try@line")
for row in v.rows:
    print(" | ".join(map(str, row)))
PY
```

Captured output (complete):

```text
line | module | names/alias | lexical scope | guarded by try@line
19 | __future__ | annotations | <module> | -
21 | hashlib | - | <module> | -
22 | logging | - | <module> | -
23 | os | - | <module> | -
24 | pickle | - | <module> | -
25 | threading | - | <module> | -
26 | time | - | <module> | -
27 | datetime | date, datetime, timedelta | <module> | -
28 | pathlib | Path | <module> | -
29 | typing | Optional | <module> | -
31 | numpy | np | <module> | -
32 | pandas | pd | <module> | -
34 | .schemas | DDIAlert, DLPredictionResult, DrugItem, PredictRequest, PredictResponse, RiskLevel, Severity, INTERVENTION_MAP | <module> | -
38 | .dl_predictor | DLModel | <module> | -
39 | .hana_history | HANAHistoryProvider | <module> | -
143 | rules.risk_drug_constants | HIGH_RISK_KEYWORDS as _HIGH_RISK_KEYWORDS, HIGH_RISK_ATC_PREFIXES as _HIGH_RISK_ATC_PREFIXES, RENAL_RISK_KEYWORDS as _RENAL_RISK_KEYWORDS, RENAL_RISK_ATC_PREFIXES as _RENAL_RISK_ATC_PREFIXES, HEPATIC_RISK_KEYWORDS as _HEPATIC_RISK_KEYWORDS, HEPATIC_RISK_ATC_PREFIXES as _HEPATIC_RISK_ATC_PREFIXES | <module> | -
172 | sys | - | _detect_risk_flags | 171
174 | scripts.etl.prescription_aggregator | _RENAL_RISK_KEYWORDS, _RENAL_RISK_ATC_PREFIXES, _HEPATIC_RISK_KEYWORDS, _HEPATIC_RISK_ATC_PREFIXES | _detect_risk_flags | 171
214 | sys | - | _run_safety_net | 212
216 | rules.safety_net | SafetyNet | _run_safety_net | 212
273 | rules.duplicate_detector | DuplicateDetector | _run_duplicate_detector | 271
394 | scripts.etl.prescription_aggregator | DDI_FEATURE_SEMANTICS_VERSION as _cur_ddi_ver | MLModel.load | 373
460 | pickle | _pk | MLModel.load | 459
477 | numpy | np | MLModel.load._EnsembleWrapper.predict_proba | 459
499 | scripts.train.gat_trainer | GATTrainer | MLModel.load | 498
502 | json | - | MLModel.load | 498
503 | datetime | datetime, timezone | MLModel.load | 498
575 | itertools | combinations | MLModel.predict_proba_gat | -
665 | json | - | HierarchicalPredictor.load | 664
666 | joblib | - | HierarchicalPredictor.load | 664
674 | hana_app.core.hierarchical_runner | STAGE2_LABELS as _CURRENT_STAGE2_LABELS | HierarchicalPredictor.load | 664
689 | scripts.etl.prescription_aggregator | DDI_FEATURE_SEMANTICS_VERSION as _CUR_DDI_VER | HierarchicalPredictor.load | 664
786 | sys | _sys | HierarchicalPredictor.predict_risk_single | -
788 | hana_app.core.hierarchical_runner | predict_risk | HierarchicalPredictor.predict_risk_single | -
844 | scripts.etl.models | PrescriptionRecord | RequestFeatureBuilder._build_ddi_records | -
879 | scripts.etl.overlap_calculator | calculate_overlaps_for_patient | RequestFeatureBuilder._count_ddi | -
880 | scripts.etl.prescription_aggregator | count_ddi_severities | RequestFeatureBuilder._count_ddi | -
892 | scripts.etl.overlap_calculator | calculate_overlaps_for_patient | RequestFeatureBuilder.ddi_alert_pairs | -
893 | scripts.etl.prescription_aggregator | ddi_pair_severities | RequestFeatureBuilder.ddi_alert_pairs | -
917 | scripts.etl.models | PrescriptionRecord, PatientFeatures | RequestFeatureBuilder._count_dup_features | -
918 | scripts.etl.overlap_calculator | get_concurrent_drug_count | RequestFeatureBuilder._count_dup_features | -
919 | scripts.etl.prescription_aggregator | _fill_dup_features | RequestFeatureBuilder._count_dup_features | -
960 | types | SimpleNamespace | RequestFeatureBuilder._rule_namespace | -
964 | scripts.etl.prescription_aggregator | detect_triple_whammy, detect_risk_drug, _HIGH_RISK_KEYWORDS, _RENAL_RISK_KEYWORDS, _HEPATIC_RISK_KEYWORDS | RequestFeatureBuilder._rule_namespace | -
987 | scripts.etl.clinical_rules | collect_red_triggers | RequestFeatureBuilder.red_triggers | -
998 | scripts.etl.clinical_rules | collect_severe_immediate_triggers | RequestFeatureBuilder.rule_floor | -
1058 | collections | Counter | RequestFeatureBuilder.build | -
1091 | scripts.etl.prescription_aggregator | detect_triple_whammy, detect_risk_drug | RequestFeatureBuilder.build | -
1119 | pandas | pd | RequestFeatureBuilder.build | -
1214 | scripts.etl.code_standardizer | CodeStandardizer | HybridPredictor.__init__ | 1213
1223 | scripts.features.cyp_features | CYPFeatureExtractor | HybridPredictor.__init__ | 1222
1265 | sys | _sys | HybridPredictor.__init__ | 1264
1267 | rules.safety_net | SafetyNet | HybridPredictor.__init__ | 1264
1273 | rules.duplicate_detector | DuplicateDetector | HybridPredictor.__init__ | 1272
1390 | scripts.etl.prescription_aggregator | FEATURE_SEMANTICS_VERSION | HybridPredictor.predict | -
1470 | hana_app.core.hierarchical_runner | ACTION_BY_LABEL | HybridPredictor.predict | -
```

## Appendix D: scoped cycle check

Command:

```bash
PYTHONWARNINGS='ignore::SyntaxWarning' .venv/bin/python - <<'PY'
import ast
from pathlib import Path

ROOTS = ("serving", "hana_app", "scripts", "rules")
mods = {}
for root in ROOTS:
    for p in Path(root).rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        parts = list(p.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        mods[".".join(parts)] = p

def resolve(owner, node):
    if isinstance(node, ast.Import):
        return [a.name for a in node.names]
    if not node.level:
        return [node.module or ""]
    package = owner.split(".")[:-1]
    base = package[:len(package) - (node.level - 1)]
    return [".".join(base + ([node.module] if node.module else []))]

def local_target(name):
    parts = name.split(".")
    for i in range(len(parts), 0, -1):
        candidate = ".".join(parts[:i])
        if candidate in mods:
            return candidate

def build(import_time_only):
    graph = {m: set() for m in mods}
    for owner, p in mods.items():
        tree = ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
        if import_time_only:
            nodes = []
            class ImportTime(ast.NodeVisitor):
                def visit_FunctionDef(self, node): pass
                visit_AsyncFunctionDef = visit_FunctionDef
                def visit_Lambda(self, node): pass
                def visit_Import(self, node): nodes.append(node)
                def visit_ImportFrom(self, node): nodes.append(node)
            ImportTime().visit(tree)
        else:
            nodes = [n for n in ast.walk(tree)
                     if isinstance(n, (ast.Import, ast.ImportFrom))]
        for node in nodes:
            for name in resolve(owner, node):
                target = local_target(name)
                if target and target != owner:
                    graph[owner].add(target)
    return graph

def cycles(graph, start):
    found = set()
    def dfs(node, path, positions):
        if node in positions:
            core = path[positions[node]:]
            rotations = [tuple(core[i:] + core[:i]) for i in range(len(core))]
            found.add(min(rotations)); return
        positions = dict(positions); positions[node] = len(path); path = path + [node]
        for nxt in sorted(graph.get(node, ())):
            dfs(nxt, path, positions)
    dfs(start, [], {})
    return found

def paths_back(graph, start):
    found = []
    for dep in sorted(graph[start]):
        def walk(node, path):
            if node == start:
                found.append(path); return
            if node in path[:-1]:
                return
            for nxt in sorted(graph.get(node, ())):
                walk(nxt, path + [nxt])
        walk(dep, [start, dep])
    return found

for label, import_time_only in (("IMPORT_TIME", True), ("ALL_LEXICAL_SCOPES", False)):
    graph = build(import_time_only)
    print(label)
    back = paths_back(graph, "serving.predictor")
    print("paths back to serving.predictor:", "NONE" if not back else "")
    for path in back:
        print("  " + " -> ".join(path))
    found_cycles = cycles(graph, "serving.predictor")
    print("reachable cycles:", "NONE" if not found_cycles else "")
    for cycle in sorted(found_cycles):
        print("  " + " -> ".join(cycle + (cycle[0],)))
print(f"PARSED_LOCAL_MODULES={len(mods)}")
PY
```

Captured output:

```text
IMPORT_TIME
paths back to serving.predictor: NONE
reachable cycles: NONE
ALL_LEXICAL_SCOPES
paths back to serving.predictor: NONE
reachable cycles:
  scripts.train.base_graph_trainer -> scripts.train.trainer -> scripts.train.gat_trainer -> scripts.train.base_graph_trainer
PARSED_LOCAL_MODULES=115
```

## Appendix E: document validation

This asserts the exact six-file Markdown allowlist in `contract-baseline`, balanced code fences, resolvable relative Markdown links (including heading fragments), source-snapshot SHA presence, and exact analyzed Python path/blob parity with `BASELINE_SOURCE_SHA`. Only the report directory and the Python files analyzed under `serving`, `hana_app`, `scripts`, and `rules` are inspected; unrelated worktree files are ignored. Any missing or extra Markdown file, malformed fence, broken link, source path drift, or source blob drift exits nonzero. It runs no product tests. The output below was captured on the final documents; embedding it adds balanced fences and no local links, so a re-run on the published files reproduces the same lines.

Command:

```bash
.venv/bin/python - <<'PY'
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote

BASE = Path("docs/superpowers/reports/contract-baseline")
BASELINE_SOURCE_SHA = "3d8d64e78601a3ff56dc38034a9da62853e6b656"
EXPECTED_MARKDOWN = (
    "README.md",
    "phase0a-bundle-metadata-record.md",
    "phase0a-feature-dispersion-table.md",
    "phase0a-profile-contract-map.md",
    "phase0b-baseline-report.md",
    "phase0b-dependency-graph.md",
)
ANALYZED_SOURCE_ROOTS = ("serving", "hana_app", "scripts", "rules")
failures = []

def fence_state(text):
    opened = None
    body = []
    for line in text.splitlines():
        match = re.match(r"^\s*(`{3,}|~{3,})", line)
        if match:
            marker = match.group(1)
            if opened is None:
                opened = (marker[0], len(marker))
            elif marker[0] == opened[0] and len(marker) >= opened[1]:
                opened = None
            continue
        if opened is None:
            body.append(line)
    return opened is None, "\n".join(body)

def heading_anchors(text):
    anchors = set()
    counts = {}
    for line in text.splitlines():
        match = re.match(r"^#{1,6}\s+(.+?)\s*#*\s*$", line)
        if not match:
            continue
        slug = match.group(1).strip().lower()
        slug = re.sub(r"[^\w\- ]", "", slug)
        slug = re.sub(r"\s", "-", slug)
        index = counts.get(slug, 0)
        counts[slug] = index + 1
        anchors.add(slug if index == 0 else f"{slug}-{index}")
    return anchors

actual_markdown = tuple(sorted(p.name for p in BASE.iterdir()
                               if p.is_file() and p.suffix == ".md"))
missing_markdown = sorted(set(EXPECTED_MARKDOWN) - set(actual_markdown))
unexpected_markdown = sorted(set(actual_markdown) - set(EXPECTED_MARKDOWN))
allowlist_ok = not missing_markdown and not unexpected_markdown
print(f"markdown_allowlist={'OK' if allowlist_ok else 'FAIL'} expected={len(EXPECTED_MARKDOWN)} actual={len(actual_markdown)}")
if not allowlist_ok:
    failures.append(f"markdown_allowlist missing={missing_markdown} unexpected={unexpected_markdown}")

for name in EXPECTED_MARKDOWN:
    path = BASE / name
    if not path.is_file():
        continue
    text = (BASE / name).read_text(encoding="utf-8")
    balanced, body = fence_state(text)
    broken_links = []
    raw_links = re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", body)
    link_syntax_ok = body.count("](") == len(raw_links)
    for raw_target in raw_links:
        target = unquote(raw_target.strip().split()[0].strip("<>"))
        if re.match(r"^[a-z][a-z0-9+.-]*:", target, re.I):
            continue
        rel, _, fragment = target.partition("#")
        linked = path if not rel else (path.parent / rel).resolve()
        if not linked.is_file():
            broken_links.append(target)
            continue
        if fragment:
            linked_text = linked.read_text(encoding="utf-8")
            linked_balanced, linked_body = fence_state(linked_text)
            if not linked_balanced or fragment not in heading_anchors(linked_body):
                broken_links.append(target)
    sha_ok = BASELINE_SOURCE_SHA in text
    links_ok = link_syntax_ok and not broken_links
    print(f"{name}: fences={'OK' if balanced else 'FAIL'} links={'OK' if links_ok else 'FAIL'} snapshot_sha={'OK' if sha_ok else 'FAIL'}")
    if not balanced:
        failures.append(f"unbalanced_fences:{name}")
    if not links_ok:
        failures.append(f"broken_links:{name}:syntax={link_syntax_ok}:{sorted(set(broken_links))}")
    if not sha_ok:
        failures.append(f"missing_snapshot_sha:{name}")

tree = subprocess.run(
    ["git", "ls-tree", "-r", "-z", BASELINE_SOURCE_SHA, "--", *ANALYZED_SOURCE_ROOTS],
    check=True, capture_output=True,
).stdout
baseline_blobs = {}
for record in tree.split(b"\0"):
    if not record:
        continue
    metadata, raw_path = record.split(b"\t", 1)
    _, kind, blob = metadata.decode().split()
    source_path = raw_path.decode()
    if kind == "blob" and source_path.endswith(".py"):
        baseline_blobs[source_path] = blob

actual_source_paths = {
    p.as_posix()
    for root in ANALYZED_SOURCE_ROOTS
    for p in Path(root).rglob("*.py")
    if "__pycache__" not in p.parts
}
baseline_source_paths = set(baseline_blobs)
missing_source = sorted(baseline_source_paths - actual_source_paths)
unexpected_source = sorted(actual_source_paths - baseline_source_paths)
path_ok = not missing_source and not unexpected_source
blob_drift = []
for source_path in sorted(actual_source_paths & baseline_source_paths):
    actual_blob = subprocess.run(
        ["git", "hash-object", "--", source_path], check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if actual_blob != baseline_blobs[source_path]:
        blob_drift.append(source_path)
blob_ok = not blob_drift
print(f"analyzed_source: paths={'OK' if path_ok else 'FAIL'} blobs={'OK' if blob_ok else 'FAIL'} files={len(baseline_source_paths)}")
if not path_ok:
    failures.append(f"source_path_drift missing={missing_source} unexpected={unexpected_source}")
if not blob_ok:
    failures.append(f"source_blob_drift:{blob_drift}")

print("VALIDATION=" + ("PASS" if not failures else f"FAIL:{failures}"))
if failures:
    raise SystemExit(1)
PY
```

Captured output:

```text
markdown_allowlist=OK expected=6 actual=6
README.md: fences=OK links=OK snapshot_sha=OK
phase0a-bundle-metadata-record.md: fences=OK links=OK snapshot_sha=OK
phase0a-feature-dispersion-table.md: fences=OK links=OK snapshot_sha=OK
phase0a-profile-contract-map.md: fences=OK links=OK snapshot_sha=OK
phase0b-baseline-report.md: fences=OK links=OK snapshot_sha=OK
phase0b-dependency-graph.md: fences=OK links=OK snapshot_sha=OK
analyzed_source: paths=OK blobs=OK files=115
VALIDATION=PASS
```
