# Phase 0B: `serving/predictor.py` Dependency Graph

## 0. Scope and source snapshot

This report implements Phase 0B Task 4 only. It is a static inventory: every result below comes from reading source text with `ast`, `rg`, `sed`, or Git. No project module was imported, no model or data artifact was opened, and no parquet or protected path was read or changed.

| Item | Value |
|---|---|
| Git snapshot SHA | `3d8d64e78601a3ff56dc38034a9da62853e6b656` |
| `serving/predictor.py` Git blob | `3903066df459885b44a48530aa6e1911801d9fb8` |
| `serving/predictor.py` SHA-256 | `24f2b162950f5f34247bb094e205c9e372df67ae1a4c07e278aee3808968877c` |
| Source length | 1,560 lines |
| Static-analysis interpreter | `.venv/bin/python`, Python 3.12.3 |

The working-tree blob and `HEAD:serving/predictor.py` blob were identical, so the analysis is tied to the named commit rather than an uncommitted source variant.

Reproduction command and output:

```bash
git rev-parse HEAD
git hash-object serving/predictor.py
git rev-parse HEAD:serving/predictor.py
git diff --quiet HEAD -- serving/predictor.py; echo "predictor_diff_exit=$?"
sha256sum serving/predictor.py
wc -l serving/predictor.py
.venv/bin/python --version
```

```text
3d8d64e78601a3ff56dc38034a9da62853e6b656
3903066df459885b44a48530aa6e1911801d9fb8
3903066df459885b44a48530aa6e1911801d9fb8
predictor_diff_exit=0
24f2b162950f5f34247bb094e205c9e372df67ae1a4c07e278aee3808968877c  serving/predictor.py
1560 serving/predictor.py
Python 3.12.3
```

## 1. Import inventory

The plan's `node.col_offset` heuristic was replaced with lexical-scope tracking. This distinguishes module-scope imports from imports inside functions, methods, and the nested `_EnsembleWrapper` class without executing the file. `guarded by try@line` records the enclosing `try`, not whether an import is optional in every caller.

Reproduction command:

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

### 1.1 Module-scope imports

```text
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
```

### 1.2 Function/method-local imports

```text
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

## 2. Dependency categories

| Category | Direct modules | Load behavior and role |
|---|---|---|
| Python standard library | `hashlib`, `logging`, `os`, `pickle`, `threading`, `time`, `datetime`, `pathlib`, `typing`; local `sys`, `json`, `itertools`, `types`, `collections` | Integrity, synchronization, environment/deadline handling, serialization, path handling, and local helpers. `pickle` is both module-scope and locally aliased. |
| Third party | `numpy`, `pandas`; local `joblib` | Numeric/dataframe serving and hierarchical bundle loading. `numpy`/`pandas` are eager; `joblib` is delayed until hierarchical load. |
| Serving-internal | `serving.schemas`, `serving.dl_predictor`, `serving.hana_history` | Request/response/domain types and intervention map; DL model; HANA history provider. All three are import-time edges. |
| Rule policy | `rules.risk_drug_constants`; local `rules.safety_net`, `rules.duplicate_detector` | Risk keyword/ATC constants are eager. Safety and duplicate engines are lazy/guarded and also initialized once in `HybridPredictor.__init__`. |
| ETL/feature parity | local `scripts.etl.models`, `overlap_calculator`, `prescription_aggregator`, `clinical_rules`, `code_standardizer`; `scripts.features.cyp_features` | Reuses training-side record types, overlap/DDI/duplicate/rule functions, semantic-version constants, EDI mapping, and CYP extraction. These are call- or initialization-time edges. |
| UI/domain policy | local `hana_app.core.hierarchical_runner` | Supplies the hierarchical label guard, inference function, and action mapping. This is the explicit `serving -> hana_app` boundary violation. |
| Optional training/runtime | local `scripts.train.gat_trainer` | Loaded only for an `EnsembleTrainer3Way` bundle. It pulls a broader training dependency closure into serving. |

`RequestFeatureBuilder._drug_master()` does **not** import `DrugMaster`; it reaches `self._std.drug_master` through a `CodeStandardizer` instance. The draft graph's implied direct `DrugMaster` edge is therefore stale.

## 3. Function/class dependency map

| Owner | Project dependencies | Contract/runtime purpose |
|---|---|---|
| `_detect_risk_flags` (169) | `scripts.etl.prescription_aggregator` risk constants | Lazy fallback source for renal/hepatic flags; mutates `sys.path` first. |
| `_run_safety_net` (199) | `rules.safety_net.SafetyNet` | Rule grade, reasons, and DDI alerts; optional construction path. |
| `_run_duplicate_detector` (265) | `rules.duplicate_detector.DuplicateDetector` | Duplicate count and reasons; optional construction path. |
| `MLModel` (305) | `prescription_aggregator.DDI_FEATURE_SEMANTICS_VERSION`; `scripts.train.gat_trainer.GATTrainer` | Tabular bundle semantic guard and optional three-way/GAT restoration. `predict_proba_gat` also uses stdlib `itertools`. |
| `HierarchicalPredictor` (626) | `hierarchical_runner.STAGE2_LABELS`, `hierarchical_runner.predict_risk`; `prescription_aggregator.DDI_FEATURE_SEMANTICS_VERSION` | Bundle label/version guards and request-time hierarchical inference. |
| `RequestFeatureBuilder` (805) | `scripts.etl.models`; `overlap_calculator`; `prescription_aggregator`; `clinical_rules` | Online feature construction reusing training-side records, overlap/DDI/duplicate logic, risk rules, and deterministic backstops. `_drug_master` is an indirect object edge via `CodeStandardizer`. |
| `HybridPredictor` (1179) | `CodeStandardizer`; `CYPFeatureExtractor`; `SafetyNet`; `DuplicateDetector`; `prescription_aggregator.FEATURE_SEMANTICS_VERSION`; `hierarchical_runner.ACTION_BY_LABEL`; serving-internal schema/DL/history types | Composes resources, models, rule engines, hot-swap locks, main prediction flow, rule-feature gating, subtype action mapping, and auxiliary DL inference. |
| `get_predictor` / `init_predictor` (1550/1557) | `HybridPredictor` only | Module singleton access/creation; no additional imports. |

Shared module-scope schema symbols (`DrugItem`, `PredictRequest`, `PredictResponse`, `RiskLevel`, `Severity`, `DDIAlert`, `DLPredictionResult`, `INTERVENTION_MAP`) couple nearly every helper and class to `serving.schemas`, even where no local import appears in the table.

## 4. Direct local dependency graph

```text
serving.predictor
├── serving.schemas                  [eager: API/domain types + INTERVENTION_MAP]
├── serving.dl_predictor             [eager: DLModel]
├── serving.hana_history             [eager: HANAHistoryProvider]
├── rules.risk_drug_constants        [eager: keyword/ATC policy constants]
├── rules.safety_net                 [lazy/guarded: SafetyNet]
├── rules.duplicate_detector         [lazy/guarded: DuplicateDetector]
├── scripts.etl.code_standardizer    [lazy/guarded: CodeStandardizer]
├── scripts.features.cyp_features    [lazy/guarded: CYPFeatureExtractor]
├── scripts.etl.models               [lazy: PrescriptionRecord, PatientFeatures]
├── scripts.etl.overlap_calculator   [lazy: overlap and concurrency functions]
├── scripts.etl.prescription_aggregator
│   └── [lazy: DDI/rule semantic versions, DDI/duplicate/risk functions/constants]
├── scripts.etl.clinical_rules       [lazy: Red/severe trigger collectors]
├── scripts.train.gat_trainer        [lazy/optional: GATTrainer]
└── hana_app.core.hierarchical_runner
    └── [lazy: STAGE2_LABELS, predict_risk, ACTION_BY_LABEL]
```

AST resolution found these 14 direct local-module edges:

```text
serving.predictor -> hana_app.core.hierarchical_runner
serving.predictor -> rules.duplicate_detector
serving.predictor -> rules.risk_drug_constants
serving.predictor -> rules.safety_net
serving.predictor -> scripts.etl.clinical_rules
serving.predictor -> scripts.etl.code_standardizer
serving.predictor -> scripts.etl.models
serving.predictor -> scripts.etl.overlap_calculator
serving.predictor -> scripts.etl.prescription_aggregator
serving.predictor -> scripts.features.cyp_features
serving.predictor -> scripts.train.gat_trainer
serving.predictor -> serving.dl_predictor
serving.predictor -> serving.hana_history
serving.predictor -> serving.schemas
```

Notable transitive eager edge: `serving.dl_predictor -> serving.hana_history` (`serving/dl_predictor.py:25`). This is not a cycle. `serving.dl_predictor` loads `torch` dynamically with `importlib.import_module("torch")` at line 227, keeping Torch off the initial import path.

## 5. Explicit `serving -> hana_app` edge

| Serving location | Imported symbol | Timing | Use |
|---|---|---|---|
| `HierarchicalPredictor.load`, line 674 | `STAGE2_LABELS` | Bundle-load time, inside `try` at 664 | Fail-fast equality guard for serialized Stage 2 labels. |
| `HierarchicalPredictor.predict_risk_single`, line 788 | `predict_risk` | Request/inference time | Executes two-stage hierarchical inference. |
| `HybridPredictor.predict`, line 1470 | `ACTION_BY_LABEL` | Request time, only when a deterministic subtype floor exists | Maps the enforced Yellow subtype to its action. |

The source also makes `hana_app.core.hierarchical_runner` mutate `sys.path` at import time (lines 21-25) before importing `scripts.etl.clinical_rules`. Thus the serving edge crosses both a UI/domain ownership boundary and a process-global import-path boundary.

## 6. Circular-dependency check

The plan's proposed test (“does any imported module import `serving.*`?”) is too broad and incomplete: it would treat legitimate sibling edges such as `serving.dl_predictor -> serving.hana_history` as circular, while its fixed file list omits current direct dependencies. The replacement statically parses every Python module under `serving/`, `hana_app/`, `scripts/`, and `rules/` (115 modules), resolves local imports, and checks both import-time and all-lexical-scope graphs.

Reproduction command:

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

Output:

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

Conclusion:

- There is **no circular dependency involving `serving.predictor`**: no direct dependency has a static path back to it in either graph.
- The import-time dependency closure reachable from `serving.predictor` is acyclic.
- Contrary to the stale plan statement that the entire graph is a DAG, the all-scope graph contains a lazy cycle in the optional GAT training branch. The edges are `base_graph_trainer.py:9 -> trainer`, method-local imports at `trainer.py:316` and `trainer.py:496 -> gat_trainer`, and `gat_trainer.py:13 -> base_graph_trainer`. Because the `trainer -> gat_trainer` edges are inside methods, this is not an import-time cycle, but it is architectural coupling and should not be described as a global DAG.

## 7. Lazy-import and `sys.path` risks

| Risk | Static evidence | Impact |
|---|---|---|
| Process-global path mutation | `serving/predictor.py:173,215,787,1266` unconditionally calls `sys.path.insert(0, repo_root)` on the entered branch; `hana_app/core/hierarchical_runner.py:25` conditionally does the same at import time. | Repeated calls can prepend duplicate entries; import resolution can shadow installed packages; behavior depends on checkout layout and process history. The mutation is never rolled back. |
| Broad lazy-import exception handling | Many optional imports sit inside `try/except Exception`, including `MLModel.load`, `HierarchicalPredictor.load`, and `HybridPredictor.__init__`. | Missing packages and genuine source/API errors may collapse into the same fallback or load-failure result, delaying detection until a particular bundle or request path executes. |
| Serving pulls training code lazily | `MLModel.load:499` imports `scripts.train.gat_trainer.GATTrainer`; feature building imports multiple `scripts.etl.*` modules. | A closed-network production bundle may require code/dependencies normally viewed as training-only. The failure surface varies by selected model/profile. |
| Lazy GAT structural cycle | `base_graph_trainer -> trainer -> (method-local) gat_trainer -> base_graph_trainer`. | Safe from an immediate import-time loop in the current layout, but refactoring either lazy edge to module scope could create a real partially initialized-module failure. |
| Deferred Torch load | `serving/dl_predictor.py:227` uses `importlib.import_module("torch")`. | Initial import stays light, but DL failures move to first prediction/runtime-load use; deployment checks must cover that boundary separately. |
| Runtime policy ownership | `STAGE2_LABELS`, `predict_risk`, and `ACTION_BY_LABEL` come from `hana_app.core.hierarchical_runner` at different times. | Serving startup/load and request behavior depend on a UI-owned module plus its sklearn/pandas and path-mutation import surface. |

No `importlib` or `__import__` call exists in `serving/predictor.py` itself; its dynamic behavior is implemented with ordinary imports inside functions/methods. The only direct-serving-module `importlib` use found is the deliberate lazy Torch load in `serving.dl_predictor`.

## 8. Corrected stale plan assumptions

1. Add the omitted eager edge to `rules.risk_drug_constants`.
2. Add the current `HybridPredictor.predict -> prescription_aggregator.FEATURE_SEMANTICS_VERSION` edge at line 1390.
3. Do not claim a direct `DrugMaster` import; the object is reached through `CodeStandardizer.drug_master`.
4. Treat `serving.schemas`, `serving.dl_predictor`, and `serving.hana_history` as part of the circular scan; the fixed plan list omitted them.
5. Do not classify every `serving.*` import in a dependency as circular. `serving.dl_predictor -> serving.hana_history` is a sibling dependency, not a path back to `serving.predictor`.
6. Replace “the dependency graph is a DAG” with the scoped result in section 6: the import-time closure is acyclic and no path returns to `serving.predictor`, but the all-scope optional GAT closure contains a lazy structural cycle.
7. `GATTrainer` is the imported symbol; `EnsembleTrainer3Way` is the bundle/trainer-class condition that activates the branch, not an alias of `GATTrainer`.

## 9. Phase 3 boundary

Phase 0B records dependencies only. It does not split `serving/predictor.py`, move or copy `predict_risk`, `ACTION_BY_LABEL`, or `STAGE2_LABELS`, remove `sys.path` mutations, reorganize training/ETL modules, change feature/label/version/threshold contracts, merge inference engines, migrate artifacts, or retrain.

The authorized future Phase 3 boundary is predictor/domain extraction: separate `serving/predictor.py` responsibilities, move the pure hierarchical domain policy to a neutral shared module, remove the runtime `serving -> hana_app.core` dependency, and preserve compatibility imports where required. Wide tabular/hierarchical engine integration remains NO-GO. Any label-space, train-serving schema, or HANA-query impact requires the repository's critical cross-family review and serving parity gates. `FEATURE_SCHEMA_LENIENT` removal is a separate post-Phase-3 change, not part of this task.

The lazy GAT cycle and `sys.path` cleanup are risks for future design review, not authorization to expand Phase 3 or change them here.

## 10. Static-safety and change boundary

The analysis did not import `serving`, `hana_app`, `scripts`, or `rules`; it only parsed their source. It did not access `models/`, `data/`, generated parquet, `packages_win/py312/`, `mlruns/`, or `out/`. The only file created for Task 4 is this report. No commit was made.
