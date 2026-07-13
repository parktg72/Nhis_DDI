# Phase 0A: Bundle Metadata Record

**Status:** Static-source inventory only. No artifact was opened or executed.

**Source snapshot:** `3d8d64e78601a3ff56dc38034a9da62853e6b656`. The inspected `serving/`, `hana_app/`, `scripts/`, and `config/` source trees were unmodified at inspection time.

## Safety boundary

This record was produced from repository text and Python source only. **No bundle bytes were inspected. No model-side JSON metadata was inspected.** In particular, no model, pickle, joblib, Torch, manifest, `stage_meta.json`, `model_config.json`, `schema_version.json`, or other artifact file was opened. No `pickle.load`, `pickle.loads`, `joblib.load`, `torch.load`, `torch.jit.load`, or `pickletools` command was run.

The inspection did not read `models/`, `packages_win/py312/`, `mlruns/`, generated Parquet files, `out/`, or the frozen holdout. Names and metadata shapes below describe source-code contracts, not the contents of any deployed bundle.

## 1. Configured path semantics

### 1.1 Tabular binary

- Training source writes `model_dir / f"ddi_model_{partition}.pkl"` (`scripts/train/pipeline.py:143`). This is a training output pattern, not the serving startup default.
- Serving configuration defaults `MODEL_DIR` to `/app/models`, derives `MODEL_PROD_PATH` as `MODEL_DIR / "current" / "model_prod.pkl"`, and permits `MODEL_PATH` to override that startup path (`config/settings.py:16,24`; `serving/main.py:72-77`). Thus the current production default is `/app/models/current/model_prod.pkl`, not `models/ddi_model_{partition}.pkl`.
- `MLModel.load(path)` uses the supplied path directly. It reads the main hash from `<path>.sha256` and rejects a missing or mismatched hash (`serving/predictor.py:324-341,371-378`).
- `scaler_path` and `selector_path`, when present in the main state, are interpreted relative to the main model's parent. The resolved path must remain inside that parent; traversal, absence, or hash failure rejects the load (`serving/predictor.py:424-448`). Training stores these as relative paths (`scripts/train/pipeline.py:146-157`).
- Ensemble sidecars are sibling names derived with `path.with_suffix(".xgb.pkl")` and `.lgb.pkl`; each also needs its own `.sha256` (`serving/predictor.py:454-468`). For `EnsembleTrainer3Way`, `gat_model.pt` is fixed in the main model's parent, and `gat_graph_meta.json` is optionally read from the same directory (`serving/predictor.py:491-506`).
- Startup only calls the tabular loader when the resolved path exists and no hierarchical model was accepted (`serving/predictor.py:1258-1259`). Admin reload resolves the request and confines it under the configured `MODEL_DIR` (`serving/routers/health.py:228-250`).

Source-visible main payload fields are not all equally required. `MLModel.load` reads `model`, `best_threshold` (default `0.5`), `trainer_class` (default `"unknown"`), `feature_names` (default `[]`), `artifact_version` (default `1`), `partition`, `ddi_feature_semantics_version` when a `ddi_*` feature is present, `scaler_path`, `selector_path`, and `weights` for ensembles (`serving/predictor.py:378-397,424-469,492-524`). Current training also writes `params` and `feature_importances` for a single trainer, while ensemble top-level payloads omit `model` and use sidecars (`scripts/train/trainer.py:79-94,272-289,449-473`). This corrects the plan's stale implication that one fixed key list applies to every tabular bundle.

### 1.2 Hierarchical

- `HIERARCHICAL_MODEL_DIR` is an optional startup environment value. If set, it is tried before the tabular model; a missing, invalid, or schema-rejected directory falls back to the tabular path (`serving/main.py:74-81`; `serving/predictor.py:1232-1259`).
- The directory contract fixes three child names: `stage_meta.json`, `stage1_red.joblib`, and `stage2_yellow.joblib` (`serving/predictor.py:652-657`). Admin reload confines the directory under `MODEL_DIR` (`serving/routers/health.py:253-274`).
- Current training source writes metadata keys `clinical_standards_version`, `ddi_feature_semantics_version`, `feature_semantics_version`, `feature_cols`, `thresholds`, `stage2_labels`, `stage2_label_counts`, `y_other_excluded_count`, `stage1_sha256`, `stage2_sha256`, `cost_sensitive`, `cost_ratio_by_class`, `stage1_trained`, and `stage1_red_count` (`hana_app/core/hierarchical_runner.py:620-642`). The plan's shorter list was incomplete.
- The loader parses and validates JSON metadata and both hashes before either `joblib.load` call (`serving/predictor.py:664-735`). This report records that source behavior without reading any actual `stage_meta.json`.

### 1.3 DL history

- Configuration defines `DL_BUNDLE_DIR` from `DDI_DL_BUNDLE_DIR`, defaulting to `MODEL_DIR / "dl" / "current"` (`config/settings.py:21`). A scoped grep of the serving, HANA application, scripts, and configuration source trees found no consumer other than this definition. Current startup constructs an empty `DLModel` but does not pass or load `settings.DL_BUNDLE_DIR` (`serving/predictor.py:1198-1201`; `serving/main.py:76-83`). Therefore the configured constant is not an automatic startup-load path in current source.
- Admin DL reload accepts a requested bundle directory only under `MODEL_DIR / "dl"` (`serving/routers/health.py:277-300`). `DLModel.load` validates manifest/hash/lookback metadata and stores the directory, but defers executable artifact loading until the first prediction (`serving/dl_predictor.py:93-121,189-223`).
- Source declares required bundle files `model.pt`, `model_config.json`, `drug_vocab.json`, `edge_index.pt`, `feature_normalizer.pkl`, and `schema_version.json`, plus manifest name `MANIFEST.json` (`scripts/datasets/contracts.py:41-51`). These are declared names only; none was inspected.

## 2. Reproducible grep evidence

The source snapshot and clean source-tree state were recorded with:

```bash
git rev-parse HEAD
git status --short -- serving hana_app scripts config
```

Output:

```text
3d8d64e78601a3ff56dc38034a9da62853e6b656
```

The `git status` command produced no output, confirming that `serving/`, `hana_app/`, `scripts/`, and `config/` were unmodified.

The `DL_BUNDLE_DIR` consumer search was:

```bash
rg -n "DL_BUNDLE_DIR" serving hana_app scripts config
```

Output:

```text
config/settings.py:21:DL_BUNDLE_DIR   = Path(os.environ.get("DDI_DL_BUNDLE_DIR",  str(MODEL_DIR / "dl" / "current")))
```

The plan's exact Step 1 command was run from the worktree root:

```bash
grep -n "MODELS_DIR\|models/\|model_dir\|model_path\|ddi_model_\|stage_meta\|gat_model\|\.pkl\|\.joblib\|pickle\.load\|pickle\.loads\|joblib\.load" serving/predictor.py | head -30
```

Output:

```text
11:  - models/ddi_model_{partition}.pkl (XGBoost/LightGBM)
349:        sidecar 도 read_bytes → _verify_hash → pickle.loads 동일 패턴.
364:            obj = pickle.loads(content)
366:            logger.error("sidecar pickle.loads 실패: %s — %s", path, e)
378:            state = pickle.loads(content)
424:            model_dir = path.parent
430:                candidate = (model_dir / stored).resolve()
431:                # path traversal 방어 — model_dir 외부 경로 거부
433:                    candidate.relative_to(model_dir.resolve())
436:                        "%s 경로가 model_dir 외부 — 로드 거부: %s", key, candidate
456:                xgb_path = path.with_suffix(".xgb.pkl")
457:                lgb_path = path.with_suffix(".lgb.pkl")
493:                gat_model_path = path.parent / "gat_model.pt"
494:                if not gat_model_path.exists():
496:                        f"EnsembleTrainer3Way는 gat_model.pt가 필수입니다: {gat_model_path}"
500:                    self._gat_trainer = GATTrainer.load_gat(gat_model_path)
525:                    logger.info("GATTrainer 로드 완료: %s", gat_model_path)
627:    """계층 분류 모델 래퍼 — stage1_red.joblib + stage2_yellow.joblib + stage_meta.json.
629:    stage_meta.json 의 stage{1,2}_sha256 로 joblib 무결성 검증.
652:    def load(self, model_dir: str | Path) -> bool:
653:        model_dir = Path(model_dir)
654:        meta_path = model_dir / "stage_meta.json"
655:        p1 = model_dir / "stage1_red.joblib"
656:        p2 = model_dir / "stage2_yellow.joblib"
704:                    model_dir,
714:                    model_dir, _missing,
736:            self._stage1 = joblib.load(p1)
737:            bundle = joblib.load(p2)
760:                model_dir,
1187:        model_path: Optional[str | Path] = None,
```

The plan's exact Step 2 command was also run:

```bash
grep -n "class.*Predictor\|class.*MLModel\|class.*Wrapper\|class.*Constant" serving/predictor.py hana_app/core/hierarchical_runner.py | head -10
```

Output:

```text
serving/predictor.py:305:class MLModel:
serving/predictor.py:471:                        class _EnsembleWrapper:
serving/predictor.py:626:class HierarchicalPredictor:
serving/predictor.py:1179:class HybridPredictor:
hana_app/core/hierarchical_runner.py:40:class _ConstantNegativeStage1:
```

The complete static loader-site search used for this report was:

```bash
rg --sort path -n "json\.(load|loads)|(?:pickle|_pk)\.(load|loads)|joblib\.load|torch(?:\.jit)?\.load|graph_builder = GraphBuilder\.load|read_bytes\(|read_text\(" \
  serving/predictor.py serving/dl_predictor.py hana_app/core/hierarchical_runner.py scripts/train/gat_trainer.py scripts/features/graph_builder.py
```

Relevant transitive-call and deserialization output:

```text
serving/predictor.py:364:            obj = pickle.loads(content)
serving/predictor.py:378:            state = pickle.loads(content)
serving/predictor.py:467:                        xgb_state = _pk.loads(xgb_content)
serving/predictor.py:468:                        lgb_state = _pk.loads(lgb_content)
serving/predictor.py:506:                        meta = json.loads(meta_path.read_text())
serving/predictor.py:667:            self._meta = json.loads(meta_path.read_text())
serving/predictor.py:736:            self._stage1 = joblib.load(p1)
serving/predictor.py:737:            bundle = joblib.load(p2)
serving/dl_predictor.py:205:        model = torch.jit.load(str(self._bundle_dir / "model.pt"), map_location=device)
serving/dl_predictor.py:208:        edge_index = torch.load(
serving/dl_predictor.py:213:        feature_normalizer = pickle.loads(
serving/dl_predictor.py:241:        config = json.loads(path.read_text(encoding="utf-8"))
serving/dl_predictor.py:263:        raw = json.loads(path.read_text(encoding="utf-8"))
scripts/train/gat_trainer.py:312:        payload = pickle.loads(content)
scripts/train/gat_trainer.py:321:        graph_builder = GraphBuilder.load(path.parent)
scripts/features/graph_builder.py:233:        saved = torch.load(graph_path, weights_only=False)
scripts/features/graph_builder.py:239:            obj._meta = json.loads(meta_path.read_text())
```

## 3. Deserialization inventory

| Profile/component | Executing site | Source-visible object shape and guard |
|---|---|---|
| Tabular main | `serving/predictor.py:375-378` | Reads bytes once, verifies adjacent SHA-256, then `pickle.loads`; requires a dict. |
| Tabular scaler/selector | `serving/predictor.py:345-369,424-448` | Confined relative sidecar, adjacent SHA-256, then `pickle.loads`; no post-load type allowlist. |
| Tabular XGB/LGB ensemble sidecars | `serving/predictor.py:454-468` | Fixed sibling names, adjacent SHA-256, then two pickle loads; code expects dicts containing `model`. |
| Tabular GAT submodel and graph | `serving/predictor.py:491-500`; `scripts/train/gat_trainer.py:296-329`; `scripts/features/graph_builder.py:208-240` | The fixed `gat_model.pt` sibling is hash-verified, then pickle-loaded; its payload includes state, parameters, and an optional calibrator, while `GATModel` is reconstructed from source. `GATTrainer.load_gat` then calls `GraphBuilder.load`, which separately hash-verifies `gat_graph.pt` before `torch.load(graph_path, weights_only=False)` and expects `data` and `drug_to_idx`. |
| Hierarchical metadata | `serving/predictor.py:652-735` | Parses `stage_meta.json`; validates label/semantic/schema/hash state before joblib. JSON parsing is non-pickle but still reads artifact metadata at runtime. |
| Hierarchical Stage 1 | `serving/predictor.py:736` | `joblib.load(stage1_red.joblib)` after hash verification. |
| Hierarchical Stage 2 | `serving/predictor.py:737-740` | `joblib.load(stage2_yellow.joblib)`; code expects `model`, `encoder`, and `classes_present`. |
| DL metadata load | `serving/dl_predictor.py:93-121` | Manifest/hash/lookback validation only; no Torch or pickle load at `DLModel.load`. |
| DL lazy runtime | `serving/dl_predictor.py:189-215` | Revalidates bundle, then `torch.jit.load(model.pt)`, `torch.load(edge_index.pt, weights_only=True)`, and `pickle.loads(feature_normalizer.pkl)`. It also parses model/vocabulary JSON. |

Cryptographic hashes detect byte changes but do not make pickle/joblib payloads, or the GAT graph loaded by `torch.load(..., weights_only=False)`, safe. These formats can execute import/reconstruction behavior during loading; matching an adjacent digest establishes integrity against that digest, not trusted provenance or safe deserialization. Only trusted, provenance-controlled artifacts should reach these sites.

## 4. Pickle/joblib module-path risks

### 4.1 Confirmed project-local class reference: `_ConstantNegativeStage1` - high

When Stage 1 cannot train, current training creates `_ConstantNegativeStage1` and directly `joblib.dump`s it (`hana_app/core/hierarchical_runner.py:512-517,603-606`). The class is intentionally module-level (`hana_app/core/hierarchical_runner.py:40-59`), so such a joblib payload can encode `hana_app.core.hierarchical_runner._ConstantNegativeStage1`. Renaming the class, moving the module, removing the project root from importability, or loading under a different package name can break old artifacts with import/attribute resolution errors. Preserve that import path or supply a compatibility symbol before refactoring.

### 4.2 Stale plan assertion corrected: `_EnsembleWrapper` is not a bundle class

The plan labels `_EnsembleWrapper` as a critical pickle reference at `serving.predictor.MLModel.load.<locals>._EnsembleWrapper`. Current source does not serialize or deserialize this class. It defines and instantiates the local adapter only after the two sidecar dictionaries have already been unpickled (`serving/predictor.py:454-486`). Therefore moving this local class may change runtime behavior, but it is not, based on current source, an old-bundle module-path compatibility obligation. The actual pickle dependencies are the models stored under `xgb_state["model"]` and `lgb_state["model"]`.

### 4.3 External estimator paths and versions - high

- Tabular single and ensemble sidecars store fitted `xgboost.XGBClassifier` and `lightgbm.LGBMClassifier` objects (`scripts/train/trainer.py:132-170,183-225,272-285`). Their pickle globals depend on installed XGBoost/LightGBM module paths and compatible library versions.
- Hierarchical joblibs store XGBoost classifiers and a scikit-learn `LabelEncoder` (`hana_app/core/hierarchical_runner.py:490-492,542-543,568,600-613`). Stage 1 may instead store the project-local constant class above. The Windows offline environment must contain compatible packages and the project module path.
- The GAT pickle can contain a scikit-learn `LogisticRegression` calibrator; the neural model class itself is rebuilt from `scripts.train.gat_model.GATModel` after the payload is loaded (`scripts/train/gat_trainer.py:273-286,312-328`). The subsequent graph load uses `torch.load(..., weights_only=False)` because `torch_geometric.data.Data` is pickle-serialized (`scripts/features/graph_builder.py:230-233`), so PyTorch, PyTorch Geometric, and scikit-learn compatibility matter and the graph has the same trusted-provenance requirement despite its preceding hash check.

Exact module globals are artifact-dependent and cannot be proven without byte inspection. This report therefore names source-visible object types and does not claim the precise globals present in deployed bytes.

### 4.4 Scaler, selector, and DL normalizer - conditional

Current `FeatureNormalizer.save` and `FeatureSelector.save` write plain dictionaries, which reduces project-class path coupling (`scripts/features/normalizer.py:109-118`; `scripts/features/selector.py:137-148`). The serving sidecar loader nevertheless accepts any successfully unpickled object and performs no type allowlist, so an alternate artifact could still contain class globals.

The current sparse-linear DL exporter also writes a plain dictionary as `feature_normalizer.pkl` (`scripts/datasets/export_sparse_linear_bundle.py:182-183`), but `DLModel` likewise accepts any unpickled object. Treat the low module-path risk as a property of the current writer, not of the loader contract or of uninspected deployed bytes.

### 4.5 Import-root and Windows parity - high for project-local globals

`serving/predictor.py` inserts the repository root into `sys.path` at several lazy-import sites (`:173`, `:215`, `:787`, `:1266`), while `hana_app/core/hierarchical_runner.py:23-26` does the same before its absolute `scripts.etl` import. This can make project modules importable, but it does not translate a module name embedded in a pickle. A payload naming `hana_app.core.hierarchical_runner._ConstantNegativeStage1` still requires that exact module and attribute at load time. Python 3.12 dev/prod parity and the Windows closed-network package layout therefore need an explicit compatibility check before any relocation or dependency upgrade.

## 5. Byte and JSON inspection status

**Not performed.** No LO approval for artifact access was requested or assumed because Task 2 was explicitly constrained to static source analysis. No `pickletools.dis` or `pickletools.genops` was run, even though those tools do not execute pickle opcodes. No JSON metadata under any model/bundle directory was read.

Consequently, this report does not assert deployed values for feature names, thresholds, labels, hashes, schema versions, run IDs, lookback days, estimator versions, dtypes, or pickle global opcodes. It records only the paths, fields, checks, and object types visible in current source.

## 6. Risks and follow-up boundary

1. `_ConstantNegativeStage1` is a confirmed project-local compatibility path for degraded hierarchical Stage 1 artifacts.
2. XGBoost, LightGBM, scikit-learn, and PyTorch serialization compatibility is artifact- and version-dependent; current source cannot establish deployed byte-level globals.
3. Hash verification provides integrity, not safe deserialization. This includes the transitive GAT graph path: `GATTrainer.load_gat` calls `GraphBuilder.load`, which verifies `gat_graph.pt` and then executes `torch.load(..., weights_only=False)`. An attacker able to replace both artifact and trusted hash can still reach executing loaders.
4. `DDI_DL_BUNDLE_DIR` is configured but is not wired into current startup loading. Treating it as active startup behavior would be stale.
5. The next freeze-safe step is Phase 0A Task 3 review of the three Phase 0A reports. Any future artifact-byte or model JSON inspection requires explicit OpenCode LO approval and a separately bounded procedure.
