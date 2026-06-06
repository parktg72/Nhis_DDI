# Extracted Sparse Training UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show prebuilt sparse training datasets from `data/datasets` in the desktop Page 3 workflow without mixing them into the existing `features_df` training path.

**Architecture:** Add a focused `hana_app.core.sparse_research` helper that discovers sparse dataset artifacts, summarizes metadata, finds reports, and builds safe smoke-training CLI commands. Page 3 renders those helpers in a separate Research/Smoke section; raw-to-dataset builds remain CLI-only.

**Tech Stack:** Python 3.12, Streamlit, pathlib, JSON, pytest, existing `scripts/ops/sparse_training_smoke.py`.

---

### Task 1: Sparse Dataset Registry Helper

**Files:**
- Create: `hana_app/core/sparse_research.py`
- Test: `tests/test_hana_app/test_sparse_research.py`

- [ ] **Step 1: Write failing tests**

Create tests that build temporary dataset directories with `metadata.json`, `X_csr.npz`, and `y.npy`. Verify that complete datasets are listed, incomplete directories are ignored, malformed metadata is reported as unavailable, labels/windows are summarized, smoke report paths are detected, and generated commands use `sys.executable`.

- [ ] **Step 2: Run tests to verify RED**

Run: `pytest tests/test_hana_app/test_sparse_research.py -q`

Expected: import failure for `hana_app.core.sparse_research`.

- [ ] **Step 3: Implement helper**

Implement:
- `PROJECT_ROOT`
- `DATASETS_ROOT = PROJECT_ROOT / "data" / "datasets"`
- `SparseDatasetSummary`
- `list_sparse_datasets(dataset_root=DATASETS_ROOT)`
- `dataset_display_rows(summaries)`
- `find_report_paths(dataset_dir)`
- `default_smoke_output_dir(dataset_dir, model="linear")`
- `build_smoke_command(dataset_dir, output_dir, python_executable=sys.executable, epochs=20, batch_size=2048, seed=42, device="cpu")`
- `lock_path_for(output_dir)`, `log_path_for(output_dir)`, and `read_log_tail(path, max_lines=30)`

- [ ] **Step 4: Run helper tests**

Run: `pytest tests/test_hana_app/test_sparse_research.py -q`

Expected: all tests pass.

### Task 2: Page 3 Research/Smoke Section

**Files:**
- Modify: `hana_app/pages/3_🤖_모델_학습.py`

- [ ] **Step 1: Add helper import**

Import the sparse research helper near the existing app imports.

- [ ] **Step 2: Render isolated section**

Add a `st.expander("추출 산출물 학습 (Research/Smoke)")` before the existing feature selection/training controls. The section lists `data/datasets` sparse artifacts, shows selected metadata/report content, and displays the exact smoke command. Do not write into `features_df`, `data_mode`, or `train_model()`.

- [ ] **Step 3: Blank state**

If no complete sparse datasets exist, show an info message that points to `data/datasets`.

### Task 3: Documentation Refresh

**Files:**
- Modify: `docs/superpowers/specs/2026-05-23-future-outcome-label-design.md`
- Modify: `data/reports/phase3_baseline_summary.md`

- [ ] **Step 1: Update stale December premise**

Replace statements that say 2024-12 Raw is unavailable with the current state: 2024-12 Raw is available and enables Nov->Dec temporal holdout.

- [ ] **Step 2: Preserve caution**

Keep language that avoids production or clinical claims until holdout metrics are generated and reviewed.

### Task 4: Verification

**Files:**
- Test: `tests/test_hana_app/test_sparse_research.py`
- Test: `tests/test_ops/test_sparse_training_smoke.py`

- [ ] **Step 1: Run focused tests**

Run: `pytest tests/test_hana_app/test_sparse_research.py tests/test_ops/test_sparse_training_smoke.py -q`

Expected: all selected tests pass, or any dependency-related skip/failure is recorded explicitly.

- [ ] **Step 2: Inspect diff**

Run: `git diff -- hana_app/core/sparse_research.py "hana_app/pages/3_🤖_모델_학습.py" docs/superpowers/specs/2026-05-23-future-outcome-label-design.md data/reports/phase3_baseline_summary.md tests/test_hana_app/test_sparse_research.py docs/superpowers/plans/2026-05-26-extracted-sparse-training-ui.md`

Expected: changes are scoped to sparse research UI and December premise refresh.
