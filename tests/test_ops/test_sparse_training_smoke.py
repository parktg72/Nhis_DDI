from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse


def test_sparse_linear_smoke_trains_and_reports_metrics() -> None:
    from scripts.ops.sparse_training_smoke import train_sparse_linear_smoke

    rng = np.random.default_rng(42)
    rows = []
    cols = []
    for row in range(80):
        active = rng.choice(np.arange(1, 12), size=3, replace=False)
        rows.extend([row] * len(active))
        cols.extend(active.tolist())
    X = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(80, 12),
        dtype=np.float32,
    )
    y = np.array([1] * 20 + [0] * 60, dtype=np.int8)

    result = train_sparse_linear_smoke(
        X,
        y,
        epochs=2,
        batch_size=16,
        seed=42,
        device="cpu",
    )

    assert result["n_train"] == 64
    assert result["n_val"] == 16
    assert result["n_positive_train"] > 0
    assert result["n_positive_val"] > 0
    assert result["train_loss_final"] >= 0
    assert 0 <= result["val_auc"] <= 1
    assert 0 <= result["val_pr_auc"] <= 1
    assert 0 <= result["val_best_f1"] <= 1
    assert 0 <= result["val_best_threshold"] <= 1
    assert 0 <= result["val_precision_at_top1_pct"] <= 1
    assert 0 <= result["val_precision_at_top5_pct"] <= 1
    assert 0 <= result["val_recall_at_top1_pct"] <= 1
    assert 0 <= result["val_recall_at_top5_pct"] <= 1


def test_run_sparse_training_smoke_loads_dataset(tmp_path) -> None:
    import json
    from scripts.ops.sparse_training_smoke import run_sparse_training_smoke

    X = sparse.eye(10, 6, dtype=np.float32, format="csr")
    y = np.array([1, 1] + [0] * 8, dtype=np.int8)
    sparse.save_npz(tmp_path / "X_csr.npz", X)
    np.save(tmp_path / "y.npy", y)
    np.save(tmp_path / "patient_ids.npy", np.array([f"P{i}" for i in range(10)], dtype=object), allow_pickle=True)
    (tmp_path / "metadata.json").write_text(
        json.dumps({"label_source": "therapeutic_duplication", "therapeutic_dup_threshold": 6}),
        encoding="utf-8",
    )

    report = run_sparse_training_smoke(
        tmp_path,
        epochs=1,
        batch_size=4,
        seed=42,
        device="cpu",
    )

    assert report["dataset_dir"] == str(tmp_path)
    assert report["n_patients"] == 10
    assert report["input_dim"] == 6
    assert report["label_positive"] == 2
    assert report["label_source"] == "therapeutic_duplication"
    assert "train" in report


def test_run_sparse_temporal_training_smoke_uses_explicit_val_dataset(tmp_path) -> None:
    import json
    from scripts.ops.sparse_training_smoke import run_sparse_temporal_training_smoke

    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    train_dir.mkdir()
    val_dir.mkdir()
    X_train = sparse.eye(12, 6, dtype=np.float32, format="csr")
    y_train = np.array([1, 1, 1] + [0] * 9, dtype=np.int8)
    X_val = sparse.eye(8, 6, dtype=np.float32, format="csr")
    y_val = np.array([1, 1] + [0] * 6, dtype=np.int8)
    for path, X, y in [(train_dir, X_train, y_train), (val_dir, X_val, y_val)]:
        sparse.save_npz(path / "X_csr.npz", X)
        np.save(path / "y.npy", y)
        np.save(path / "patient_ids.npy", np.array([f"P{i}" for i in range(len(y))], dtype=object), allow_pickle=True)
        (path / "metadata.json").write_text(
            json.dumps({"label_source": "multi_institution", "multi_institution_threshold": 6}),
            encoding="utf-8",
        )

    report = run_sparse_temporal_training_smoke(
        train_dir,
        val_dir,
        epochs=1,
        batch_size=4,
        seed=42,
        device="cpu",
    )

    assert report["train_dataset_dir"] == str(train_dir)
    assert report["val_dataset_dir"] == str(val_dir)
    assert report["n_train_dataset"] == 12
    assert report["n_val_dataset"] == 8
    assert report["input_dim"] == 6
    assert report["label_source"] == "multi_institution"
    assert report["train"]["n_train"] == 12
    assert report["train"]["n_val"] == 8
    assert report["patient_overlap_count"] == 8
    assert report["patient_overlap_val_rate_pct"] == 100.0

    from scripts.ops.sparse_training_smoke import write_report

    json_path, md_path = write_report(report, tmp_path / "report")
    assert json_path.exists()
    assert md_path.exists()


def test_xgboost_temporal_smoke_trains_when_available() -> None:
    pytest.importorskip("xgboost")
    from scripts.ops.sparse_training_smoke import train_sparse_xgboost_temporal_smoke

    rng = np.random.default_rng(42)
    X_train = sparse.random(40, 20, density=0.15, format="csr", random_state=42, dtype=np.float32)
    X_val = sparse.random(20, 20, density=0.15, format="csr", random_state=43, dtype=np.float32)
    y_train = np.array([1] * 12 + [0] * 28, dtype=np.int8)
    y_val = np.array([1] * 6 + [0] * 14, dtype=np.int8)
    rng.shuffle(y_train)
    rng.shuffle(y_val)

    result = train_sparse_xgboost_temporal_smoke(
        X_train,
        y_train,
        X_val,
        y_val,
        n_estimators=5,
        max_depth=2,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.8,
        min_child_weight=1,
        early_stopping_rounds=2,
        seed=42,
        n_jobs=1,
    )

    assert result["model"] == "xgboost"
    assert result["n_train"] == 40
    assert result["n_val"] == 20
    assert result["n_positive_train"] == 12
    assert result["n_positive_val"] == 6
    assert result["scale_pos_weight"] > 0
    assert 0 <= result["val_auc"] <= 1
    assert 0 <= result["val_pr_auc"] <= 1
    assert result["n_estimators_used"] >= 1


def test_run_sparse_temporal_training_smoke_routes_xgboost(tmp_path) -> None:
    pytest.importorskip("xgboost")
    import json
    from scripts.ops.sparse_training_smoke import run_sparse_temporal_training_smoke

    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    train_dir.mkdir()
    val_dir.mkdir()
    X_train = sparse.random(30, 12, density=0.2, format="csr", random_state=44, dtype=np.float32)
    X_val = sparse.random(18, 12, density=0.2, format="csr", random_state=45, dtype=np.float32)
    y_train = np.array([1] * 9 + [0] * 21, dtype=np.int8)
    y_val = np.array([1] * 5 + [0] * 13, dtype=np.int8)
    for path, X, y in [(train_dir, X_train, y_train), (val_dir, X_val, y_val)]:
        sparse.save_npz(path / "X_csr.npz", X)
        np.save(path / "y.npy", y)
        np.save(path / "patient_ids.npy", np.array([f"{path.name}_P{i}" for i in range(len(y))], dtype=object), allow_pickle=True)
        (path / "metadata.json").write_text(
            json.dumps({"label_source": "multi_institution", "multi_institution_threshold": 6}),
            encoding="utf-8",
        )

    report = run_sparse_temporal_training_smoke(
        train_dir,
        val_dir,
        model="xgboost",
        xgb_n_estimators=5,
        xgb_max_depth=2,
        xgb_min_child_weight=1,
        xgb_early_stopping_rounds=2,
        xgb_n_jobs=1,
    )

    assert report["model"] == "xgboost"
    assert report["train"]["model"] == "xgboost"
    assert report["patient_overlap_count"] == 0
