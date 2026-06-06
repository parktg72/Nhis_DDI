"""Sparse CSR training smoke for full reference-day datasets."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import sys
from time import perf_counter
from typing import Sequence

import numpy as np
from scipy import sparse

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.ops.mlp_smoke_train import (
    _precision,
    _recall,
    _roc_auc,
    compute_pos_weight,
    stratified_train_val_indices,
)


@dataclass(frozen=True)
class SparseTrainResult:
    train_loss_final: float
    val_auc: float
    val_pr_auc: float
    val_best_f1: float
    val_best_threshold: float
    val_precision: float
    val_recall: float
    val_precision_at_top1_pct: float
    val_precision_at_top5_pct: float
    val_recall_at_top1_pct: float
    val_recall_at_top5_pct: float
    n_train: int
    n_val: int
    n_positive_train: int
    n_positive_val: int
    elapsed_sec: float


def _torch():
    import torch

    return torch


def _nn():
    import torch.nn as nn

    return nn


def train_sparse_linear_smoke(
    X: sparse.csr_matrix,
    y: np.ndarray,
    *,
    epochs: int = 5,
    batch_size: int = 2048,
    lr: float = 1e-3,
    val_fraction: float = 0.2,
    seed: int = 42,
    device: str = "cpu",
) -> dict:
    if not sparse.isspmatrix_csr(X):
        X = X.tocsr()
    y = np.asarray(y, dtype=np.int64)
    start = perf_counter()
    _seed_everything(seed)
    torch = _torch()
    nn = _nn()

    train_idx, val_idx = stratified_train_val_indices(y, val_fraction=val_fraction, seed=seed)
    model = nn.Linear(X.shape[1], 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    pos_weight = torch.tensor(
        [compute_pos_weight(y[train_idx])],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    final_loss = 0.0
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        order = np.array(train_idx, copy=True)
        rng.shuffle(order)
        model.train()
        for start_idx in range(0, len(order), batch_size):
            batch_indices = order[start_idx : start_idx + batch_size]
            xb = _csr_rows_to_tensor(X, batch_indices, device=device)
            yb = torch.tensor(y[batch_indices].reshape(-1, 1), dtype=torch.float32, device=device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().cpu().item())

    scores = _predict_scores(model, X, val_idx, batch_size=batch_size, device=device)
    y_val = y[val_idx].astype(np.int64)
    pr_metrics = _precision_recall_metrics(y_val, scores)
    result = SparseTrainResult(
        train_loss_final=round(final_loss, 6),
        val_auc=round(_roc_auc(y_val, scores), 6),
        val_pr_auc=round(pr_metrics["pr_auc"], 6),
        val_best_f1=round(pr_metrics["best_f1"], 6),
        val_best_threshold=round(pr_metrics["best_threshold"], 6),
        val_precision=round(_precision(y_val, scores), 6),
        val_recall=round(_recall(y_val, scores), 6),
        val_precision_at_top1_pct=round(pr_metrics["precision_at_top1_pct"], 6),
        val_precision_at_top5_pct=round(pr_metrics["precision_at_top5_pct"], 6),
        val_recall_at_top1_pct=round(pr_metrics["recall_at_top1_pct"], 6),
        val_recall_at_top5_pct=round(pr_metrics["recall_at_top5_pct"], 6),
        n_train=int(len(train_idx)),
        n_val=int(len(val_idx)),
        n_positive_train=int(y[train_idx].sum()),
        n_positive_val=int(y[val_idx].sum()),
        elapsed_sec=round(perf_counter() - start, 3),
    )
    return asdict(result)


def train_sparse_linear_temporal_smoke(
    X_train: sparse.csr_matrix,
    y_train: np.ndarray,
    X_val: sparse.csr_matrix,
    y_val: np.ndarray,
    *,
    epochs: int = 20,
    batch_size: int = 2048,
    lr: float = 1e-3,
    seed: int = 42,
    device: str = "cpu",
) -> dict:
    if X_train.shape[1] != X_val.shape[1]:
        raise ValueError("train and val input_dim must match")
    if not sparse.isspmatrix_csr(X_train):
        X_train = X_train.tocsr()
    if not sparse.isspmatrix_csr(X_val):
        X_val = X_val.tocsr()
    y_train = np.asarray(y_train, dtype=np.int64)
    y_val = np.asarray(y_val, dtype=np.int64)

    start = perf_counter()
    _seed_everything(seed)
    torch = _torch()
    nn = _nn()
    model = nn.Linear(X_train.shape[1], 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    pos_weight = torch.tensor([compute_pos_weight(y_train)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    final_loss = 0.0
    rng = np.random.default_rng(seed)
    train_indices = np.arange(X_train.shape[0])
    for _ in range(epochs):
        order = np.array(train_indices, copy=True)
        rng.shuffle(order)
        model.train()
        for start_idx in range(0, len(order), batch_size):
            batch_indices = order[start_idx : start_idx + batch_size]
            xb = _csr_rows_to_tensor(X_train, batch_indices, device=device)
            yb = torch.tensor(y_train[batch_indices].reshape(-1, 1), dtype=torch.float32, device=device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().cpu().item())

    val_indices = np.arange(X_val.shape[0])
    scores = _predict_scores(model, X_val, val_indices, batch_size=batch_size, device=device)
    pr_metrics = _precision_recall_metrics(y_val, scores)
    result = SparseTrainResult(
        train_loss_final=round(final_loss, 6),
        val_auc=round(_roc_auc(y_val, scores), 6),
        val_pr_auc=round(pr_metrics["pr_auc"], 6),
        val_best_f1=round(pr_metrics["best_f1"], 6),
        val_best_threshold=round(pr_metrics["best_threshold"], 6),
        val_precision=round(_precision(y_val, scores), 6),
        val_recall=round(_recall(y_val, scores), 6),
        val_precision_at_top1_pct=round(pr_metrics["precision_at_top1_pct"], 6),
        val_precision_at_top5_pct=round(pr_metrics["precision_at_top5_pct"], 6),
        val_recall_at_top1_pct=round(pr_metrics["recall_at_top1_pct"], 6),
        val_recall_at_top5_pct=round(pr_metrics["recall_at_top5_pct"], 6),
        n_train=int(len(y_train)),
        n_val=int(len(y_val)),
        n_positive_train=int(y_train.sum()),
        n_positive_val=int(y_val.sum()),
        elapsed_sec=round(perf_counter() - start, 3),
    )
    return asdict(result)


def train_sparse_xgboost_temporal_smoke(
    X_train: sparse.csr_matrix,
    y_train: np.ndarray,
    X_val: sparse.csr_matrix,
    y_val: np.ndarray,
    *,
    n_estimators: int = 300,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    subsample: float = 0.8,
    colsample_bytree: float = 0.3,
    min_child_weight: float = 5.0,
    early_stopping_rounds: int = 20,
    seed: int = 42,
    n_jobs: int = -1,
    device: str = "cpu",
    save_model_path: str | Path | None = None,
) -> dict:
    if X_train.shape[1] != X_val.shape[1]:
        raise ValueError("train and val input_dim must match")
    if not sparse.isspmatrix_csr(X_train):
        X_train = X_train.tocsr()
    if not sparse.isspmatrix_csr(X_val):
        X_val = X_val.tocsr()
    y_train = np.asarray(y_train, dtype=np.int64)
    y_val = np.asarray(y_val, dtype=np.int64)

    start = perf_counter()
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise RuntimeError("xgboost is required for --model xgboost") from exc

    scale_pos_weight = compute_pos_weight(y_train)
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
        scale_pos_weight=scale_pos_weight,
        eval_metric=["auc", "aucpr"],
        tree_method="hist",
        device=device,
        random_state=seed,
        n_jobs=n_jobs,
        early_stopping_rounds=early_stopping_rounds,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    if save_model_path is not None:
        model_path = Path(save_model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(model_path)
    scores = model.predict_proba(X_val)[:, 1]
    pr_metrics = _precision_recall_metrics(y_val, scores)
    result = {
        "model": "xgboost",
        "train_loss_final": 0.0,
        "val_auc": round(_roc_auc(y_val, scores), 6),
        "val_pr_auc": round(pr_metrics["pr_auc"], 6),
        "val_best_f1": round(pr_metrics["best_f1"], 6),
        "val_best_threshold": round(pr_metrics["best_threshold"], 6),
        "val_precision": round(_precision(y_val, scores), 6),
        "val_recall": round(_recall(y_val, scores), 6),
        "val_precision_at_top1_pct": round(pr_metrics["precision_at_top1_pct"], 6),
        "val_precision_at_top5_pct": round(pr_metrics["precision_at_top5_pct"], 6),
        "val_recall_at_top1_pct": round(pr_metrics["recall_at_top1_pct"], 6),
        "val_recall_at_top5_pct": round(pr_metrics["recall_at_top5_pct"], 6),
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_positive_train": int(y_train.sum()),
        "n_positive_val": int(y_val.sum()),
        "elapsed_sec": round(perf_counter() - start, 3),
        "n_estimators": int(n_estimators),
        "n_estimators_used": _xgb_estimators_used(model),
        "max_depth": int(max_depth),
        "learning_rate": float(learning_rate),
        "subsample": float(subsample),
        "colsample_bytree": float(colsample_bytree),
        "min_child_weight": float(min_child_weight),
        "scale_pos_weight": round(float(scale_pos_weight), 6),
        "early_stopping_rounds": int(early_stopping_rounds),
        "xgb_device": device,
        "model_path": str(save_model_path) if save_model_path is not None else None,
    }
    return result


def run_sparse_training_smoke(
    dataset_dir: str | Path,
    *,
    epochs: int = 5,
    batch_size: int = 2048,
    seed: int = 42,
    device: str = "cpu",
) -> dict:
    dataset_path = Path(dataset_dir)
    X = sparse.load_npz(dataset_path / "X_csr.npz")
    y = np.load(dataset_path / "y.npy")
    metadata_path = dataset_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    train = train_sparse_linear_smoke(
        X,
        y,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
        device=device,
    )
    return {
        "dataset_dir": str(dataset_path),
        "n_patients": int(X.shape[0]),
        "input_dim": int(X.shape[1]),
        "nnz": int(X.nnz),
        "label_positive": int(y.sum()),
        "label_positive_rate_pct": _pct(int(y.sum()), len(y)),
        "label_source": metadata.get("label_source"),
        "therapeutic_dup_threshold": metadata.get("therapeutic_dup_threshold"),
        "epochs": epochs,
        "batch_size": batch_size,
        "seed": seed,
        "device": device,
        "train": train,
    }


def run_sparse_temporal_training_smoke(
    train_dataset_dir: str | Path,
    val_dataset_dir: str | Path,
    *,
    model: str = "linear",
    epochs: int = 20,
    batch_size: int = 2048,
    seed: int = 42,
    device: str = "cpu",
    xgb_n_estimators: int = 300,
    xgb_max_depth: int = 6,
    xgb_learning_rate: float = 0.1,
    xgb_subsample: float = 0.8,
    xgb_colsample_bytree: float = 0.3,
    xgb_min_child_weight: float = 5.0,
    xgb_early_stopping_rounds: int = 20,
    xgb_n_jobs: int = -1,
    save_model_path: str | Path | None = None,
) -> dict:
    train_path = Path(train_dataset_dir)
    val_path = Path(val_dataset_dir)
    X_train, y_train, train_metadata = _load_dataset(train_path)
    X_val, y_val, val_metadata = _load_dataset(val_path)
    _validate_temporal_metadata(X_train, X_val, train_metadata, val_metadata)
    overlap = _patient_overlap_stats(train_path, val_path)
    train = _train_temporal_model(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
        device=device,
        xgb_n_estimators=xgb_n_estimators,
        xgb_max_depth=xgb_max_depth,
        xgb_learning_rate=xgb_learning_rate,
        xgb_subsample=xgb_subsample,
        xgb_colsample_bytree=xgb_colsample_bytree,
        xgb_min_child_weight=xgb_min_child_weight,
        xgb_early_stopping_rounds=xgb_early_stopping_rounds,
        xgb_n_jobs=xgb_n_jobs,
        save_model_path=save_model_path,
    )
    return {
        "model": model,
        "train_dataset_dir": str(train_path),
        "val_dataset_dir": str(val_path),
        "n_train_dataset": int(X_train.shape[0]),
        "n_val_dataset": int(X_val.shape[0]),
        "input_dim": int(X_train.shape[1]),
        "train_label_positive": int(y_train.sum()),
        "train_label_positive_rate_pct": _pct(int(y_train.sum()), len(y_train)),
        "val_label_positive": int(y_val.sum()),
        "val_label_positive_rate_pct": _pct(int(y_val.sum()), len(y_val)),
        "label_source": train_metadata.get("label_source"),
        "multi_institution_threshold": train_metadata.get("multi_institution_threshold"),
        **overlap,
        "epochs": epochs,
        "batch_size": batch_size,
        "seed": seed,
        "device": device,
        "train": train,
    }


def _train_temporal_model(
    model: str,
    X_train: sparse.csr_matrix,
    y_train: np.ndarray,
    X_val: sparse.csr_matrix,
    y_val: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    seed: int,
    device: str,
    xgb_n_estimators: int,
    xgb_max_depth: int,
    xgb_learning_rate: float,
    xgb_subsample: float,
    xgb_colsample_bytree: float,
    xgb_min_child_weight: float,
    xgb_early_stopping_rounds: int,
    xgb_n_jobs: int,
    save_model_path: str | Path | None,
) -> dict:
    if model == "linear":
        result = train_sparse_linear_temporal_smoke(
            X_train,
            y_train,
            X_val,
            y_val,
            epochs=epochs,
            batch_size=batch_size,
            seed=seed,
            device=device,
        )
        result["model"] = "linear"
        return result
    if model == "xgboost":
        return train_sparse_xgboost_temporal_smoke(
            X_train,
            y_train,
            X_val,
            y_val,
            n_estimators=xgb_n_estimators,
            max_depth=xgb_max_depth,
            learning_rate=xgb_learning_rate,
            subsample=xgb_subsample,
            colsample_bytree=xgb_colsample_bytree,
            min_child_weight=xgb_min_child_weight,
            early_stopping_rounds=xgb_early_stopping_rounds,
            seed=seed,
            n_jobs=xgb_n_jobs,
            device=device,
            save_model_path=save_model_path,
        )
    raise ValueError(f"unsupported model: {model}")


def write_report(report: dict, output_dir: str | Path) -> tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    json_path = output_path / "sparse_training_smoke_report.json"
    md_path = output_path / "sparse_training_smoke_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_markdown_report(report), encoding="utf-8")
    return json_path, md_path


def _load_dataset(dataset_path: Path) -> tuple[sparse.csr_matrix, np.ndarray, dict]:
    X = sparse.load_npz(dataset_path / "X_csr.npz")
    y = np.load(dataset_path / "y.npy")
    metadata_path = dataset_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    return X.tocsr(), y, metadata


def _validate_temporal_metadata(
    X_train: sparse.csr_matrix,
    X_val: sparse.csr_matrix,
    train_metadata: dict,
    val_metadata: dict,
) -> None:
    if X_train.shape[1] != X_val.shape[1]:
        raise ValueError("train and val input_dim must match")
    if train_metadata.get("label_source") != val_metadata.get("label_source"):
        raise ValueError("train and val label_source must match")
    if train_metadata.get("multi_institution_threshold") != val_metadata.get("multi_institution_threshold"):
        raise ValueError("train and val multi_institution_threshold must match")
    if train_metadata.get("vocab_sha256") and val_metadata.get("vocab_sha256"):
        if train_metadata["vocab_sha256"] != val_metadata["vocab_sha256"]:
            raise ValueError("train and val vocab_sha256 must match")


def _patient_overlap_stats(train_path: Path, val_path: Path) -> dict:
    train_file = train_path / "patient_ids.npy"
    val_file = val_path / "patient_ids.npy"
    if not train_file.exists() or not val_file.exists():
        return {
            "patient_overlap_count": 0,
            "patient_overlap_train_rate_pct": 0.0,
            "patient_overlap_val_rate_pct": 0.0,
        }
    train_ids = set(np.load(train_file, allow_pickle=True).tolist())
    val_ids = set(np.load(val_file, allow_pickle=True).tolist())
    overlap_count = len(train_ids & val_ids)
    return {
        "patient_overlap_count": overlap_count,
        "patient_overlap_train_rate_pct": _pct(overlap_count, len(train_ids)),
        "patient_overlap_val_rate_pct": _pct(overlap_count, len(val_ids)),
    }


def _csr_rows_to_tensor(X: sparse.csr_matrix, indices: np.ndarray, *, device: str):
    torch = _torch()
    dense = X[indices].toarray().astype(np.float32, copy=False)
    return torch.tensor(dense, dtype=torch.float32, device=device)


def _predict_scores(
    model,
    X: sparse.csr_matrix,
    indices: np.ndarray,
    *,
    batch_size: int,
    device: str,
) -> np.ndarray:
    torch = _torch()
    scores: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start_idx in range(0, len(indices), batch_size):
            batch_indices = indices[start_idx : start_idx + batch_size]
            xb = _csr_rows_to_tensor(X, batch_indices, device=device)
            score = torch.sigmoid(model(xb)).detach().cpu().numpy().reshape(-1)
            scores.append(score)
    if not scores:
        return np.array([], dtype=np.float32)
    return np.concatenate(scores)


def _xgb_estimators_used(model) -> int:
    best_iteration = getattr(model, "best_iteration", None)
    if best_iteration is not None:
        return int(best_iteration) + 1
    booster = model.get_booster()
    return int(getattr(booster, "num_boosted_rounds")())


def _seed_everything(seed: int) -> None:
    torch = _torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _precision_recall_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(y_true.sum())
    if len(y_true) == 0 or positives == 0:
        return {
            "pr_auc": 0.0,
            "best_f1": 0.0,
            "best_threshold": 1.0,
            "precision_at_top1_pct": 0.0,
            "precision_at_top5_pct": 0.0,
            "recall_at_top1_pct": 0.0,
            "recall_at_top5_pct": 0.0,
        }

    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_y = y_true[order]
    tp = np.cumsum(sorted_y == 1)
    fp = np.cumsum(sorted_y == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / positives

    unique_last_indices = np.r_[np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]), len(sorted_scores) - 1]
    precision = precision[unique_last_indices]
    recall = recall[unique_last_indices]
    thresholds = sorted_scores[unique_last_indices]

    precision_curve = np.r_[1.0, precision]
    recall_curve = np.r_[0.0, recall]
    pr_auc = float(np.trapezoid(precision_curve, recall_curve))
    f1 = (2 * precision * recall) / np.maximum(precision + recall, 1e-12)
    best_idx = int(np.argmax(f1)) if len(f1) else 0
    return {
        "pr_auc": pr_auc,
        "best_f1": float(f1[best_idx]) if len(f1) else 0.0,
        "best_threshold": float(thresholds[best_idx]) if len(thresholds) else 1.0,
        **_top_k_metrics(y_true, scores),
    }


def _top_k_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(y_true.sum())
    if len(y_true) == 0 or positives == 0:
        return {
            "precision_at_top1_pct": 0.0,
            "precision_at_top5_pct": 0.0,
            "recall_at_top1_pct": 0.0,
            "recall_at_top5_pct": 0.0,
        }
    order = np.argsort(-scores, kind="mergesort")
    top1 = _precision_recall_at_k(y_true[order], positives, fraction=0.01)
    top5 = _precision_recall_at_k(y_true[order], positives, fraction=0.05)
    return {
        "precision_at_top1_pct": top1["precision"],
        "precision_at_top5_pct": top5["precision"],
        "recall_at_top1_pct": top1["recall"],
        "recall_at_top5_pct": top5["recall"],
    }


def _precision_recall_at_k(sorted_y: np.ndarray, positives: int, *, fraction: float) -> dict[str, float]:
    k = max(1, int(np.ceil(len(sorted_y) * fraction)))
    top_positive = int(sorted_y[:k].sum())
    return {
        "precision": float(top_positive / k),
        "recall": float(top_positive / positives) if positives else 0.0,
    }


def _pct(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round((numerator / denominator) * 100, 4)


def _markdown_report(report: dict) -> str:
    train = report["train"]
    if "train_dataset_dir" in report:
        header_lines = [
            "# Sparse Temporal Training Smoke Report",
            "",
            f"- train_dataset_dir: {report['train_dataset_dir']}",
            f"- val_dataset_dir: {report['val_dataset_dir']}",
            f"- n_train_dataset: {report['n_train_dataset']}",
            f"- n_val_dataset: {report['n_val_dataset']}",
            f"- input_dim: {report['input_dim']}",
            f"- train_label_positive_rate_pct: {report['train_label_positive_rate_pct']}",
            f"- val_label_positive_rate_pct: {report['val_label_positive_rate_pct']}",
            f"- label_source: {report['label_source']}",
            f"- multi_institution_threshold: {report['multi_institution_threshold']}",
        ]
    else:
        header_lines = [
            "# Sparse Training Smoke Report",
            "",
            f"- dataset_dir: {report['dataset_dir']}",
            f"- n_patients: {report['n_patients']}",
            f"- input_dim: {report['input_dim']}",
            f"- label_positive: {report['label_positive']}",
            f"- label_positive_rate_pct: {report['label_positive_rate_pct']}",
            f"- label_source: {report['label_source']}",
            f"- therapeutic_dup_threshold: {report['therapeutic_dup_threshold']}",
        ]
    return "\n".join([
        *header_lines,
        "",
        "| metric | value |",
        "|---|---:|",
        *[f"| {key} | {value} |" for key, value in train.items()],
        "",
    ])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run sparse CSR linear training smoke.")
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--train-dataset-dir", default=None)
    parser.add_argument("--val-dataset-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--model", choices=["linear", "xgboost"], default="linear")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--xgb-n-estimators", type=int, default=300)
    parser.add_argument("--xgb-max-depth", type=int, default=6)
    parser.add_argument("--xgb-learning-rate", type=float, default=0.1)
    parser.add_argument("--xgb-subsample", type=float, default=0.8)
    parser.add_argument("--xgb-colsample-bytree", type=float, default=0.3)
    parser.add_argument("--xgb-min-child-weight", type=float, default=5.0)
    parser.add_argument("--xgb-early-stopping-rounds", type=int, default=20)
    parser.add_argument("--xgb-n-jobs", type=int, default=-1)
    parser.add_argument("--save-model-path", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.train_dataset_dir and args.val_dataset_dir:
        report = run_sparse_temporal_training_smoke(
            args.train_dataset_dir,
            args.val_dataset_dir,
            model=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            device=args.device,
            xgb_n_estimators=args.xgb_n_estimators,
            xgb_max_depth=args.xgb_max_depth,
            xgb_learning_rate=args.xgb_learning_rate,
            xgb_subsample=args.xgb_subsample,
            xgb_colsample_bytree=args.xgb_colsample_bytree,
            xgb_min_child_weight=args.xgb_min_child_weight,
            xgb_early_stopping_rounds=args.xgb_early_stopping_rounds,
            xgb_n_jobs=args.xgb_n_jobs,
            save_model_path=args.save_model_path,
        )
    else:
        if args.dataset_dir is None:
            raise SystemExit("--dataset-dir is required unless --train-dataset-dir and --val-dataset-dir are set")
        report = run_sparse_training_smoke(
            args.dataset_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            device=args.device,
        )
    output_dir = args.output_dir or args.dataset_dir
    json_path, md_path = write_report(report, output_dir)
    print(f"[OK] wrote {json_path}")
    print(f"[OK] wrote {md_path}")
    print(f"val_auc={report['train']['val_auc']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
