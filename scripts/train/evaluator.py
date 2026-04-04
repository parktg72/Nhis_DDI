"""
모델 평가 모듈

평가 지표:
  - AUC-ROC     : 전체 판별력
  - AUC-PR      : 불균형 클래스에서 정밀도-재현율 균형
  - Recall      : Red 환자 탐지율 (핵심 — 미탐지 = 의료 위해)
  - Precision   : Red 예측 정확도
  - F1-score    : Recall/Precision 조화 평균
  - 혼동행렬    : TP/FP/FN/TN
  - 임계값 최적화: Recall ≥ 목표값을 만족하는 최소 임계값 탐색

임계값 전략:
  Recall 최우선 → Precision 희생을 감수하되
  Precision이 50% 미만이면 경고 (과다경보)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """단일 평가 결과."""
    split: str                  # train / val / test
    auc_roc: float = 0.0
    auc_pr: float = 0.0
    recall: float = 0.0
    precision: float = 0.0
    f1: float = 0.0
    accuracy: float = 0.0
    threshold: float = 0.5
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    n_positive: int = 0
    n_negative: int = 0
    warnings: list[str] = field(default_factory=list)

    min_recall: float = 0.90   # 합격 기준 — TrainConfig에서 주입
    min_auc:    float = 0.85   # 합격 기준 — TrainConfig에서 주입

    @property
    def passed_recall(self) -> bool:
        return self.recall >= self.min_recall

    @property
    def passed_auc(self) -> bool:
        return self.auc_roc >= self.min_auc

    @property
    def passed(self) -> bool:
        return self.passed_recall and self.passed_auc

    def print(self) -> None:
        status = "PASS" if self.passed else "FAIL"
        print(f"\n[{self.split.upper()}] {status}")
        print(f"  AUC-ROC  : {self.auc_roc:.4f}  {'✓' if self.passed_auc else '✗ (<0.85)'}")
        print(f"  AUC-PR   : {self.auc_pr:.4f}")
        print(f"  Recall   : {self.recall:.4f}  {'✓' if self.passed_recall else '✗ (<0.90)'}")
        print(f"  Precision: {self.precision:.4f}")
        print(f"  F1-score : {self.f1:.4f}")
        print(f"  Threshold: {self.threshold:.3f}")
        print(f"  TP={self.tp}, FP={self.fp}, FN={self.fn}, TN={self.tn}")
        for w in self.warnings:
            print(f"  [경고] {w}")


def _try_import_sklearn():
    try:
        from sklearn.metrics import (
            roc_auc_score, average_precision_score,
            precision_recall_curve, confusion_matrix,
        )
        return roc_auc_score, average_precision_score, precision_recall_curve, confusion_matrix
    except ImportError:
        return None


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    split: str = "val",
) -> EvalResult:
    """
    이진 분류 평가 지표 계산.

    Parameters
    ----------
    y_true : 실제 레이블 (0/1)
    y_prob : 양성 클래스 확률
    threshold : 분류 임계값
    """
    result = EvalResult(split=split)
    result.n_positive = int(y_true.sum())
    result.n_negative = int((y_true == 0).sum())
    result.threshold = threshold

    sklearn_fns = _try_import_sklearn()

    if sklearn_fns is not None:
        roc_auc_score, average_precision_score, precision_recall_curve, confusion_matrix = sklearn_fns
        if len(np.unique(y_true)) < 2:
            result.auc_roc = 0.0
            result.auc_pr = 0.0
        else:
            result.auc_roc = float(roc_auc_score(y_true, y_prob))
            result.auc_pr  = float(average_precision_score(y_true, y_prob))
    else:
        # sklearn 없을 때 numpy 기반 근사 AUC (trapezoidal)
        result.auc_roc = _numpy_auc_roc(y_true, y_prob)
        result.auc_pr  = 0.0

    # 임계값 기반 분류
    y_pred = (y_prob >= threshold).astype(int)

    if sklearn_fns is not None:
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            result.tn, result.fp, result.fn, result.tp = cm.ravel()
        else:
            _fill_cm_manual(result, y_true, y_pred)
    else:
        _fill_cm_manual(result, y_true, y_pred)

    total = result.tp + result.fp + result.fn + result.tn
    result.recall    = result.tp / (result.tp + result.fn) if (result.tp + result.fn) > 0 else 0.0
    result.precision = result.tp / (result.tp + result.fp) if (result.tp + result.fp) > 0 else 0.0
    result.accuracy  = (result.tp + result.tn) / total if total > 0 else 0.0
    denom = result.recall + result.precision
    result.f1 = 2 * result.recall * result.precision / denom if denom > 0 else 0.0

    # 경고
    if result.precision < 0.50 and result.precision > 0:
        result.warnings.append(f"Precision {result.precision:.1%} < 50% (과다경보 위험)")
    if result.fn > 0:
        result.warnings.append(f"미탐지(FN) {result.fn}건 — 실제 Red 환자 누락")

    return result


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_recall: float = 0.90,
    split: str = "val",
) -> tuple[float, EvalResult]:
    """
    Recall ≥ min_recall 을 만족하는 임계값 중 Precision 최대화.

    Returns
    -------
    (optimal_threshold, EvalResult)
    """
    sklearn_fns = _try_import_sklearn()

    if sklearn_fns is not None:
        _, _, precision_recall_curve, _ = sklearn_fns
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
        # precision_recall_curve는 마지막에 1,0 추가 → thresholds 길이 = len(precisions)-1
        best_thresh = 0.5
        best_precision = 0.0
        for prec, rec, thresh in zip(precisions[:-1], recalls[:-1], thresholds):
            if rec >= min_recall and prec > best_precision:
                best_precision = prec
                best_thresh = thresh
    else:
        # numpy 기반: 0.01 간격 탐색
        best_thresh = 0.5
        best_precision = 0.0
        for t in np.arange(0.05, 0.95, 0.01):
            y_pred = (y_prob >= t).astype(int)
            tp = int(((y_pred == 1) & (y_true == 1)).sum())
            fn = int(((y_pred == 0) & (y_true == 1)).sum())
            fp = int(((y_pred == 1) & (y_true == 0)).sum())
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            prec   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            if recall >= min_recall and prec > best_precision:
                best_precision = prec
                best_thresh = float(t)

    result = compute_metrics(y_true, y_prob, threshold=best_thresh, split=split)
    logger.info(
        "최적 임계값: %.3f → Recall=%.3f, Precision=%.3f",
        best_thresh, result.recall, result.precision,
    )
    return best_thresh, result


def evaluate_all_splits(
    y_true_tr: np.ndarray, y_prob_tr: np.ndarray,
    y_true_va: np.ndarray, y_prob_va: np.ndarray,
    y_true_te: np.ndarray, y_prob_te: np.ndarray,
    min_recall: float = 0.90,
    min_auc: float = 0.85,
) -> dict[str, EvalResult]:
    """Val로 임계값 최적화 후 Train/Test에 동일 임계값 적용."""
    thresh, val_result = find_optimal_threshold(y_true_va, y_prob_va, min_recall, "val")
    train_result = compute_metrics(y_true_tr, y_prob_tr, thresh, "train")
    test_result  = compute_metrics(y_true_te, y_prob_te, thresh, "test")
    for r in (val_result, train_result, test_result):
        r.min_recall = min_recall
        r.min_auc    = min_auc
    return {"train": train_result, "val": val_result, "test": test_result}


# ─────────────────────────────────────────────────────────────────────────────
# 내부 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

def _fill_cm_manual(result: EvalResult, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    result.tp = int(((y_pred == 1) & (y_true == 1)).sum())
    result.fp = int(((y_pred == 1) & (y_true == 0)).sum())
    result.fn = int(((y_pred == 0) & (y_true == 1)).sum())
    result.tn = int(((y_pred == 0) & (y_true == 0)).sum())


def _numpy_auc_roc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """sklearn 없을 때 numpy trapezoidal AUC."""
    thresholds = np.sort(np.unique(y_prob))[::-1]
    tprs, fprs = [0.0], [0.0]
    pos = y_true.sum()
    neg = len(y_true) - pos
    if pos == 0 or neg == 0:
        return 0.5
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        tprs.append(tp / pos)
        fprs.append(fp / neg)
    tprs.append(1.0)
    fprs.append(1.0)
    return float(np.trapezoid(tprs, fprs))
