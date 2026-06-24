"""계층 분류 평가 메트릭 리포트 — Task 5.

Stage 1 (이진) 과 Stage 2 (6-class) 에 대한 평가 지표를 계산하고
JSON + Markdown 으로 저장한다.

설계:
- `compute_hierarchical_metrics()` — 순수 함수, 예측 배열만 받아 dict 반환 (testable).
- `evaluate_hierarchical_bundle()` — 편의 래퍼. train_hierarchical 반환 번들 +
  검증 데이터를 받아 predict_risk 내부 호출 후 메트릭 계산.
- `save_metrics_report()` — dict → JSON + Markdown 파일 저장.

Stage 2 는 non-Red 샘플에 대해서만 정의되므로, 호출자가 y2_true / p2_probs 를
non-Red 서브셋으로 필터링해 전달해야 한다 (pure 함수 계약 명시).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from hana_app.core.hierarchical_runner import (
    STAGE2_LABELS,
    _normalize_stage2_local_predictions,
)


def compute_stage1_metrics(
    y_true: np.ndarray,
    p_red: np.ndarray,
) -> dict:
    """Stage 1 이진 분류 메트릭: PR-AUC, ROC-AUC, Brier.

    Parameters
    ----------
    y_true : np.ndarray
        이진 {0, 1} 정답. 최소 양/음 1 건 이상 필요.
    p_red : np.ndarray
        Red 예측 확률 [0, 1].

    Returns
    -------
    dict
        {pr_auc, roc_auc, brier, n_samples, n_positive}
    """
    y_true = np.asarray(y_true, dtype=int)
    p_red = np.asarray(p_red, dtype=float)
    if y_true.size != p_red.size:
        raise ValueError(
            f"y_true/p_red 길이 불일치: {y_true.size} vs {p_red.size}"
        )
    if np.unique(y_true).size < 2:
        raise ValueError("y_true 는 양/음 샘플을 모두 포함해야 함")

    return {
        "pr_auc": float(average_precision_score(y_true, p_red)),
        "roc_auc": float(roc_auc_score(y_true, p_red)),
        "brier": float(brier_score_loss(y_true, p_red)),
        "n_samples": int(y_true.size),
        "n_positive": int(y_true.sum()),
    }


def compute_stage2_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    stage2_labels: tuple[str, ...] = STAGE2_LABELS,
) -> dict:
    """Stage 2 다중 클래스 메트릭: macro F1, per-class P·R·F1, 6×6 confusion matrix.

    Parameters
    ----------
    y_true : np.ndarray
        정수 인코딩된 정답 라벨 (0..len(stage2_labels)-1).
    y_pred : np.ndarray
        정수 인코딩된 예측 라벨 (argmax of proba).
    stage2_labels : tuple[str, ...]
        인덱스별 라벨 이름. 기본 STAGE2_LABELS (6-class).

    Returns
    -------
    dict
        {macro_f1, per_class: {label: {precision, recall, f1, support}}, confusion_matrix, labels}
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if y_true.size != y_pred.size:
        raise ValueError(f"y_true/y_pred 길이 불일치: {y_true.size} vs {y_pred.size}")

    n_classes = len(stage2_labels)
    label_indices = list(range(n_classes))

    macro_f1 = float(f1_score(y_true, y_pred, labels=label_indices, average="macro", zero_division=0))
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=label_indices, zero_division=0,
    )
    per_class = {
        stage2_labels[i]: {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f[i]),
            "support": int(s[i]),
        }
        for i in range(n_classes)
    }
    cm = confusion_matrix(y_true, y_pred, labels=label_indices)
    return {
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "labels": list(stage2_labels),
        "n_samples": int(y_true.size),
    }


def compute_hierarchical_metrics(
    y1_true: np.ndarray,
    p1_red: np.ndarray,
    y2_true: np.ndarray,
    y2_pred: np.ndarray,
    stage2_labels: tuple[str, ...] = STAGE2_LABELS,
) -> dict:
    """Stage 1 + Stage 2 통합 메트릭.

    Stage 2 배열은 non-Red 서브셋에 대해서만 정의 (호출자가 필터링 책임).

    Returns
    -------
    {"stage1": {...}, "stage2": {...}}
    """
    return {
        "stage1": compute_stage1_metrics(y1_true, p1_red),
        "stage2": compute_stage2_metrics(y2_true, y2_pred, stage2_labels),
    }


def _format_markdown(report: dict) -> str:
    s1 = report["stage1"]
    s2 = report["stage2"]
    lines = [
        "# 계층 분류 평가 리포트",
        "",
        "## Stage 1 (Red 이진)",
        "",
        f"- 샘플 수: {s1['n_samples']:,} (양성 {s1['n_positive']:,})",
        f"- PR-AUC:  {s1['pr_auc']:.4f}",
        f"- ROC-AUC: {s1['roc_auc']:.4f}",
        f"- Brier:   {s1['brier']:.4f}",
        "",
        "## Stage 2 (6-class)",
        "",
        f"- 샘플 수: {s2['n_samples']:,}",
        f"- Macro F1: {s2['macro_f1']:.4f}",
        "",
        "### 클래스별 지표",
        "",
        "| 라벨 | Precision | Recall | F1 | Support |",
        "|---|---|---|---|---|",
    ]
    for lbl in s2["labels"]:
        m = s2["per_class"][lbl]
        if m["support"] == 0:
            # 등장 안 한 클래스 — 0.0 을 "zero performance" 로 오독 방지
            lines.append(f"| {lbl} | N/A | N/A | N/A | 0 |")
        else:
            lines.append(
                f"| {lbl} | {m['precision']:.4f} | {m['recall']:.4f} "
                f"| {m['f1']:.4f} | {m['support']:,} |"
            )
    lines += [
        "",
        "### Confusion Matrix (행=정답, 열=예측)",
        "",
        "| | " + " | ".join(s2["labels"]) + " |",
        "|" + "|".join(["---"] * (len(s2["labels"]) + 1)) + "|",
    ]
    for i, lbl in enumerate(s2["labels"]):
        row = " | ".join(str(v) for v in s2["confusion_matrix"][i])
        lines.append(f"| **{lbl}** | {row} |")
    return "\n".join(lines) + "\n"


def save_metrics_report(report: dict, output_dir: str | Path) -> dict[str, Path]:
    """JSON + Markdown 저장. 반환: 저장된 경로 dict."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "hierarchical_metrics.json"
    md_path = output_dir / "hierarchical_metrics.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    md_path.write_text(_format_markdown(report))
    return {"json": json_path, "markdown": md_path}


def evaluate_hierarchical_bundle(
    bundle: dict,
    X_val: np.ndarray,
    y1_val: np.ndarray,
    y2_val: np.ndarray,
    classes_present: list[int],
    y2_mask: np.ndarray | None = None,
) -> dict:
    """train_hierarchical 반환 번들 + 검증 데이터로 메트릭 계산 (편의 래퍼).

    Parameters
    ----------
    bundle : dict
        train_hierarchical 반환값 — stage1_model / stage2_model 포함.
    X_val : np.ndarray
        (n, n_features) 검증 피처.
    y1_val : np.ndarray
        (n,) Stage 1 이진 라벨 (0/1).
    y2_val : np.ndarray
        (n,) Stage 2 **global** 정수 라벨 (STAGE2_LABELS 인덱스). Red 샘플은
        의미 없으므로 y2_mask 로 제외.
    classes_present : list[int]
        stage2 모델이 학습한 클래스의 global STAGE2_LABELS 인덱스 리스트.
        `stage2_yellow.joblib['classes_present']` 에서 로드해 전달.
        local (stage2.predict 결과) → global 인덱스 매핑에 사용.
    y2_mask : np.ndarray, optional
        (n,) bool 배열, True 인 샘플만 Stage 2 메트릭에 반영 (보통 y1_val == 0).
        None 이면 y1_val == 0 을 기본 mask 로 사용.

    Returns
    -------
    dict — compute_hierarchical_metrics 과 동일 구조.
    """
    X_val = np.asarray(X_val)
    y1_val = np.asarray(y1_val, dtype=int)
    y2_val = np.asarray(y2_val, dtype=int)
    if y2_mask is None:
        y2_mask = y1_val == 0

    stage1 = bundle["stage1_model"]
    stage2 = bundle["stage2_model"]
    p1_red = stage1.predict_proba(X_val)[:, 1]

    X2 = X_val[y2_mask]
    y2_true = y2_val[y2_mask]

    if len(X2) > 0:
        y2_local = _normalize_stage2_local_predictions(stage2.predict(X2))
        # local index → global STAGE2_LABELS 인덱스 (단순 테이블 룩업)
        cp = np.asarray(classes_present, dtype=int)
        if y2_local.max(initial=-1) >= len(cp):
            raise ValueError(
                f"stage2.predict 가 local 인덱스 {y2_local.max()} 반환 — "
                f"classes_present 길이({len(cp)}) 초과. 잘못된 classes_present 전달."
            )
        y2_pred = cp[y2_local]
    else:
        y2_pred = np.array([], dtype=int)

    return compute_hierarchical_metrics(
        y1_true=y1_val, p1_red=p1_red,
        y2_true=y2_true, y2_pred=y2_pred,
    )
