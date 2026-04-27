"""계층 분류 StratifiedKFold CV — Task 6.

train_hierarchical 을 K 회 반복 평가해 (a) 메트릭의 폴드 간 변동, (b) τ 안정성을
가시화. 최종 τ 결정은 Task 4b 의 일이며 본 모듈은 *모델링 절차의 성능* 측정용.

주요 설계 (advisor 권고):
- 분층화 컬럼: 기본 `risk_level` (4-class). Red rare (~2-5%) 인 분포에서 폴드별
  Red 개수 보존이 핵심.
- 분류 지표 풀링: confusion matrix 는 폴드 합산 (pool counts), Stage 2 P/R/F1·
  macro F1 은 풀링된 CM 에서 유도 (rate-then-pool 의 분산-편향 회피).
- Stage 1 스칼라(PR-AUC/ROC-AUC/Brier) 는 폴드별 평균±표준편차 (rank-based 지표는
  폴드 간 보정 미보장 → 풀링 부적절).
- 폴드별 τ_red / τ_review 분산 노출. 변동이 크면 임계값 비안정 시그널.
"""
from __future__ import annotations

import tempfile
from collections.abc import Iterable
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from hana_app.core.hierarchical_metrics import (
    compute_stage1_metrics,
    compute_stage2_metrics,
)
from hana_app.core.hierarchical_runner import STAGE2_LABELS, train_hierarchical


def _encode_stage2_for_eval(
    df: pd.DataFrame, stage2_labels: Iterable[str] = STAGE2_LABELS,
) -> np.ndarray:
    """각 행의 (risk_level, yellow_subtype) → STAGE2_LABELS 인덱스.

    Red 는 Stage 2 대상 아님 → -1.
    Yellow + yellow_subtype 이 STAGE2_LABELS 에 있으면 해당 인덱스.
    Yellow + Y_OTHER 등 매핑 불가 → No_Alert 인덱스로 폴백.
    Green / Normal → No_Alert.
    """
    labels_list = list(stage2_labels)
    no_alert_idx = labels_list.index("No_Alert")

    def _enc(row):
        rl = row["risk_level"]
        if rl == "Red":
            return -1
        if rl == "Yellow":
            ys = row.get("yellow_subtype")
            if ys in labels_list:
                return labels_list.index(ys)
        return no_alert_idx

    return df.apply(_enc, axis=1).to_numpy()


def _pool_stage2_metrics(
    pooled_y_true: np.ndarray,
    pooled_y_pred: np.ndarray,
    stage2_labels: tuple[str, ...] = STAGE2_LABELS,
) -> dict:
    """Stage 2 풀링 메트릭 — pool-then-rate 원칙."""
    return compute_stage2_metrics(pooled_y_true, pooled_y_pred, stage2_labels)


def _tau_variance(per_fold: list[dict]) -> dict:
    """폴드별 τ 값 모아 평균/표준편차/최소/최대 계산."""
    out: dict = {}
    for key in ("tau_red", "tau_review"):
        vals = np.array([f[key] for f in per_fold], dtype=float)
        out[key] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=0)),
            "min": float(vals.min()),
            "max": float(vals.max()),
            "values": vals.tolist(),
        }
    return out


def cross_validate_hierarchical(
    df: pd.DataFrame,
    feature_cols: list[str],
    n_splits: int = 5,
    stratify_col: str = "risk_level",
    seed: int = 42,
    train_kwargs: dict | None = None,
) -> dict:
    """K-fold StratifiedKFold 교차 검증.

    Parameters
    ----------
    df : pd.DataFrame
        원본 데이터 — risk_level / yellow_subtype / feature_cols 포함.
    feature_cols : list[str]
        train_hierarchical 에 전달할 피처 컬럼.
    n_splits : int, default 5
    stratify_col : str, default "risk_level"
        분층화 기준 컬럼. yellow_subtype 은 sparse 해서 부적절 (advisor 권고).
    seed : int, default 42
    train_kwargs : dict, optional
        train_hierarchical 추가 인자 (recall_floor, review_recall_target,
        cost_sensitive 등). seed/output_dir 은 본 함수가 관리.

    Returns
    -------
    dict
        {
          "per_fold": list[dict] — 각 폴드의 메트릭 + τ + 샘플 수,
          "pooled": dict — Stage 1 mean±std (스칼라), Stage 2 풀링 CM 기반 P/R/F1,
          "tau_variance": dict — τ_red/τ_review 폴드 간 평균/표준/최대/최소,
          "n_splits": int,
          "stratify_col": str,
        }
    """
    if train_kwargs is None:
        train_kwargs = {}

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y_strat = df[stratify_col].to_numpy()

    per_fold: list[dict] = []
    pooled_y1_true: list[np.ndarray] = []
    pooled_p1_red: list[np.ndarray] = []
    pooled_y2_true: list[np.ndarray] = []
    pooled_y2_pred: list[np.ndarray] = []

    with tempfile.TemporaryDirectory() as tmp_root:
        tmp_root_path = Path(tmp_root)
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df, y_strat)):
            fold_dir = tmp_root_path / f"fold_{fold_idx}"
            train_df = df.iloc[train_idx].copy()
            val_df = df.iloc[val_idx].copy()

            bundle = train_hierarchical(
                df=train_df,
                feature_cols=feature_cols,
                output_dir=fold_dir,
                seed=seed,
                **train_kwargs,
            )

            X_val = val_df[feature_cols].to_numpy()
            y1_val = (val_df["risk_level"] == "Red").astype(int).to_numpy()

            y2_val_raw = _encode_stage2_for_eval(val_df)
            mask = y2_val_raw != -1

            stage2_bundle = joblib.load(fold_dir / "stage2_yellow.joblib")
            classes_present = np.asarray(stage2_bundle["classes_present"], dtype=int)

            stage1 = bundle["stage1_model"]
            stage2 = bundle["stage2_model"]
            p1 = stage1.predict_proba(X_val)[:, 1]

            X2 = X_val[mask]
            y2_true_fold = y2_val_raw[mask].astype(int)
            if len(X2) > 0:
                y2_local = np.asarray(stage2.predict(X2), dtype=int)
                if y2_local.max(initial=-1) >= len(classes_present):
                    raise ValueError(
                        f"fold {fold_idx}: stage2.predict local max "
                        f"({y2_local.max()}) >= classes_present 길이 "
                        f"({len(classes_present)})"
                    )
                y2_pred_fold = classes_present[y2_local].astype(int)
            else:
                y2_pred_fold = np.array([], dtype=int)

            stage1_fold = compute_stage1_metrics(y1_val, p1)
            stage2_fold = compute_stage2_metrics(y2_true_fold, y2_pred_fold)

            per_fold.append({
                "fold": fold_idx,
                "n_train": int(len(train_df)),
                "n_val": int(len(val_df)),
                "n_red_val": int(y1_val.sum()),
                "n_stage2_val": int(mask.sum()),
                "tau_red": float(bundle["thresholds"]["tau_red"]),
                "tau_review": float(bundle["thresholds"]["tau_review"]),
                "stage1": stage1_fold,
                "stage2": stage2_fold,
            })

            pooled_y1_true.append(y1_val)
            pooled_p1_red.append(p1)
            pooled_y2_true.append(y2_true_fold)
            pooled_y2_pred.append(y2_pred_fold)

    # ── Stage 1 스칼라: 폴드별 mean ± std (rank-based 지표는 풀링 부적절) ──
    stage1_keys = ("pr_auc", "roc_auc", "brier")
    stage1_pooled = {}
    for k in stage1_keys:
        vals = np.array([f["stage1"][k] for f in per_fold], dtype=float)
        stage1_pooled[k] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=0)),
            "values": vals.tolist(),
        }
    stage1_pooled["n_samples_total"] = int(sum(len(a) for a in pooled_y1_true))
    stage1_pooled["n_positive_total"] = int(sum(int(a.sum()) for a in pooled_y1_true))

    # ── Stage 2: pool counts → derive rates ─────────────────────────────────
    pooled_y2t = np.concatenate(pooled_y2_true) if pooled_y2_true else np.array([], dtype=int)
    pooled_y2p = np.concatenate(pooled_y2_pred) if pooled_y2_pred else np.array([], dtype=int)
    stage2_pooled = _pool_stage2_metrics(pooled_y2t, pooled_y2p)

    pooled = {
        "stage1": stage1_pooled,
        "stage2": stage2_pooled,
    }

    return {
        "per_fold": per_fold,
        "pooled": pooled,
        "tau_variance": _tau_variance(per_fold),
        "n_splits": n_splits,
        "stratify_col": stratify_col,
    }
