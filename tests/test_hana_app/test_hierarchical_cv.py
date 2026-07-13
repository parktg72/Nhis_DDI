"""hierarchical_cv 의 4 가지 핵심 계약 검증 (advisor 권고).

1. 구조: 반환 dict 키
2. 재현성: 같은 seed → 동일 per_fold 메트릭
3. 분층화: 폴드별 Red 개수 분산 ≤ ±1 (StratifiedKFold 작동 확인)
4. Pool-then-rate: 풀링 CM = sum of fold CMs, macro_f1 은 sklearn 풀링 재현
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from hana_app.core.hierarchical_cv import (
    _encode_stage2_for_eval,
    cross_validate_hierarchical,
)
from hana_app.core.hierarchical_runner import STAGE2_LABELS

# ─────────────────────────────────────────────────────────────────────────────
# 공통 fixture — 작은 합성 데이터로 5-fold 가능
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def cv_dataset():
    rng = np.random.default_rng(42)
    n = 600
    # 분포: Red 30 (5%), Yellow 120 (20%), Green 180 (30%), Normal 270 (45%)
    risk_levels = (
        ["Red"] * 30 + ["Yellow"] * 120
        + ["Green"] * 180 + ["Normal"] * 270
    )
    yellow_subtypes = (
        [None] * 30
        + ["Y_TRIPLE"] * 20 + ["Y_DDI_MAJOR"] * 25 + ["Y_DDI_MOD"] * 30
        + ["Y_DUP"] * 25 + ["Y_FRAG"] * 20
        + [None] * 450
    )
    df = pd.DataFrame({
        "patient_id": [f"P{i}" for i in range(n)],
        "feat_a": rng.random(n),
        "feat_b": rng.random(n),
        "feat_c": rng.random(n),
        "risk_level": risk_levels,
        "yellow_subtype": yellow_subtypes,
    })
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 1. 구조
# ─────────────────────────────────────────────────────────────────────────────

def test_cv_returns_expected_top_level_keys(cv_dataset):
    out = cross_validate_hierarchical(
        cv_dataset, feature_cols=["feat_a", "feat_b", "feat_c"],
        n_splits=3, seed=0,
    )
    assert set(out.keys()) == {
        "per_fold", "pooled", "tau_variance", "n_splits", "stratify_col"
    }
    assert out["n_splits"] == 3
    assert out["stratify_col"] == "risk_level"
    assert len(out["per_fold"]) == 3


def test_cv_per_fold_structure(cv_dataset):
    out = cross_validate_hierarchical(
        cv_dataset, feature_cols=["feat_a", "feat_b", "feat_c"],
        n_splits=3, seed=0,
    )
    fold0 = out["per_fold"][0]
    expected_keys = {
        "fold", "n_train", "n_val", "n_red_val", "n_stage2_val",
        "tau_red", "tau_review", "stage1", "stage2",
    }
    assert set(fold0.keys()) == expected_keys
    # τ 불변식
    assert fold0["tau_review"] < fold0["tau_red"]


def test_cv_tau_variance_structure(cv_dataset):
    out = cross_validate_hierarchical(
        cv_dataset, feature_cols=["feat_a", "feat_b", "feat_c"],
        n_splits=3, seed=0,
    )
    tv = out["tau_variance"]
    assert set(tv.keys()) == {"tau_red", "tau_review"}
    for k in tv:
        assert set(tv[k].keys()) == {"mean", "std", "min", "max", "values"}
        assert len(tv[k]["values"]) == 3


# ─────────────────────────────────────────────────────────────────────────────
# 2. 재현성
# ─────────────────────────────────────────────────────────────────────────────

def test_cv_reproducibility_same_seed(cv_dataset):
    out1 = cross_validate_hierarchical(
        cv_dataset, feature_cols=["feat_a", "feat_b", "feat_c"],
        n_splits=3, seed=123,
    )
    out2 = cross_validate_hierarchical(
        cv_dataset, feature_cols=["feat_a", "feat_b", "feat_c"],
        n_splits=3, seed=123,
    )
    # τ 값 정확 일치
    assert out1["tau_variance"]["tau_red"]["values"] == \
        out2["tau_variance"]["tau_red"]["values"]
    # 폴드별 PR-AUC 정확 일치
    pr1 = [f["stage1"]["pr_auc"] for f in out1["per_fold"]]
    pr2 = [f["stage1"]["pr_auc"] for f in out2["per_fold"]]
    assert pr1 == pr2


# ─────────────────────────────────────────────────────────────────────────────
# 3. 분층화 (Red 보존)
# ─────────────────────────────────────────────────────────────────────────────

def test_cv_stratification_preserves_red_balance(cv_dataset):
    """30 Red / 5 folds → 폴드별 Red 6개 ±1 이내."""
    out = cross_validate_hierarchical(
        cv_dataset, feature_cols=["feat_a", "feat_b", "feat_c"],
        n_splits=5, seed=0,
    )
    red_counts = [f["n_red_val"] for f in out["per_fold"]]
    expected = 30 // 5  # = 6
    for c in red_counts:
        assert abs(c - expected) <= 1, f"폴드별 Red 분포 불균형: {red_counts}"
    # 합산은 정확히 전체 Red 수
    assert sum(red_counts) == 30


# ─────────────────────────────────────────────────────────────────────────────
# 4. Pool-then-rate (풀링된 CM = fold CM 합)
# ─────────────────────────────────────────────────────────────────────────────

def test_cv_pooled_cm_equals_fold_cm_sum(cv_dataset):
    """advisor: pool counts, derive rates from pooled. CM 합이 풀링과 일치."""
    out = cross_validate_hierarchical(
        cv_dataset, feature_cols=["feat_a", "feat_b", "feat_c"],
        n_splits=3, seed=0,
    )
    fold_cms = [
        np.array(f["stage2"]["confusion_matrix"]) for f in out["per_fold"]
    ]
    expected_pooled = np.sum(fold_cms, axis=0)
    actual_pooled = np.array(out["pooled"]["stage2"]["confusion_matrix"])
    np.testing.assert_array_equal(actual_pooled, expected_pooled)


def test_cv_pooled_macro_f1_derived_from_pooled_cm(cv_dataset):
    """pooled.macro_f1 은 풀링된 (y_true, y_pred) 에서 sklearn 으로 직접 계산한
    값과 일치 (rate-then-pool 의 평균과는 일반적으로 다름)."""
    out = cross_validate_hierarchical(
        cv_dataset, feature_cols=["feat_a", "feat_b", "feat_c"],
        n_splits=3, seed=0,
    )
    pooled_cm = np.array(out["pooled"]["stage2"]["confusion_matrix"])
    n_classes = len(STAGE2_LABELS)
    # CM 으로부터 y_true, y_pred 재구성
    y_true_rec = []
    y_pred_rec = []
    for i in range(n_classes):
        for j in range(n_classes):
            count = int(pooled_cm[i, j])
            y_true_rec.extend([i] * count)
            y_pred_rec.extend([j] * count)
    macro_f1_from_pooled = f1_score(
        y_true_rec, y_pred_rec,
        labels=list(range(n_classes)), average="macro", zero_division=0,
    )
    assert out["pooled"]["stage2"]["macro_f1"] == pytest.approx(macro_f1_from_pooled)


# ─────────────────────────────────────────────────────────────────────────────
def test_cv_accepts_stage2_softprob_predict_matrix(cv_dataset, monkeypatch):
    """CV Stage2 평가도 XGBoost softprob 행렬 반환 변형을 허용한다."""
    from xgboost import XGBClassifier

    def _predict_softprob(self, X, *args, **kwargs):
        return self.predict_proba(X)

    monkeypatch.setattr(XGBClassifier, "predict", _predict_softprob)

    out = cross_validate_hierarchical(
        cv_dataset, feature_cols=["feat_a", "feat_b", "feat_c"],
        n_splits=3, seed=0,
    )

    assert len(out["per_fold"]) == 3
    assert 0.0 <= out["pooled"]["stage2"]["macro_f1"] <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 부수: encoder 보조 함수
# ─────────────────────────────────────────────────────────────────────────────

def test_encode_stage2_for_eval_handles_red():
    df = pd.DataFrame({
        "risk_level": ["Red", "Yellow", "Green", "Normal"],
        "yellow_subtype": [None, "Y_TRIPLE", None, None],
    })
    out = _encode_stage2_for_eval(df)
    assert out[0] == -1   # Red
    assert out[1] == STAGE2_LABELS.index("Y_TRIPLE")
    assert out[2] == STAGE2_LABELS.index("No_Alert")
    assert out[3] == STAGE2_LABELS.index("No_Alert")


def test_encode_stage2_for_eval_masks_y_other_and_invalid_yellow():
    """Yellow + Y_OTHER / 무효 / null subtype → -1 (mask out).

    학습(`stratified_sample_stage2`)이 Y_OTHER·무효 subtype Yellow 를 제외하므로
    평가도 동일하게 마스킹해야 Stage 2 메트릭이 학습 분포와 정합한다.
    (이전엔 No_Alert 로 폴백 — 2026-06-02 RCA B1 에서 학습/평가 불일치로 확정.
    `docs/reports/2026-06-02_ml_dl_and_diskfull_review.md`.)
    """
    df = pd.DataFrame({
        "risk_level": ["Yellow", "Yellow", "Yellow"],
        "yellow_subtype": ["Y_OTHER", "Y_INVALID", None],
    })
    out = _encode_stage2_for_eval(df)
    assert out[0] == -1  # Y_OTHER → mask (학습 제외)
    assert out[1] == -1  # 무효 subtype → mask
    assert out[2] == -1  # null subtype Yellow → mask
