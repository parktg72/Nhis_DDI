"""hierarchical_metrics 순수 계산 함수 테스트.

Stage 1 PR-AUC/ROC-AUC/Brier + Stage 2 macro F1/per-class/CM 의 경계 동작
(완벽 / 최악 / 입력 계약 위반 / 리포트 직렬화) 검증.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from hana_app.core.hierarchical_metrics import (
    _format_markdown,
    compute_hierarchical_metrics,
    compute_stage1_metrics,
    compute_stage2_metrics,
    save_metrics_report,
)
from hana_app.core.hierarchical_runner import STAGE2_LABELS


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1
# ─────────────────────────────────────────────────────────────────────────────

def test_stage1_perfect_classifier():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    p_red = np.array([0.01, 0.05, 0.10, 0.90, 0.95, 0.99])
    m = compute_stage1_metrics(y_true, p_red)
    assert m["pr_auc"] == pytest.approx(1.0)
    assert m["roc_auc"] == pytest.approx(1.0)
    assert m["brier"] < 0.05  # 확률 추정도 근접
    assert m["n_samples"] == 6
    assert m["n_positive"] == 3


def test_stage1_worst_classifier():
    """확률이 정답과 완전 반대 — ROC-AUC = 0.0."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    p_red = np.array([0.99, 0.95, 0.90, 0.10, 0.05, 0.01])
    m = compute_stage1_metrics(y_true, p_red)
    assert m["roc_auc"] == pytest.approx(0.0)
    # Brier 는 ≈ (0.99²*3 + 0.9²*3) / 6 ≈ 0.896
    assert m["brier"] > 0.8


def test_stage1_rejects_single_class():
    """양 또는 음 하나만 있으면 ValueError."""
    with pytest.raises(ValueError, match="양/음"):
        compute_stage1_metrics(np.zeros(10), np.random.rand(10))


def test_stage1_rejects_length_mismatch():
    with pytest.raises(ValueError, match="길이 불일치"):
        compute_stage1_metrics(np.array([0, 1]), np.array([0.1, 0.2, 0.3]))


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2
# ─────────────────────────────────────────────────────────────────────────────

def test_stage2_perfect_multiclass():
    """모든 예측이 정답 — macro F1 = 1.0, CM 은 대각선."""
    n_classes = len(STAGE2_LABELS)
    y_true = np.arange(n_classes * 3) % n_classes
    y_pred = y_true.copy()
    m = compute_stage2_metrics(y_true, y_pred)
    assert m["macro_f1"] == pytest.approx(1.0)
    cm = np.array(m["confusion_matrix"])
    # 대각선만 채워짐
    assert np.all(np.diag(cm) > 0)
    assert np.sum(cm) == len(y_true)
    assert np.sum(cm - np.diag(np.diag(cm))) == 0


def test_stage2_worst_multiclass():
    """모든 예측이 같은 (오답) 클래스 — macro F1 근접 0."""
    n_classes = len(STAGE2_LABELS)
    y_true = np.arange(n_classes)
    y_pred = np.zeros(n_classes, dtype=int)  # 모두 class 0
    m = compute_stage2_metrics(y_true, y_pred)
    # class 0 만 맞음 → f1_0 = 소량, 나머지 0 → macro 낮음
    assert m["macro_f1"] < 0.5


def test_stage2_per_class_structure():
    """per_class dict 가 모든 STAGE2_LABELS 키 포함."""
    n = len(STAGE2_LABELS)
    y_true = np.arange(n)
    y_pred = y_true.copy()
    m = compute_stage2_metrics(y_true, y_pred)
    assert set(m["per_class"].keys()) == set(STAGE2_LABELS)
    for lbl, stats in m["per_class"].items():
        assert set(stats.keys()) == {"precision", "recall", "f1", "support"}


def test_stage2_confusion_matrix_shape():
    """CM 은 항상 6×6 (누락 클래스도 0 행/열로 유지)."""
    # 일부 클래스만 등장
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 0])
    m = compute_stage2_metrics(y_true, y_pred)
    cm = np.array(m["confusion_matrix"])
    assert cm.shape == (len(STAGE2_LABELS), len(STAGE2_LABELS))
    # 등장하지 않는 class (2..5) 행/열 모두 0
    for i in range(2, len(STAGE2_LABELS)):
        assert cm[i].sum() == 0
        assert cm[:, i].sum() == 0


def test_stage2_rejects_length_mismatch():
    with pytest.raises(ValueError, match="길이 불일치"):
        compute_stage2_metrics(np.array([0, 1]), np.array([0, 1, 2]))


# ─────────────────────────────────────────────────────────────────────────────
# 통합 + save
# ─────────────────────────────────────────────────────────────────────────────

def test_compute_hierarchical_metrics_merges():
    y1_true = np.array([0, 1, 0, 1])
    p1_red = np.array([0.1, 0.9, 0.2, 0.85])
    # non-Red 서브셋 (y1_true==0) 에 대해서만 Stage 2
    y2_true = np.array([5, 0])  # No_Alert, Y_TRIPLE
    y2_pred = np.array([5, 0])
    rep = compute_hierarchical_metrics(y1_true, p1_red, y2_true, y2_pred)
    assert "stage1" in rep
    assert "stage2" in rep
    assert rep["stage1"]["pr_auc"] == pytest.approx(1.0)
    assert rep["stage2"]["macro_f1"] > 0


def test_save_metrics_report_roundtrip(tmp_path):
    rep = {
        "stage1": {
            "pr_auc": 0.85, "roc_auc": 0.92, "brier": 0.10,
            "n_samples": 100, "n_positive": 30,
        },
        "stage2": {
            "macro_f1": 0.75,
            "per_class": {
                lbl: {"precision": 0.8, "recall": 0.7, "f1": 0.75, "support": 50}
                for lbl in STAGE2_LABELS
            },
            "confusion_matrix": [[10]*len(STAGE2_LABELS)]*len(STAGE2_LABELS),
            "labels": list(STAGE2_LABELS),
            "n_samples": 70,
        },
    }
    paths = save_metrics_report(rep, tmp_path)
    assert paths["json"].exists()
    assert paths["markdown"].exists()

    # JSON round-trip
    loaded = json.loads(paths["json"].read_text())
    assert loaded["stage1"]["pr_auc"] == pytest.approx(0.85)
    assert len(loaded["stage2"]["confusion_matrix"]) == len(STAGE2_LABELS)

    # Markdown 주요 필드 포함
    md = paths["markdown"].read_text()
    assert "# 계층 분류 평가 리포트" in md
    assert "Stage 1" in md and "Stage 2" in md
    assert "Confusion Matrix" in md
    for lbl in STAGE2_LABELS:
        assert lbl in md


def test_markdown_absent_class_shows_na():
    """support=0 인 클래스는 0.0000 대신 N/A 로 렌더."""
    rep = {
        "stage1": {"pr_auc": 0.9, "roc_auc": 0.95, "brier": 0.05,
                   "n_samples": 10, "n_positive": 3},
        "stage2": {
            "macro_f1": 0.5,
            "per_class": {
                lbl: ({"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}
                      if lbl == "Y_OTHER" or lbl == "Y_FRAG"
                      else {"precision": 0.7, "recall": 0.6, "f1": 0.65, "support": 5})
                for lbl in STAGE2_LABELS
            },
            "confusion_matrix": [[0]*len(STAGE2_LABELS) for _ in STAGE2_LABELS],
            "labels": list(STAGE2_LABELS),
            "n_samples": 20,
        },
    }
    md = _format_markdown(rep)
    # support=0 라벨은 N/A, 0 support 표시
    # Y_FRAG 가 support=0
    for line in md.split("\n"):
        if line.startswith("| Y_FRAG |"):
            assert "N/A" in line, f"support=0 인 Y_FRAG 이 N/A 로 표시되지 않음: {line}"


def test_format_markdown_smoke():
    """_format_markdown 이 예외 없이 문자열 반환."""
    rep = {
        "stage1": {"pr_auc": 0.9, "roc_auc": 0.95, "brier": 0.05,
                   "n_samples": 50, "n_positive": 10},
        "stage2": {
            "macro_f1": 0.80,
            "per_class": {
                lbl: {"precision": 0.8, "recall": 0.7, "f1": 0.75, "support": 10}
                for lbl in STAGE2_LABELS
            },
            "confusion_matrix": [[1]*len(STAGE2_LABELS) for _ in STAGE2_LABELS],
            "labels": list(STAGE2_LABELS),
            "n_samples": 40,
        },
    }
    s = _format_markdown(rep)
    assert isinstance(s, str)
    assert s.strip().startswith("# 계층 분류 평가 리포트")


# ─────────────────────────────────────────────────────────────────────────────
# bundle 래퍼 — train_hierarchical 과 end-to-end
# ─────────────────────────────────────────────────────────────────────────────

def test_evaluate_hierarchical_bundle_roundtrip(tmp_path):
    """train_hierarchical → evaluate_hierarchical_bundle end-to-end."""
    import pandas as pd

    from hana_app.core.hierarchical_metrics import evaluate_hierarchical_bundle
    from hana_app.core.hierarchical_runner import train_hierarchical

    rng = np.random.default_rng(42)
    n = 500
    df = pd.DataFrame({
        "patient_id": [f"P{i}" for i in range(n)],
        "feat_a": rng.random(n),
        "feat_b": rng.random(n),
        "feat_c": rng.random(n),
        "risk_level": (["Red"] * 25 + ["Yellow"] * 100
                       + ["Green"] * 150 + ["Normal"] * 225),
        "yellow_subtype": (
            [None] * 25
            + ["Y_TRIPLE"] * 10 + ["Y_DDI_MAJOR"] * 15 + ["Y_DDI_MOD"] * 30
            + ["Y_DUP"] * 25 + ["Y_FRAG"] * 20
            + [None] * 375
        ),
    })
    bundle = train_hierarchical(
        df=df, feature_cols=["feat_a", "feat_b", "feat_c"],
        output_dir=tmp_path, seed=42,
    )

    # classes_present 는 stage2_yellow.joblib 번들에서 로드 (실운영 경로)
    import joblib
    stage2_bundle = joblib.load(tmp_path / "stage2_yellow.joblib")
    classes_present = stage2_bundle["classes_present"]

    # 동일 데이터로 검증 (정확도보단 구조/실행 검증이 목적)
    X_val = df[["feat_a", "feat_b", "feat_c"]].to_numpy()
    y1_val = (df["risk_level"] == "Red").astype(int).to_numpy()
    # Stage 2 라벨 인코딩
    stage2_labels_list = list(STAGE2_LABELS)

    def _encode(row):
        if row["risk_level"] == "Red":
            return -1  # Red 는 stage2 대상 아님
        if row["risk_level"] == "Yellow":
            sub = row["yellow_subtype"]
            if sub in stage2_labels_list:
                return stage2_labels_list.index(sub)
        return stage2_labels_list.index("No_Alert")
    y2_val = df.apply(_encode, axis=1).to_numpy()
    mask = y2_val != -1  # non-Red 만

    # evaluate_hierarchical_bundle 은 내부에서 mask 처리 (y1_val==0 기본)
    rep = evaluate_hierarchical_bundle(
        bundle, X_val, y1_val,
        y2_val=np.where(mask, y2_val, 0),  # Red 위치는 임시 0 (mask 로 어차피 제외됨)
        classes_present=classes_present,
        y2_mask=mask,
    )

    assert "stage1" in rep and "stage2" in rep
    assert 0.0 <= rep["stage1"]["pr_auc"] <= 1.0
    assert 0.0 <= rep["stage1"]["roc_auc"] <= 1.0
    assert 0.0 <= rep["stage2"]["macro_f1"] <= 1.0
    assert len(rep["stage2"]["confusion_matrix"]) == len(STAGE2_LABELS)


def test_evaluate_hierarchical_bundle_accepts_stage2_softprob_matrix():
    """Stage2 predict 가 확률 행렬을 반환해도 local→global 매핑 후 평가한다."""
    from hana_app.core.hierarchical_metrics import evaluate_hierarchical_bundle

    class _Stage1:
        def predict_proba(self, X):
            return np.array([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.7, 0.3]])

    class _Stage2Softprob:
        def predict(self, X):
            return np.array([
                [0.9, 0.1, 0.0],
                [0.1, 0.8, 0.1],
                [0.0, 0.2, 0.8],
            ])

    bundle = {"stage1_model": _Stage1(), "stage2_model": _Stage2Softprob()}
    x_val = np.zeros((4, 2))
    y1_val = np.array([0, 0, 1, 0])
    classes_present = [5, 0, 1]  # local 0/1/2 → global No_Alert/Y_TRIPLE/Y_DOUBLE
    y2_val = np.array([5, 0, 0, 1])

    rep = evaluate_hierarchical_bundle(
        bundle, x_val, y1_val, y2_val=y2_val,
        classes_present=classes_present, y2_mask=(y1_val == 0),
    )

    cm = np.array(rep["stage2"]["confusion_matrix"])
    assert cm.sum() == 3
    assert cm[5, 5] == 1
    assert cm[0, 0] == 1
    assert cm[1, 1] == 1
