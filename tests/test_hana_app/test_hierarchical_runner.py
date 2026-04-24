"""hierarchical_runner: Stage 1/2 라벨 상수 및 인코딩."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hana_app.core.hierarchical_runner import (
    YELLOW_SUBTYPE_LABELS,
    STAGE2_LABELS,
    build_stage2_label,
    encode_stage2_labels,
    decode_stage2_labels,
)


def test_yellow_subtype_labels_constant():
    assert YELLOW_SUBTYPE_LABELS == ("Y_MIX", "Y_DDI_MAJOR", "Y_DDI_MOD", "Y_DUP", "Y_FRAG")


def test_stage2_labels_includes_no_alert():
    assert STAGE2_LABELS == ("Y_MIX", "Y_DDI_MAJOR", "Y_DDI_MOD", "Y_DUP", "Y_FRAG", "No_Alert")
    assert len(STAGE2_LABELS) == 6


def test_build_stage2_label_yellow_subtype():
    assert build_stage2_label(risk_level="Yellow", yellow_subtype="Y_MIX") == "Y_MIX"
    assert build_stage2_label(risk_level="Yellow", yellow_subtype="Y_DDI_MAJOR") == "Y_DDI_MAJOR"


def test_build_stage2_label_green_normal_are_no_alert():
    assert build_stage2_label(risk_level="Green", yellow_subtype=None) == "No_Alert"
    assert build_stage2_label(risk_level="Normal", yellow_subtype=None) == "No_Alert"


def test_build_stage2_label_red_raises():
    """Red 는 Stage 2 대상이 아님."""
    with pytest.raises(ValueError, match="Red"):
        build_stage2_label(risk_level="Red", yellow_subtype=None)


def test_build_stage2_label_unknown_risk_level_raises():
    """알 수 없는 risk_level 은 ValueError (silent drift 방지)."""
    with pytest.raises(ValueError, match="유효하지 않은 risk_level"):
        build_stage2_label(risk_level="Unknown", yellow_subtype=None)

    with pytest.raises(ValueError, match="유효하지 않은 risk_level"):
        build_stage2_label(risk_level="yellow", yellow_subtype="Y_MIX")  # 대소문자 오염

    with pytest.raises(ValueError, match="유효하지 않은 risk_level"):
        build_stage2_label(risk_level="", yellow_subtype=None)


def test_build_stage2_label_y_other_is_excluded():
    """Y_OTHER 는 학습셋에서 제외되어야 하므로 명시적 예외."""
    with pytest.raises(ValueError, match="Y_OTHER"):
        build_stage2_label(risk_level="Yellow", yellow_subtype="Y_OTHER")


def test_encode_decode_roundtrip():
    labels = ["Y_MIX", "No_Alert", "Y_DUP", "Y_MIX", "Y_FRAG"]
    y, encoder = encode_stage2_labels(labels)
    assert y.dtype.kind == "i"
    assert len(y) == 5
    # classes_ 는 정해진 순서 (STAGE2_LABELS) 를 따라야 함
    assert list(encoder.classes_) == list(STAGE2_LABELS)
    decoded = decode_stage2_labels(y, encoder)
    assert list(decoded) == labels


def test_encode_preserves_class_order_across_inputs():
    """입력 분포가 달라도 classes_ 순서는 STAGE2_LABELS 고정."""
    y1, enc1 = encode_stage2_labels(["Y_MIX", "No_Alert"])
    y2, enc2 = encode_stage2_labels(["No_Alert", "Y_DUP"])
    assert list(enc1.classes_) == list(STAGE2_LABELS)
    assert list(enc2.classes_) == list(STAGE2_LABELS)


def test_select_thresholds_returns_both_tau():
    from hana_app.core.hierarchical_runner import select_thresholds_from_pr

    rng = np.random.default_rng(42)
    # y_true: 10% Red, y_proba: Red 에 대해 약간 높은 값
    y_true = np.array([1] * 100 + [0] * 900)
    y_proba = np.concatenate([
        rng.beta(5, 2, 100),   # Red 쪽 확률 높게
        rng.beta(2, 5, 900),   # non-Red 확률 낮게
    ])
    thr = select_thresholds_from_pr(y_true, y_proba, recall_floor=0.90)
    assert "tau_red" in thr and "tau_review" in thr
    assert 0.0 < thr["tau_review"] < thr["tau_red"] < 1.0


def test_tau_red_respects_recall_floor():
    from hana_app.core.hierarchical_runner import select_thresholds_from_pr
    from sklearn.metrics import recall_score

    rng = np.random.default_rng(0)
    y_true = np.array([1] * 100 + [0] * 900)
    y_proba = np.concatenate([
        rng.beta(5, 2, 100),
        rng.beta(2, 5, 900),
    ])
    thr = select_thresholds_from_pr(y_true, y_proba, recall_floor=0.90)

    y_pred = (y_proba >= thr["tau_red"]).astype(int)
    assert recall_score(y_true, y_pred) >= 0.90 - 0.01  # 수치 오차 허용


def test_tau_review_is_lower_than_tau_red():
    from hana_app.core.hierarchical_runner import select_thresholds_from_pr

    rng = np.random.default_rng(7)
    y_true = np.concatenate([np.ones(50), np.zeros(950)])
    y_proba = np.concatenate([rng.beta(4, 2, 50), rng.beta(2, 4, 950)])
    thr = select_thresholds_from_pr(
        y_true, y_proba,
        recall_floor=0.90,
        review_recall_target=0.98,
    )
    # review 는 더 느슨한 임계값 → 더 낮음
    assert thr["tau_review"] < thr["tau_red"]


def test_threshold_rejects_non_binary_y_true():
    import pytest
    from hana_app.core.hierarchical_runner import select_thresholds_from_pr
    with pytest.raises(ValueError, match="이진"):
        select_thresholds_from_pr(
            y_true=np.array([0, 1, 2]),
            y_proba=np.array([0.1, 0.5, 0.9]),
        )


def test_threshold_rejects_single_class_y_true():
    import pytest
    from hana_app.core.hierarchical_runner import select_thresholds_from_pr
    with pytest.raises(ValueError, match="양성/음성"):
        select_thresholds_from_pr(
            y_true=np.zeros(10, dtype=int),
            y_proba=np.linspace(0.1, 0.9, 10),
        )


def test_threshold_rejects_out_of_range_proba():
    import pytest
    from hana_app.core.hierarchical_runner import select_thresholds_from_pr
    with pytest.raises(ValueError, match=r"\[0.0, 1.0\]"):
        select_thresholds_from_pr(
            y_true=np.array([0, 0, 1]),
            y_proba=np.array([0.1, 0.5, 1.5]),
        )


def test_threshold_rejects_review_target_below_floor():
    """review_recall_target <= recall_floor 이면 τ_review < τ_red 가 깨짐 → 거부."""
    import pytest
    from hana_app.core.hierarchical_runner import select_thresholds_from_pr
    with pytest.raises(ValueError, match="보다 커야 함"):
        select_thresholds_from_pr(
            y_true=np.array([0, 1, 0, 1]),
            y_proba=np.array([0.1, 0.6, 0.3, 0.8]),
            recall_floor=0.9,
            review_recall_target=0.8,  # floor 보다 낮음 → 오류
        )


def test_threshold_rejects_mismatched_lengths():
    import pytest
    from hana_app.core.hierarchical_runner import select_thresholds_from_pr
    with pytest.raises(ValueError, match="길이 불일치"):
        select_thresholds_from_pr(
            y_true=np.array([0, 1]),
            y_proba=np.array([0.1, 0.5, 0.9]),
        )


def test_train_hierarchical_returns_two_models(tmp_path):
    """train_hierarchical 은 Stage 1 + Stage 2 모델과 임계값을 반환."""
    from hana_app.core.hierarchical_runner import train_hierarchical

    rng = np.random.default_rng(42)
    n = 500
    df = pd.DataFrame({
        "patient_id": [f"P{i}" for i in range(n)],
        "feat_a": rng.random(n),
        "feat_b": rng.random(n),
        "feat_c": rng.random(n),
        "risk_level": (["Red"] * 25 + ["Yellow"] * 100 +
                       ["Green"] * 150 + ["Normal"] * 225),
        "yellow_subtype": (
            [None] * 25
            + ["Y_MIX"] * 10 + ["Y_DDI_MAJOR"] * 15 + ["Y_DDI_MOD"] * 30
            + ["Y_DUP"] * 25 + ["Y_FRAG"] * 20
            + [None] * 375
        ),
    })

    result = train_hierarchical(
        df=df,
        feature_cols=["feat_a", "feat_b", "feat_c"],
        output_dir=tmp_path,
        seed=42,
    )

    # 반환 구조 검증
    assert "stage1_model" in result
    assert "stage2_model" in result
    assert "thresholds" in result
    assert "tau_red" in result["thresholds"]
    assert "tau_review" in result["thresholds"]
    assert result["thresholds"]["tau_review"] < result["thresholds"]["tau_red"]

    # 파일 저장 검증
    assert (tmp_path / "stage1_red.joblib").exists()
    assert (tmp_path / "stage2_yellow.joblib").exists()
    assert (tmp_path / "stage_meta.json").exists()


def test_train_hierarchical_excludes_y_other_from_stage2(tmp_path):
    from hana_app.core.hierarchical_runner import train_hierarchical

    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({
        "patient_id": [f"P{i}" for i in range(n)],
        "feat_a": rng.random(n),
        "feat_b": rng.random(n),
        "risk_level": (["Red"] * 10 + ["Yellow"] * 100 + ["Normal"] * 190),
        "yellow_subtype": (
            [None] * 10
            + ["Y_OTHER"] * 20   # 학습셋에서 빠져야 함
            + ["Y_MIX"] * 20 + ["Y_DDI_MAJOR"] * 20
            + ["Y_DDI_MOD"] * 20 + ["Y_DUP"] * 20
            + [None] * 190
        ),
    })

    result = train_hierarchical(
        df=df, feature_cols=["feat_a", "feat_b"],
        output_dir=tmp_path, seed=0,
    )
    # stage2 학습에 사용된 라벨 집합에 Y_OTHER 없음
    assert "Y_OTHER" not in result["stage2_label_counts"]
    # 감사: Y_OTHER 제외 건수 기록
    assert result["y_other_excluded_count"] == 20
