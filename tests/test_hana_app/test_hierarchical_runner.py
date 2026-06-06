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
    assert YELLOW_SUBTYPE_LABELS == ("Y_TRIPLE", "Y_DOUBLE", "Y_DDI_MAJOR", "Y_DDI_MOD", "Y_DUP", "Y_FRAG")


def test_stage2_labels_includes_no_alert():
    assert STAGE2_LABELS == ("Y_TRIPLE", "Y_DOUBLE", "Y_DDI_MAJOR", "Y_DDI_MOD", "Y_DUP", "Y_FRAG", "No_Alert")
    assert len(STAGE2_LABELS) == 7


def test_build_stage2_label_yellow_subtype():
    assert build_stage2_label(risk_level="Yellow", yellow_subtype="Y_TRIPLE") == "Y_TRIPLE"
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
        build_stage2_label(risk_level="yellow", yellow_subtype="Y_TRIPLE")  # 대소문자 오염

    with pytest.raises(ValueError, match="유효하지 않은 risk_level"):
        build_stage2_label(risk_level="", yellow_subtype=None)


def test_build_stage2_label_y_other_is_excluded():
    """Y_OTHER 는 학습셋에서 제외되어야 하므로 명시적 예외."""
    with pytest.raises(ValueError, match="Y_OTHER"):
        build_stage2_label(risk_level="Yellow", yellow_subtype="Y_OTHER")


def test_encode_decode_roundtrip():
    labels = ["Y_TRIPLE", "No_Alert", "Y_DUP", "Y_TRIPLE", "Y_FRAG"]
    y, encoder = encode_stage2_labels(labels)
    assert y.dtype.kind == "i"
    assert len(y) == 5
    # classes_ 는 정해진 순서 (STAGE2_LABELS) 를 따라야 함
    assert list(encoder.classes_) == list(STAGE2_LABELS)
    decoded = decode_stage2_labels(y, encoder)
    assert list(decoded) == labels


def test_encode_preserves_class_order_across_inputs():
    """입력 분포가 달라도 classes_ 순서는 STAGE2_LABELS 고정."""
    y1, enc1 = encode_stage2_labels(["Y_TRIPLE", "No_Alert"])
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
            + ["Y_TRIPLE"] * 10 + ["Y_DDI_MAJOR"] * 15 + ["Y_DDI_MOD"] * 30
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
            + ["Y_TRIPLE"] * 20 + ["Y_DDI_MAJOR"] * 20
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


def test_train_hierarchical_local_global_remapping_roundtrip(tmp_path):
    """Stage 2 학습 시 일부 클래스가 누락되어도 predict_proba + classes_present 로
    올바르게 STAGE2_LABELS 문자열로 복원 가능한지."""
    from hana_app.core.hierarchical_runner import (
        train_hierarchical, STAGE2_LABELS,
    )
    import joblib

    rng = np.random.default_rng(99)
    n = 200
    # Y_FRAG 가 학습 데이터에 전혀 없도록 구성
    df = pd.DataFrame({
        "patient_id": [f"P{i}" for i in range(n)],
        "feat_a": rng.random(n),
        "feat_b": rng.random(n),
        "risk_level": (["Red"] * 10 + ["Yellow"] * 80 + ["Normal"] * 110),
        "yellow_subtype": (
            [None] * 10
            + ["Y_TRIPLE"] * 20 + ["Y_DDI_MAJOR"] * 20
            + ["Y_DDI_MOD"] * 20 + ["Y_DUP"] * 20  # Y_FRAG 없음
            + [None] * 110
        ),
    })
    result = train_hierarchical(
        df=df, feature_cols=["feat_a", "feat_b"],
        output_dir=tmp_path, seed=99,
    )
    # 저장된 stage2 번들에 classes_present 가 포함
    bundle = joblib.load(tmp_path / "stage2_yellow.joblib")
    assert "classes_present" in bundle
    classes_present = bundle["classes_present"]
    # Y_FRAG 의 global index(5, STAGE2_LABELS 재정렬 후) 는 포함되지 않아야 함
    assert 5 not in classes_present

    # 로컬 인덱스 → 전역 라벨 복원
    local_to_global_labels = [STAGE2_LABELS[g] for g in classes_present]
    assert "Y_FRAG" not in local_to_global_labels
    assert "No_Alert" in local_to_global_labels


def test_train_hierarchical_cost_sensitive_with_missing_class(tmp_path):
    """Critical regression: cost_sensitive=True + 누락 클래스 조합에서
    cost_ratio 가 local 인덱스로 잘못 적용되지 않아야 함."""
    from hana_app.core.hierarchical_runner import train_hierarchical

    rng = np.random.default_rng(77)
    n = 200
    df = pd.DataFrame({
        "patient_id": [f"P{i}" for i in range(n)],
        "feat_a": rng.random(n),
        "feat_b": rng.random(n),
        "risk_level": (["Red"] * 10 + ["Yellow"] * 80 + ["Normal"] * 110),
        "yellow_subtype": (
            [None] * 10
            + ["Y_TRIPLE"] * 20 + ["Y_DDI_MAJOR"] * 20
            + ["Y_DDI_MOD"] * 20 + ["Y_DUP"] * 20  # Y_FRAG 없음
            + [None] * 110
        ),
    })

    # cost_sensitive=True + 존재하는 클래스(Y_DUP)에 가중치
    # Y_FRAG 가 없으므로 local→global remapping 이 발동함
    result = train_hierarchical(
        df=df, feature_cols=["feat_a", "feat_b"],
        output_dir=tmp_path, seed=77,
        cost_sensitive=True,
        cost_ratio_by_class={"Y_DUP": 5.0},
    )
    # 에러 없이 완료 + label 분포 유지
    assert "Y_DUP" in result["stage2_label_counts"]
    assert "Y_FRAG" not in result["stage2_label_counts"]


def test_dispatch_no_alert_action():
    from hana_app.core.hierarchical_runner import _dispatch_result, STAGE2_LABELS
    # 마지막 원소 No_Alert 가 가장 높음 (STAGE2_LABELS 7-class)
    probs = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.70])
    r = _dispatch_result(
        p_red=0.05, stage2_probs=probs, stage2_labels=STAGE2_LABELS,
        tau_red=0.7, tau_review=0.3,
    )
    assert r["risk_level"] == "No_Alert"
    assert r["action"] == "관여 안 함"   # No_Alert(2026-06-07)


def test_dispatch_red_confirmed_above_tau_red():
    from hana_app.core.hierarchical_runner import _dispatch_result, STAGE2_LABELS
    r = _dispatch_result(
        p_red=0.95, stage2_probs=None, stage2_labels=STAGE2_LABELS,
        tau_red=0.7, tau_review=0.3,
    )
    assert r["risk_level"] == "Red"
    assert r["red_suspect"] is False
    assert r["action"] == "즉각 개입"
    assert r["stage2_probs"] is None


def test_dispatch_red_suspect_between_thresholds():
    """τ_review ≤ P(Red) < τ_red → Stage 2 출력 + red_suspect=True."""
    from hana_app.core.hierarchical_runner import _dispatch_result, STAGE2_LABELS
    probs = np.array([0.6, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05])  # index0 = Y_TRIPLE
    r = _dispatch_result(
        p_red=0.5, stage2_probs=probs, stage2_labels=STAGE2_LABELS,
        tau_red=0.7, tau_review=0.3,
    )
    assert r["risk_level"] == "Y_TRIPLE"
    assert r["red_suspect"] is True
    assert r["action"] == "문자 안내"   # Y_TRIPLE → 문자 안내(2026-06-07)


def test_predict_risk_end_to_end(tmp_path):
    """predict_risk 가 train_hierarchical 번들로 추론 결과 리스트를 반환하는지."""
    from hana_app.core.hierarchical_runner import train_hierarchical, predict_risk

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
            + ["Y_TRIPLE"] * 20 + ["Y_DOUBLE"] * 20 + ["Y_DDI_MAJOR"] * 20
            + ["Y_DDI_MOD"] * 20 + ["Y_DUP"] * 10 + ["Y_FRAG"] * 10
            + [None] * 375
        ),
    })
    bundle = train_hierarchical(
        df=df, feature_cols=["feat_a", "feat_b", "feat_c"],
        output_dir=tmp_path, seed=42,
    )
    X = df[["feat_a", "feat_b", "feat_c"]].iloc[:10].to_numpy()
    results = predict_risk(
        X=X,
        stage1_model=bundle["stage1_model"],
        stage2_model=bundle["stage2_model"],
        stage2_encoder=bundle["stage2_encoder"],
        thresholds=bundle["thresholds"],
    )
    assert len(results) == 10
    for r in results:
        assert set(r.keys()) >= {
            "risk_level", "p_red", "stage2_probs", "red_suspect", "action"
        }
        assert r["p_red"] >= 0.0 and r["p_red"] <= 1.0


def test_predict_risk_with_missing_class_uses_classes_present(tmp_path):
    """Y_FRAG 없이 학습된 모델에서 predict_risk 가 classes_present 로 올바르게 복원."""
    from hana_app.core.hierarchical_runner import (
        train_hierarchical, predict_risk, STAGE2_LABELS,
    )
    import joblib

    rng = np.random.default_rng(123)
    n = 300
    df = pd.DataFrame({
        "patient_id": [f"P{i}" for i in range(n)],
        "feat_a": rng.random(n),
        "feat_b": rng.random(n),
        "risk_level": (["Red"] * 10 + ["Yellow"] * 100 + ["Normal"] * 190),
        "yellow_subtype": (
            [None] * 10
            + ["Y_TRIPLE"] * 25 + ["Y_DDI_MAJOR"] * 25
            + ["Y_DDI_MOD"] * 25 + ["Y_DUP"] * 25  # Y_FRAG 없음
            + [None] * 190
        ),
    })
    bundle = train_hierarchical(
        df=df, feature_cols=["feat_a", "feat_b"],
        output_dir=tmp_path, seed=123,
    )
    # 저장된 번들에서 classes_present 복원
    stage2_bundle = joblib.load(tmp_path / "stage2_yellow.joblib")
    classes_present = stage2_bundle["classes_present"]
    # Y_FRAG 의 global index=5 (STAGE2_LABELS 재정렬 후) 는 포함되지 않아야 함
    assert 5 not in classes_present

    # 명시적 classes_present 로 predict_risk 호출
    X = df[["feat_a", "feat_b"]].iloc[:20].to_numpy()
    results = predict_risk(
        X=X,
        stage1_model=bundle["stage1_model"],
        stage2_model=bundle["stage2_model"],
        stage2_encoder=bundle["stage2_encoder"],
        thresholds=bundle["thresholds"],
        classes_present=classes_present,
    )
    assert len(results) == 20

    # 누락된 Y_FRAG 의 확률은 Red 가 아닌 모든 결과에서 0.0 이어야 함
    y_frag_idx = STAGE2_LABELS.index("Y_FRAG")
    for r in results:
        if r["risk_level"] != "Red":
            probs = r["stage2_probs"]
            assert probs["Y_FRAG"] == 0.0, f"Y_FRAG 확률은 0.0 이어야 함, 받은 값: {probs['Y_FRAG']}"


def test_dispatch_raises_valueerror_on_missing_stage2_probs():
    """assert 대신 ValueError — production -O 모드에서도 안전."""
    from hana_app.core.hierarchical_runner import _dispatch_result, STAGE2_LABELS
    import pytest
    with pytest.raises(ValueError, match="stage2_probs"):
        _dispatch_result(
            p_red=0.5, stage2_probs=None,
            stage2_labels=STAGE2_LABELS,
            tau_red=0.7, tau_review=0.3,
        )


def test_end_to_end_train_predict(tmp_path):
    """라벨 생성 → 학습 → 추론의 전체 플로우.

    Task 0-9 의 모든 컴포넌트를 하나의 워크플로우에서 검증.
    """
    from datetime import date
    from scripts.etl.models import PatientFeatures
    from scripts.etl.prescription_aggregator import (
        _assign_risk_level, _assign_yellow_subtype,
    )
    from hana_app.core.ml_runner import _patient_features_to_row
    from hana_app.core.hierarchical_runner import (
        train_hierarchical, predict_risk,
    )

    # 합성 PatientFeatures 생성 — 각 카테고리별 최소 30건
    features = []
    import random
    rng = random.Random(42)

    def _ft(**kw):
        base = dict(patient_id=f"P{len(features):05d}",
                    window_start=date(2026, 1, 1),
                    window_end=date(2026, 3, 31))
        base.update(kw)
        return PatientFeatures(**base)

    # Red (30)
    for _ in range(30):
        features.append(_ft(ddi_contraindicated=1, drug_count=rng.randint(3, 8)))
    # Y_DOUBLE (30) — 2 위험차원 (중등도 상호작용 + 중복). major 는 Y_DDI_MAJOR 이므로 mod 사용.
    for _ in range(30):
        features.append(_ft(ddi_moderate=2, dup_same_ingredient=1,
                             drug_count=rng.randint(3, 8)))
    # Y_TRIPLE (30) — 3 위험차원 (중등도 상호작용 + 중복 + 다기관)
    for _ in range(30):
        features.append(_ft(ddi_moderate=2, dup_same_ingredient=1, institution_count=3,
                             drug_count=rng.randint(3, 8)))
    # Y_DDI_MAJOR (30)
    for _ in range(30):
        features.append(_ft(ddi_major=1, drug_count=rng.randint(3, 8)))
    # Y_DDI_MOD (30)
    for _ in range(30):
        features.append(_ft(ddi_moderate=2, drug_count=rng.randint(3, 8)))
    # Y_DUP (30)
    for _ in range(30):
        features.append(_ft(dup_same_ingredient=1, drug_count=rng.randint(3, 8)))
    # Y_FRAG (30)
    for _ in range(30):
        features.append(_ft(institution_count=3, drug_count=rng.randint(3, 8)))
    # No_Alert — Normal (50)
    for _ in range(50):
        features.append(_ft(drug_count=rng.randint(0, 3)))

    # ETL 단계: risk_level + yellow_subtype 생성
    for f in features:
        _assign_risk_level(f)
        _assign_yellow_subtype(f)

    # row 직렬화 → 학습용 DataFrame
    df = pd.DataFrame([_patient_features_to_row(f) for f in features])

    # 검증: risk_level 과 yellow_subtype 이 올바르게 채워졌는지
    assert (df["risk_level"] == "Red").sum() == 30
    assert df.loc[df["risk_level"] == "Yellow", "yellow_subtype"].notna().all()
    assert (df["yellow_subtype"] == "Y_DOUBLE").sum() == 30
    assert (df["yellow_subtype"] == "Y_TRIPLE").sum() == 30

    feature_cols = ["drug_count", "ddi_major", "ddi_moderate",
                    "dup_same_ingredient", "institution_count"]

    # 학습
    bundle = train_hierarchical(
        df=df, feature_cols=feature_cols,
        output_dir=tmp_path, seed=42,
    )
    assert bundle["thresholds"]["tau_red"] > 0
    assert bundle["thresholds"]["tau_review"] < bundle["thresholds"]["tau_red"]

    # 추론 — 학습 데이터 일부 샘플로 스모크 테스트
    X_sample = df[feature_cols].iloc[:20].to_numpy()
    results = predict_risk(
        X=X_sample,
        stage1_model=bundle["stage1_model"],
        stage2_model=bundle["stage2_model"],
        stage2_encoder=bundle["stage2_encoder"],
        thresholds=bundle["thresholds"],
    )
    assert len(results) == 20

    # 스모크 검증: 각 결과에 필수 키, p_red 범위, action 존재
    for r in results:
        assert set(r.keys()) >= {"risk_level", "p_red", "red_suspect", "action"}
        assert 0.0 <= r["p_red"] <= 1.0
        assert isinstance(r["action"], str) and len(r["action"]) > 0


def test_end_to_end_meta_has_clinical_standards_version(tmp_path):
    """학습 결과 메타 파일이 CLINICAL_STANDARDS_VERSION 을 기록해 재현성을 보장."""
    import json
    from hana_app.core.hierarchical_runner import train_hierarchical

    rng = np.random.default_rng(7)
    n = 200
    df = pd.DataFrame({
        "patient_id": [f"P{i}" for i in range(n)],
        "feat_a": rng.random(n),
        "feat_b": rng.random(n),
        "risk_level": (["Red"] * 10 + ["Yellow"] * 80 + ["Normal"] * 110),
        "yellow_subtype": (
            [None] * 10
            + ["Y_TRIPLE"] * 20 + ["Y_DDI_MAJOR"] * 20
            + ["Y_DDI_MOD"] * 20 + ["Y_DUP"] * 20
            + [None] * 110
        ),
    })
    train_hierarchical(
        df=df, feature_cols=["feat_a", "feat_b"],
        output_dir=tmp_path, seed=7,
    )
    meta = json.loads((tmp_path / "stage_meta.json").read_text())
    assert "clinical_standards_version" in meta
    assert meta["clinical_standards_version"] == "v1.0"
    assert "stage1_sha256" in meta
    assert "stage2_sha256" in meta
    assert "stage2_label_counts" in meta
    assert "y_other_excluded_count" in meta


# ─── Red 0건 → Stage 1 자동 degrade (Stage 2 만 학습) ────────────────────────
def _no_red_df(n=400, seed=0):
    """Red 0건, Yellow(서브타입)+Green 만 있는 df — 다운로드 Raw same-window 표본 모사."""
    rng = np.random.default_rng(seed)
    n_green = 40
    n_yellow = n - n_green
    ysub = (["Y_TRIPLE", "Y_DDI_MAJOR", "Y_DDI_MOD", "Y_DUP", "Y_FRAG"] * ((n_yellow // 5) + 1))[:n_yellow]
    return pd.DataFrame({
        "patient_id": [f"P{i}" for i in range(n)],
        "feat_a": rng.random(n),
        "feat_b": rng.random(n),
        "risk_level": (["Yellow"] * n_yellow + ["Green"] * n_green),
        "yellow_subtype": (ysub + [None] * n_green),
    })


def test_train_hierarchical_degrades_stage1_when_no_red(tmp_path):
    """Red 0건이면 ValueError 대신 상수 비-Red Stage1 로 degrade, Stage2 정상 학습."""
    from hana_app.core.hierarchical_runner import train_hierarchical, _ConstantNegativeStage1
    import json

    df = _no_red_df()
    result = train_hierarchical(
        df=df, feature_cols=["feat_a", "feat_b"], output_dir=tmp_path, seed=0,
    )
    assert result["stage1_trained"] is False
    assert result["stage1_red_count"] == 0
    assert isinstance(result["stage1_model"], _ConstantNegativeStage1)
    # 번들 구조는 그대로(서빙 호환): 세 파일 모두 존재
    assert (tmp_path / "stage1_red.joblib").exists()
    assert (tmp_path / "stage2_yellow.joblib").exists()
    meta = json.loads((tmp_path / "stage_meta.json").read_text(encoding="utf-8"))
    assert meta["stage1_trained"] is False
    # 불변식 유지: tau_review < tau_red
    assert result["thresholds"]["tau_review"] < result["thresholds"]["tau_red"]


def test_degraded_predict_risk_routes_all_to_stage2(tmp_path):
    """degrade 모델로 predict_risk: 아무도 Red/red_suspect 아니고 모두 Stage2 라벨."""
    from hana_app.core.hierarchical_runner import train_hierarchical, predict_risk
    import joblib

    df = _no_red_df()
    result = train_hierarchical(
        df=df, feature_cols=["feat_a", "feat_b"], output_dir=tmp_path, seed=0,
    )
    bundle = joblib.load(tmp_path / "stage2_yellow.joblib")  # 라운드트립(서빙 경로 모사)
    preds = predict_risk(
        df[["feat_a", "feat_b"]].to_numpy(),
        stage1_model=joblib.load(tmp_path / "stage1_red.joblib"),
        stage2_model=bundle["model"],
        stage2_encoder=result["stage2_encoder"],
        thresholds=result["thresholds"],
        classes_present=bundle["classes_present"],
    )
    assert len(preds) == len(df)
    assert all(p["risk_level"] != "Red" for p in preds)
    assert all(p["red_suspect"] is False for p in preds)
    assert all(p["p_red"] == 0.0 for p in preds)
