"""계층 분류 러너 (Stage 1 Red 이진 + Stage 2 Yellow 서브라벨 6-class).

라벨 상수, 인코딩/디코딩 헬퍼, 임계값 선택, sample_weight, 학습/추론.
Stage 1 / Stage 2 모델은 각각 독립 joblib 로 저장되고
predict_risk() 가 2단 임계값 (τ_red, τ_review) 으로 분기한다.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from .ml_runner import load_features_from_parquet, stratified_sample_from_parquet

# clinical_rules 는 scripts/etl/ 에 위치 — 절대 경로 import (모듈 최상단)
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from scripts.etl.clinical_rules import CLINICAL_STANDARDS_VERSION  # noqa: E402

YELLOW_SUBTYPE_LABELS: tuple[str, ...] = (
    "Y_MIX", "Y_DDI_MAJOR", "Y_DDI_MOD", "Y_DUP", "Y_FRAG",
)
STAGE2_LABELS: tuple[str, ...] = YELLOW_SUBTYPE_LABELS + ("No_Alert",)

_VALID_RISK_LEVELS: frozenset[str] = frozenset({"Red", "Yellow", "Green", "Normal"})

_STAGE2_LABEL_TO_INT: dict[str, int] = {
    lbl: i for i, lbl in enumerate(STAGE2_LABELS)
}


def build_stage2_label(risk_level: str, yellow_subtype: str | None) -> str:
    """Stage 2 학습용 라벨 변환.

    Red 는 Stage 2 대상이 아니므로 ValueError. Y_OTHER 는 학습셋에서 제외.
    알 수 없는 risk_level 도 ValueError (silent drift 방지).
    """
    if risk_level not in _VALID_RISK_LEVELS:
        raise ValueError(
            f"유효하지 않은 risk_level: {risk_level!r}. "
            f"허용 값: {sorted(_VALID_RISK_LEVELS)}"
        )
    if risk_level == "Red":
        raise ValueError("Red 는 Stage 1 에서 처리됨 — Stage 2 라벨 아님")
    if yellow_subtype == "Y_OTHER":
        raise ValueError("Y_OTHER 는 학습셋에서 제외 대상 — Stage 2 라벨로 쓸 수 없음")
    if risk_level == "Yellow":
        if yellow_subtype is None or yellow_subtype not in YELLOW_SUBTYPE_LABELS:
            raise ValueError(
                f"Yellow 는 yellow_subtype 이 {YELLOW_SUBTYPE_LABELS} 중 하나여야 함. "
                f"받은 값: {yellow_subtype!r}"
            )
        return yellow_subtype
    return "No_Alert"


def encode_stage2_labels(
    labels: Iterable[str],
) -> tuple[np.ndarray, LabelEncoder]:
    """Stage 2 라벨 문자열 → 정수 인코딩. classes_ 는 STAGE2_LABELS 순서 고정.

    Returns
    -------
    (y_int: np.ndarray, encoder: LabelEncoder)

    주의: 반환되는 encoder 에 .fit() 을 다시 호출하지 말 것.
    classes_ 가 STAGE2_LABELS 순서로 직접 할당되어 있으므로
    .fit() 이 알파벳 정렬로 덮어쓰면 정수 인덱스가 드리프트한다.
    """
    encoder = LabelEncoder()
    # LabelEncoder.fit() sorts classes_ alphabetically; override to fix STAGE2_LABELS order.
    encoder.classes_ = np.array(list(STAGE2_LABELS))
    y = encoder.transform(list(labels))
    return y, encoder


def decode_stage2_labels(y: np.ndarray, encoder) -> np.ndarray:
    """정수 → 문자열 역변환."""
    return encoder.inverse_transform(np.asarray(y))


def _stage2_sample_weight(
    y_train: np.ndarray,
    cost_sensitive: bool = False,
    cost_ratio_by_class: dict[str, float] | None = None,
    class_names: tuple[str, ...] | None = None,
) -> np.ndarray:
    """Stage 2 6-class balanced sample_weight.

    balanced (클래스 불균형 역수) 가 기본 깔개. cost_sensitive=True 이고
    cost_ratio_by_class 가 주어지면 각 샘플에 해당 클래스 비용 배수를 곱한다.

    Parameters
    ----------
    y_train : np.ndarray
        정수 인코딩된 Stage 2 라벨 — encode_stage2_labels() 의 출력, 또는
        train_hierarchical 의 local→global 재매핑이 적용된 배열.
    cost_sensitive : bool
        True 면 cost_ratio_by_class 로 balanced 가중치를 곱함.
    cost_ratio_by_class : dict[str, float] | None
        클래스 이름(STAGE2_LABELS 중 하나) → 비용 배수.
        명시되지 않은 클래스는 1.0 (변경 없음) 으로 처리.
        STAGE2_LABELS 에 없는 키가 있으면 KeyError (오탈자 차단).
    class_names : tuple[str, ...] | None
        y_train 정수 인덱스 i → 클래스 이름 매핑.
        None 이면 기본 STAGE2_LABELS 사용 (전체 6-class 학습 가정).
        train_hierarchical 이 local→global 재매핑을 쓰는 경우에는
        local index 순서의 class_names 를 반드시 전달해야 cost_ratio 가
        올바른 클래스에 적용된다.
    """
    y_arr = np.asarray(y_train)
    if not np.issubdtype(y_arr.dtype, np.integer):
        raise TypeError(
            f"y_train 은 정수 인코딩 (encode_stage2_labels() 출력) 이어야 함 — "
            f"받은 dtype: {y_arr.dtype}"
        )

    balanced = compute_sample_weight("balanced", y_arr)
    if not cost_sensitive or cost_ratio_by_class is None:
        return balanced

    unknown = set(cost_ratio_by_class) - set(STAGE2_LABELS)
    if unknown:
        raise KeyError(
            f"cost_ratio_by_class 에 STAGE2 가 아닌 키 포함: {sorted(unknown)}"
        )

    _names = class_names if class_names is not None else STAGE2_LABELS
    label_strs = [_names[int(i)] for i in y_arr]
    cost_mult = np.array(
        [cost_ratio_by_class.get(s, 1.0) for s in label_strs],
        dtype=float,
    )
    return balanced * cost_mult


def stratified_sample_stage2(
    parquet_paths: list[Path] | str | Path,
    sample_size: int,
    seed: int = 42,
    memory_limit_mb: int = 512,
) -> pd.DataFrame:
    """Stage 2 용 층화 샘플링.

    전처리:
      1) risk_level != 'Red' prefilter (Red 는 Stage 1 영역)
      2) yellow_subtype == 'Y_OTHER' 제외 (학습 오염 방지)
      3) stage2_label 컬럼 derive — build_stage2_label() 로 validated
         (null yellow_subtype, 알 수 없는 risk_level 은 ValueError)
      4) stage2_label 기준 6-class 층화 추출

    내부적으로 ml_runner.stratified_sample_from_parquet 을 재사용한다
    (DuckDB numpy.int64 처리 등 호환 경로 보존).
    """
    df = load_features_from_parquet(parquet_paths, memory_limit_mb=memory_limit_mb)

    # DuckDB 가 null-only 컬럼을 Int32 로 읽을 수 있으므로 object 로 강제 변환
    df["yellow_subtype"] = df["yellow_subtype"].astype(object).where(
        df["yellow_subtype"].notna(), other=None
    )

    # prefilter
    df = df[df["risk_level"] != "Red"].copy()
    df = df[df["yellow_subtype"].fillna("") != "Y_OTHER"].copy()

    # stage2_label derive — build_stage2_label 이 validation 수행
    df["stage2_label"] = df.apply(
        lambda r: build_stage2_label(r["risk_level"], r["yellow_subtype"]),
        axis=1,
    )
    df["stage2_label_int"] = df["stage2_label"].map(_STAGE2_LABEL_TO_INT).astype("int64")

    # 임시 parquet → 층화 추출 (TemporaryDirectory 로 자동 정리)
    with tempfile.TemporaryDirectory(prefix="stage2_stratified_") as tmp_dir:
        tmp = Path(tmp_dir) / "stage2.parquet"
        df.to_parquet(tmp, index=False)
        sampled = stratified_sample_from_parquet(
            parquet_paths=tmp,
            target_col="stage2_label_int",
            sample_size=sample_size,
            seed=seed,
            memory_limit_mb=memory_limit_mb,
        )

    sampled = sampled.drop(columns=["stage2_label_int"], errors="ignore")
    return sampled


def select_thresholds_from_pr(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    recall_floor: float = 0.90,
    review_recall_target: float = 0.98,
) -> dict[str, float]:
    """PR 곡선에서 τ_red, τ_review 2단 임계값 선택.

    τ_red:
      Recall ≥ recall_floor 제약 하에서 Precision 이 최대가 되는 임계값.

    τ_review:
      Recall ≥ review_recall_target (더 보수적) 을 만족하는 최대 임계값
      (더 높은 threshold = 더 엄격 = 더 적은 샘플 → 가장 엄격한 후보 채택).
      precondition 으로 review_recall_target > recall_floor 을 강제하고,
      PR 곡선 해상도 부족으로 동값이 나올 때는 τ_review 를 1e-6 만큼 내려
      τ_review < τ_red 를 항상 보장한다.

    Parameters
    ----------
    y_true : np.ndarray
        이진 {0, 1} 정답 라벨. 양/음 둘 다 최소 1건 이상 필요.
    y_proba : np.ndarray
        [0, 1] 범위의 양성 클래스 예측 확률.
    recall_floor : float
        τ_red 가 보장해야 할 Recall 하한. 범위 (0, 1].
    review_recall_target : float
        τ_review 가 보장해야 할 Recall. recall_floor 보다 엄격해야 함 (더 큰 값).

    Returns
    -------
    {"tau_red": float, "tau_review": float}

    Raises
    ------
    ValueError
        입력이 계약을 위반할 때. silent garbage 대신 명시적 실패.
    """
    y_true_arr = np.asarray(y_true)
    if not np.isin(y_true_arr, [0, 1]).all():
        raise ValueError("y_true 는 이진 {0, 1} 배열이어야 함")
    if np.unique(y_true_arr).size < 2:
        raise ValueError(
            "y_true 는 양성/음성 샘플을 모두 포함해야 함 (PR 곡선 비정의)"
        )

    y_proba_arr = np.asarray(y_proba, dtype=float)
    if y_proba_arr.size != y_true_arr.size:
        raise ValueError(
            f"y_true 와 y_proba 길이 불일치: {y_true_arr.size} vs {y_proba_arr.size}"
        )
    if y_proba_arr.min() < 0.0 or y_proba_arr.max() > 1.0:
        raise ValueError("y_proba 는 [0.0, 1.0] 범위여야 함")

    if not (0.0 < recall_floor <= 1.0):
        raise ValueError(f"recall_floor 은 (0, 1] 범위 — 받은 값: {recall_floor}")
    if not (0.0 < review_recall_target <= 1.0):
        raise ValueError(
            f"review_recall_target 은 (0, 1] 범위 — 받은 값: {review_recall_target}"
        )
    if review_recall_target <= recall_floor:
        raise ValueError(
            f"review_recall_target ({review_recall_target}) 은 "
            f"recall_floor ({recall_floor}) 보다 커야 함 (τ_review < τ_red 보장)"
        )

    precision, recall, thresholds = precision_recall_curve(
        y_true_arr.astype(int), y_proba_arr
    )
    # precision_recall_curve: precision/recall 은 len N+1, thresholds 는 len N

    # τ_red: recall ≥ recall_floor 을 만족하는 후보 중 최대 precision
    valid_red = recall[:-1] >= recall_floor
    if not valid_red.any():
        tau_red = float(thresholds.min())
    else:
        cand_idx = np.where(valid_red)[0]
        best = cand_idx[np.argmax(precision[:-1][cand_idx])]
        tau_red = float(thresholds[best])

    # τ_review: recall ≥ review_recall_target 을 만족하는 최대 threshold
    valid_review = recall[:-1] >= review_recall_target
    if not valid_review.any():
        tau_review = float(thresholds.min())
    else:
        cand_idx = np.where(valid_review)[0]
        tau_review = float(thresholds[cand_idx].max())

    # PR 곡선 해상도 부족(양성 수 적음) 으로 τ_review == τ_red 동값 가능 —
    # 불변식 τ_review < τ_red 를 유지하도록 미세 조정.
    if tau_review >= tau_red:
        tau_review = max(0.0, tau_red - 1e-6)

    return {"tau_red": tau_red, "tau_review": tau_review}


def train_hierarchical(
    df: pd.DataFrame,
    feature_cols: list[str],
    output_dir,
    seed: int = 42,
    stage1_params: dict | None = None,
    stage2_params: dict | None = None,
    recall_floor: float = 0.90,
    review_recall_target: float = 0.98,
    cost_sensitive: bool = False,
    cost_ratio_by_class: dict[str, float] | None = None,
) -> dict:
    """Stage 1 (Red 이진) + Stage 2 (Yellow 서브라벨 6-class) 계층 학습.

    df 에는 risk_level, yellow_subtype, feature_cols 가 포함되어야 한다.
    Y_OTHER 는 Stage 2 학습셋에서 제외된다.

    저장 파일:
      {output_dir}/stage1_red.joblib
      {output_dir}/stage2_yellow.joblib
      {output_dir}/stage_meta.json  (임계값, feature_cols, 라벨 카운트, SHA-256)
    """
    import json
    import hashlib
    from collections import Counter

    import joblib
    from xgboost import XGBClassifier
    from sklearn.model_selection import train_test_split

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Stage 1: Red 이진 ─────────────────────────────────────────────────
    X = df[feature_cols].to_numpy()
    y1 = (df["risk_level"] == "Red").astype(int).to_numpy()

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y1, test_size=0.2, random_state=seed, stratify=y1,
    )

    pos = int(y_tr.sum())
    neg = int(len(y_tr) - pos)
    scale_pos_weight = neg / max(pos, 1)

    defaults1 = dict(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        objective="binary:logistic", eval_metric="logloss",
        scale_pos_weight=scale_pos_weight, random_state=seed, verbosity=0,
    )
    if stage1_params:
        defaults1.update(stage1_params)
    m1 = XGBClassifier(**defaults1)
    m1.fit(X_tr, y_tr)
    proba_val = m1.predict_proba(X_val)[:, 1]
    thresholds = select_thresholds_from_pr(
        y_val, proba_val,
        recall_floor=recall_floor,
        review_recall_target=review_recall_target,
    )

    # ── Stage 2: 6-class ─────────────────────────────────────────────────
    mask_non_red = df["risk_level"] != "Red"
    mask_not_other = df["yellow_subtype"].fillna("") != "Y_OTHER"
    y_other_excluded = int(((df["risk_level"] == "Yellow") &
                            (df["yellow_subtype"] == "Y_OTHER")).sum())
    df2 = df[mask_non_red & mask_not_other].copy()
    labels_str = [
        build_stage2_label(r["risk_level"], r["yellow_subtype"])
        for _, r in df2.iterrows()
    ]
    # global indices (STAGE2_LABELS 순서 고정): 0..5
    y2_global, encoder = encode_stage2_labels(labels_str)
    X2 = df2[feature_cols].to_numpy()

    # XGBoost 는 y 가 [0, num_class) 연속 범위여야 한다.
    # 학습셋에 없는 클래스(예: Y_FRAG)가 있으면 global → local 재매핑.
    classes_present = np.unique(y2_global)          # e.g. [0,1,2,3,5]
    if len(classes_present) < len(STAGE2_LABELS) or not (
        classes_present == np.arange(len(STAGE2_LABELS))
    ).all():
        global_to_local = {int(g): l for l, g in enumerate(classes_present)}
        y2 = np.array([global_to_local[int(g)] for g in y2_global], dtype=int)
        n_stage2_classes = len(classes_present)
    else:
        y2 = y2_global
        n_stage2_classes = len(STAGE2_LABELS)

    # local index → 전역 클래스 이름 매핑: cost_ratio 가 올바른 클래스에 적용되도록 보장
    local_class_names = tuple(STAGE2_LABELS[g] for g in classes_present)
    sw2 = _stage2_sample_weight(
        y2, cost_sensitive=cost_sensitive,
        cost_ratio_by_class=cost_ratio_by_class,
        class_names=local_class_names,
    )

    defaults2 = dict(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        objective="multi:softprob", num_class=n_stage2_classes,
        eval_metric="mlogloss", random_state=seed, verbosity=0,
    )
    if stage2_params:
        defaults2.update(stage2_params)
    m2 = XGBClassifier(**defaults2)
    m2.fit(X2, y2, sample_weight=sw2)

    # ── 저장 ─────────────────────────────────────────────────────────────
    p1 = out / "stage1_red.joblib"
    p2 = out / "stage2_yellow.joblib"
    joblib.dump(m1, p1)
    # stage2_classes_global: predict_risk 에서 로컬 인덱스 → 전역 STAGE2_LABELS 매핑에 사용
    joblib.dump({
        "model": m2,
        "encoder": encoder,
        "stage2_classes_global": classes_present.tolist(),
        "classes_present": classes_present.tolist(),
    }, p2)

    def _sha(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()

    label_counts = dict(Counter(labels_str))

    meta = {
        "clinical_standards_version": CLINICAL_STANDARDS_VERSION,
        "feature_cols": list(feature_cols),
        "thresholds": thresholds,
        "stage2_labels": list(STAGE2_LABELS),
        "stage2_label_counts": label_counts,
        "y_other_excluded_count": y_other_excluded,
        "stage1_sha256": _sha(p1),
        "stage2_sha256": _sha(p2),
        "cost_sensitive": cost_sensitive,
        "cost_ratio_by_class": cost_ratio_by_class,
    }
    (out / "stage_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2)
    )

    return {
        "stage1_model": m1,
        "stage2_model": m2,
        "stage2_encoder": encoder,
        "thresholds": thresholds,
        "stage2_label_counts": label_counts,
        "y_other_excluded_count": y_other_excluded,
        "meta_path": out / "stage_meta.json",
    }
