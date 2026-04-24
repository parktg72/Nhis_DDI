"""계층 분류 러너 (Stage 1 Red 이진 + Stage 2 Yellow 서브라벨 6-class).

라벨 상수, 인코딩/디코딩 헬퍼, 임계값 선택, sample_weight, 학습/추론.
Stage 1 / Stage 2 모델은 각각 독립 joblib 로 저장되고
predict_risk() 가 2단 임계값 (τ_red, τ_review) 으로 분기한다.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from .ml_runner import load_features_from_parquet, stratified_sample_from_parquet

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
) -> np.ndarray:
    """Stage 2 6-class balanced sample_weight.

    balanced (클래스 불균형 역수) 가 기본 깔개. cost_sensitive=True 이고
    cost_ratio_by_class 가 주어지면 각 샘플에 해당 클래스 비용 배수를 곱한다.

    Parameters
    ----------
    y_train : np.ndarray
        정수 인코딩된 Stage 2 라벨 — encode_stage2_labels() 의 출력.
        문자열 라벨 또는 0..len(STAGE2_LABELS)-1 범위 외 정수는 지원하지 않음.
    cost_sensitive : bool
        True 면 cost_ratio_by_class 로 balanced 가중치를 곱함.
    cost_ratio_by_class : dict[str, float] | None
        클래스 이름(STAGE2_LABELS 중 하나) → 비용 배수.
        명시되지 않은 클래스는 1.0 (변경 없음) 으로 처리.
        STAGE2_LABELS 에 없는 키가 있으면 KeyError (오탈자 차단).
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

    label_strs = [STAGE2_LABELS[int(i)] for i in y_arr]
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
      Recall ≥ review_recall_target (더 보수적) 을 만족하는 최소 임계값.
      review_recall_target > recall_floor 이어야 τ_review < τ_red 가 보장됨.

    Returns
    -------
    {"tau_red": float, "tau_review": float}
    """
    from sklearn.metrics import precision_recall_curve

    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)

    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # precision_recall_curve: precision/recall 은 len N+1, thresholds 는 len N

    # τ_red: recall ≥ recall_floor 을 만족하는 후보 중 최대 precision
    valid_red = recall[:-1] >= recall_floor  # 마지막 point 는 threshold 없음
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

    # 방어: 수치 엣지에서 순서 뒤집힘 방지
    if tau_review >= tau_red:
        tau_review = tau_red * 0.5

    return {"tau_red": tau_red, "tau_review": tau_review}
