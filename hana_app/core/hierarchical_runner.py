"""계층 분류 러너 (Stage 1 Red 이진 + Stage 2 Yellow 서브라벨 6-class).

라벨 상수, 인코딩/디코딩 헬퍼, 임계값 선택, sample_weight, 학습/추론.
Stage 1 / Stage 2 모델은 각각 독립 joblib 로 저장되고
predict_risk() 가 2단 임계값 (τ_red, τ_review) 으로 분기한다.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.preprocessing import LabelEncoder

YELLOW_SUBTYPE_LABELS: tuple[str, ...] = (
    "Y_MIX", "Y_DDI_MAJOR", "Y_DDI_MOD", "Y_DUP", "Y_FRAG",
)
STAGE2_LABELS: tuple[str, ...] = YELLOW_SUBTYPE_LABELS + ("No_Alert",)

_VALID_RISK_LEVELS: frozenset[str] = frozenset({"Red", "Yellow", "Green", "Normal"})


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

    cost_ratio_by_class 키는 STAGE2_LABELS 의 문자열이어야 한다.
    알 수 없는 키가 있으면 KeyError (오탈자 차단).
    """
    from sklearn.utils.class_weight import compute_sample_weight

    y_arr = np.asarray(y_train)
    balanced = compute_sample_weight("balanced", y_arr)
    if not cost_sensitive or cost_ratio_by_class is None:
        return balanced

    unknown = set(cost_ratio_by_class) - set(STAGE2_LABELS)
    if unknown:
        raise KeyError(
            f"cost_ratio_by_class 에 STAGE2 가 아닌 키 포함: {sorted(unknown)}"
        )

    # y_arr 은 정수 인덱스 — STAGE2_LABELS 순서로 역변환해 라벨 문자열 얻기
    label_strs = [STAGE2_LABELS[int(i)] for i in y_arr]
    cost_mult = np.array(
        [cost_ratio_by_class.get(s, 1.0) for s in label_strs],
        dtype=float,
    )
    return balanced * cost_mult
