"""계층 분류 러너 (Stage 1 Red 이진 + Stage 2 Yellow 서브라벨 6-class).

라벨 상수, 인코딩/디코딩 헬퍼, 임계값 선택, sample_weight, 학습/추론.
Stage 1 / Stage 2 모델은 각각 독립 joblib 로 저장되고
predict_risk() 가 2단 임계값 (τ_red, τ_review) 으로 분기한다.
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

YELLOW_SUBTYPE_LABELS: tuple[str, ...] = (
    "Y_MIX", "Y_DDI_MAJOR", "Y_DDI_MOD", "Y_DUP", "Y_FRAG",
)
STAGE2_LABELS: tuple[str, ...] = YELLOW_SUBTYPE_LABELS + ("No_Alert",)


def build_stage2_label(risk_level: str, yellow_subtype: Optional[str]) -> str:
    """Stage 2 학습용 라벨 변환.

    Red 는 Stage 2 대상이 아니므로 ValueError. Y_OTHER 는 학습셋에서 제외.
    """
    if risk_level == "Red":
        raise ValueError("build_stage2_label: Red is handled by Stage 1, not Stage 2")
    if yellow_subtype == "Y_OTHER":
        raise ValueError("build_stage2_label: Y_OTHER must be excluded from training set")
    if risk_level == "Yellow":
        if yellow_subtype is None or yellow_subtype not in YELLOW_SUBTYPE_LABELS:
            raise ValueError(
                f"build_stage2_label: Yellow requires yellow_subtype in "
                f"{YELLOW_SUBTYPE_LABELS}, got {yellow_subtype!r}"
            )
        return yellow_subtype
    return "No_Alert"


def encode_stage2_labels(labels: Iterable[str]):
    """Stage 2 라벨 문자열 → 정수 인코딩. classes_ 는 STAGE2_LABELS 순서 고정.

    Returns
    -------
    (y_int: np.ndarray, encoder: sklearn.preprocessing.LabelEncoder)
    """
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    # LabelEncoder.fit() sorts classes_ alphabetically; override to fix STAGE2_LABELS order.
    encoder.classes_ = np.array(list(STAGE2_LABELS))
    y = encoder.transform(list(labels))
    return y, encoder


def decode_stage2_labels(y: np.ndarray, encoder) -> np.ndarray:
    """정수 → 문자열 역변환."""
    return encoder.inverse_transform(np.asarray(y))
