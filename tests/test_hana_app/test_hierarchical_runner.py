"""hierarchical_runner: Stage 1/2 라벨 상수 및 인코딩."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

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
    import pytest
    with pytest.raises(ValueError, match="Red"):
        build_stage2_label(risk_level="Red", yellow_subtype=None)


def test_build_stage2_label_unknown_risk_level_raises():
    """알 수 없는 risk_level 은 ValueError (silent drift 방지)."""
    import pytest
    with pytest.raises(ValueError, match="유효하지 않은 risk_level"):
        build_stage2_label(risk_level="Unknown", yellow_subtype=None)

    with pytest.raises(ValueError, match="유효하지 않은 risk_level"):
        build_stage2_label(risk_level="yellow", yellow_subtype="Y_MIX")  # 대소문자 오염

    with pytest.raises(ValueError, match="유효하지 않은 risk_level"):
        build_stage2_label(risk_level="", yellow_subtype=None)


def test_build_stage2_label_y_other_is_excluded():
    """Y_OTHER 는 학습셋에서 제외되어야 하므로 명시적 예외."""
    import pytest
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
