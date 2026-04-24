"""Yellow 세분화 UI 헬퍼 순수 집계 함수 테스트.

render_yellow_subtype_section() 자체는 Streamlit 컨텍스트가 필요해서 직접
테스트하지 않고, 순수 집계 함수 (summarize_yellow_subtypes, count_red_suspect)
분기 로직을 검증.
"""
from __future__ import annotations

import pandas as pd
import pytest

from hana_app.core.hierarchical_runner import ACTION_BY_LABEL
from hana_app.core.yellow_subtype_view import (
    YELLOW_SUBTYPE_COLORS,
    count_red_suspect,
    summarize_yellow_subtypes,
)


# ─────────────────────────────────────────────────────────────────────────────
# summarize_yellow_subtypes
# ─────────────────────────────────────────────────────────────────────────────

def test_summarize_empty_when_column_missing():
    df = pd.DataFrame({"risk_level": ["Red", "Yellow"]})
    out = summarize_yellow_subtypes(df)
    assert out.empty
    assert list(out.columns) == ["yellow_subtype", "count", "action"]


def test_summarize_empty_when_all_null():
    df = pd.DataFrame({"yellow_subtype": [None, None, pd.NA]})
    out = summarize_yellow_subtypes(df)
    assert out.empty


def test_summarize_counts_and_actions():
    df = pd.DataFrame({
        "yellow_subtype": (
            ["Y_MIX"] * 3
            + ["Y_DDI_MAJOR"] * 2
            + ["Y_DDI_MOD"] * 5
            + ["Y_DUP"] * 1
            + [None] * 4   # drop
        ),
    })
    out = summarize_yellow_subtypes(df)
    # None 제외 4개 라벨
    assert len(out) == 4
    counts_by_label = dict(zip(out["yellow_subtype"], out["count"]))
    assert counts_by_label == {
        "Y_MIX": 3,
        "Y_DDI_MAJOR": 2,
        "Y_DDI_MOD": 5,
        "Y_DUP": 1,
    }
    # action 매핑 검증 (ACTION_BY_LABEL 규약)
    action_by_label = dict(zip(out["yellow_subtype"], out["action"]))
    for lbl, exp in ACTION_BY_LABEL.items():
        if lbl in action_by_label:
            assert action_by_label[lbl] == exp


def test_summarize_includes_y_other_when_present():
    """Y_OTHER 는 학습 제외지만 분포 표시 시점에는 포함되어야 (드리프트 가시화)."""
    df = pd.DataFrame({"yellow_subtype": ["Y_OTHER"] * 3 + ["Y_MIX"] * 1})
    out = summarize_yellow_subtypes(df)
    labels = set(out["yellow_subtype"])
    assert "Y_OTHER" in labels
    assert "Y_MIX" in labels
    # ACTION_BY_LABEL 에 Y_OTHER 없음 → 기본값 "알림 없음"
    action_by_label = dict(zip(out["yellow_subtype"], out["action"]))
    assert action_by_label["Y_OTHER"] == "알림 없음"


# ─────────────────────────────────────────────────────────────────────────────
# count_red_suspect
# ─────────────────────────────────────────────────────────────────────────────

def test_count_red_suspect_returns_none_when_missing():
    df = pd.DataFrame({"risk_level": ["Red", "Yellow"]})
    assert count_red_suspect(df) is None


def test_count_red_suspect_counts_true():
    df = pd.DataFrame({"red_suspect": [True, False, True, True, False]})
    assert count_red_suspect(df) == 3


def test_count_red_suspect_handles_null_as_false():
    df = pd.DataFrame({"red_suspect": [True, None, True, pd.NA, False]})
    assert count_red_suspect(df) == 2


# ─────────────────────────────────────────────────────────────────────────────
# 색상 팔레트 무결성
# ─────────────────────────────────────────────────────────────────────────────

def test_yellow_subtype_colors_covers_all_stage2_yellow_labels():
    """모든 Stage 2 Yellow 라벨 + Y_OTHER 가 색상 매핑에 있어야."""
    from hana_app.core.hierarchical_runner import YELLOW_SUBTYPE_LABELS
    for lbl in YELLOW_SUBTYPE_LABELS:
        assert lbl in YELLOW_SUBTYPE_COLORS, f"Y_SUBTYPE_COLORS 에 {lbl} 누락"
    # Y_OTHER 는 STAGE2_LABELS 에 없지만 드리프트 모니터링용으로 팔레트 필요
    assert "Y_OTHER" in YELLOW_SUBTYPE_COLORS
