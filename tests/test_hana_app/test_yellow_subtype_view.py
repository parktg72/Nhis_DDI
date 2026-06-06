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
    RED_ACTION,
    YELLOW_SUBTYPE_COLORS,
    count_red_suspect,
    summarize_actions,
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
            ["Y_TRIPLE"] * 3
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
        "Y_TRIPLE": 3,
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
    df = pd.DataFrame({"yellow_subtype": ["Y_OTHER"] * 3 + ["Y_TRIPLE"] * 1})
    out = summarize_yellow_subtypes(df)
    labels = set(out["yellow_subtype"])
    assert "Y_OTHER" in labels
    assert "Y_TRIPLE" in labels
    # ACTION_BY_LABEL 에 Y_OTHER 없음 → 기본값 "알림 없음"
    action_by_label = dict(zip(out["yellow_subtype"], out["action"]))
    assert action_by_label["Y_OTHER"] == "알림 없음"


# ─────────────────────────────────────────────────────────────────────────────
# summarize_actions — Red("즉각 개입") + Yellow 개입 합산
# ─────────────────────────────────────────────────────────────────────────────

def test_summarize_actions_includes_red():
    """Red(risk_level=='Red') 가 '즉각 개입' 으로 개입 분포에 합산된다."""
    df = pd.DataFrame({
        "risk_level": (["Red"] * 4 + ["Yellow"] * 3 + ["Green"] * 2),
        "yellow_subtype": ([None] * 4 + ["Y_TRIPLE"] * 3 + [None] * 2),
    })
    out = summarize_actions(df)
    actions = dict(zip(out["action"], out["count"]))
    # Red(4) + Y_TRIPLE(3) 모두 즉각 개입 (2026-06-06 재설계: Y_TRIPLE 액션 상향)
    assert actions[RED_ACTION] == 7
    assert "의료인 전화" not in actions
    # count 내림차순 정렬 보장
    assert list(out["count"]) == sorted(out["count"], reverse=True)


def test_summarize_actions_merges_same_action():
    """같은 action(Y_DOUBLE/Y_DDI_MOD/Y_FRAG = '문자 알림')은 합산된다."""
    df = pd.DataFrame({
        "risk_level": ["Yellow"] * 5,
        "yellow_subtype": ["Y_DOUBLE", "Y_DDI_MOD", "Y_FRAG", "Y_DOUBLE", "Y_DUP"],
    })
    out = summarize_actions(df)
    actions = dict(zip(out["action"], out["count"]))
    assert actions["문자 알림"] == 4              # Y_DOUBLE×2 + Y_DDI_MOD + Y_FRAG
    assert actions["문서 + 문자 알림"] == 1       # Y_DUP


def test_summarize_actions_empty_without_red_or_yellow():
    df = pd.DataFrame({"risk_level": ["Green", "Normal"]})
    out = summarize_actions(df)
    assert out.empty
    assert list(out.columns) == ["action", "count"]


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
