"""Y_OTHER 모니터링 순수 함수 테스트.

compute_y_other_snapshot / compare_snapshots 의 4 가지 측면:
- 정상 분포 / 컬럼 없음 / 비율 정확성 / 드리프트 알람 임계
"""
from __future__ import annotations

import pandas as pd
import pytest

from hana_app.core.y_other_monitoring import (
    DEFAULT_DELTA_PP,
    DEFAULT_DELTA_RELATIVE,
    compare_snapshots,
    compute_y_other_snapshot,
)

# ─────────────────────────────────────────────────────────────────────────────
# compute_y_other_snapshot
# ─────────────────────────────────────────────────────────────────────────────

def test_snapshot_no_column_returns_zeros():
    df = pd.DataFrame({"risk_level": ["Yellow"] * 10})
    snap = compute_y_other_snapshot(df)
    assert snap["y_other_count"] == 0
    assert snap["total_yellow_count"] == 0
    assert snap["has_yellow_subtype_col"] is False
    assert snap["total_count"] == 10


def test_snapshot_empty_df():
    df = pd.DataFrame({"yellow_subtype": []})
    snap = compute_y_other_snapshot(df)
    assert snap["total_count"] == 0
    assert snap["y_other_pct_total"] == 0.0
    assert snap["y_other_pct_in_yellow"] == 0.0


def test_snapshot_correct_counts_and_percentages():
    df = pd.DataFrame({
        "yellow_subtype": (
            ["Y_TRIPLE"] * 10
            + ["Y_DDI_MAJOR"] * 5
            + ["Y_OTHER"] * 5
            + [None] * 80   # Green/Normal/Red 등 — yellow 아님
        ),
    })
    snap = compute_y_other_snapshot(df)
    assert snap["y_other_count"] == 5
    assert snap["total_yellow_count"] == 20  # 10 + 5 + 5
    assert snap["y_other_pct_in_yellow"] == pytest.approx(25.0)
    assert snap["total_count"] == 100
    assert snap["y_other_pct_total"] == pytest.approx(5.0)
    assert snap["has_yellow_subtype_col"] is True


def test_snapshot_no_yellow_at_all():
    """yellow_subtype 컬럼은 있지만 모두 None — Yellow 0건."""
    df = pd.DataFrame({"yellow_subtype": [None] * 50})
    snap = compute_y_other_snapshot(df)
    assert snap["y_other_count"] == 0
    assert snap["total_yellow_count"] == 0
    assert snap["y_other_pct_in_yellow"] == 0.0   # 0/0 → 0
    assert snap["has_yellow_subtype_col"] is True


# ─────────────────────────────────────────────────────────────────────────────
# compare_snapshots
# ─────────────────────────────────────────────────────────────────────────────

def _make_snap(pct: float) -> dict:
    """Test helper — y_other_pct_in_yellow 만 의미 있는 미니 스냅샷."""
    return {
        "y_other_count": 0, "total_yellow_count": 100,
        "y_other_pct_in_yellow": pct,
        "total_count": 1000, "y_other_pct_total": pct / 10,
        "has_yellow_subtype_col": True,
    }


def test_compare_no_drift_below_thresholds():
    cmp = compare_snapshots(_make_snap(10.0), _make_snap(10.5))
    # delta=-0.5pp, relative=-4.8% — 어떤 임계도 미달
    assert cmp["alert"] is False
    assert cmp["reasons"] == []
    assert "✅" in cmp["message"]


def test_compare_alert_on_pp_threshold():
    """절대 증가 5pp 이상 → alert."""
    cmp = compare_snapshots(_make_snap(15.0), _make_snap(8.0))
    assert cmp["alert"] is True
    assert cmp["delta_pp"] == pytest.approx(7.0)
    assert any("절대 증가" in r for r in cmp["reasons"])


def test_compare_alert_on_relative_threshold():
    """상대 증가 20% 이상 (절대 5pp 미만) → alert."""
    # 4 → 5 (절대 +1pp, 상대 +25%)
    cmp = compare_snapshots(_make_snap(5.0), _make_snap(4.0))
    assert cmp["alert"] is True
    assert cmp["delta_pp"] == pytest.approx(1.0)
    assert cmp["delta_relative"] == pytest.approx(0.25)
    assert any("상대 증가" in r for r in cmp["reasons"])


def test_compare_baseline_zero_current_positive():
    """baseline=0 인데 현재>0 — '신규 출현' 알람."""
    cmp = compare_snapshots(_make_snap(2.0), _make_snap(0.0))
    assert cmp["alert"] is True
    assert cmp["delta_relative"] == float("inf")
    assert any("신규 출현" in r for r in cmp["reasons"])


def test_compare_both_zero_no_alert():
    cmp = compare_snapshots(_make_snap(0.0), _make_snap(0.0))
    assert cmp["alert"] is False
    assert cmp["delta_relative"] == 0.0


def test_compare_decrease_no_alert():
    """비율 하락은 알람 없음 (드리프트 모니터링은 상승 감지가 목적)."""
    cmp = compare_snapshots(_make_snap(5.0), _make_snap(10.0))
    assert cmp["alert"] is False
    assert cmp["delta_pp"] == pytest.approx(-5.0)


def test_compare_thresholds_overridable():
    """기본 임계 무시하고 더 엄격한 임계 → 알람 발생."""
    cmp = compare_snapshots(
        _make_snap(10.5), _make_snap(10.0),
        delta_pp_threshold=0.4,  # +0.5pp 가 임계 0.4 초과
        delta_relative_threshold=10.0,  # 상대는 무시
    )
    assert cmp["alert"] is True
    assert any("절대 증가" in r for r in cmp["reasons"])


# ─────────────────────────────────────────────────────────────────────────────
# 기본 임계 합리성 — 실수로 너무 느슨/엄격하지 않은지
# ─────────────────────────────────────────────────────────────────────────────

def test_default_thresholds_reasonable():
    """기본값이 합리적 범위 — 5pp 절대, 20% 상대."""
    assert 1.0 <= DEFAULT_DELTA_PP <= 20.0
    assert 0.05 <= DEFAULT_DELTA_RELATIVE <= 1.0
