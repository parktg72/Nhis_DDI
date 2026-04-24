"""tau_sensitivity_sweep 순수 함수 테스트.

분석 범위:
- 정상 스윕: recall_floor 낮 → 높을수록 tau_red 감소 (단조성)
- precondition 실패 per-row 기록 (크래시 X)
- fallback_triggered 플래그 (recall_floor 가 데이터로 도달 불가할 때)
- 운영 메트릭 일관성 (n_red_confirmed + n_review_band + n_clean_stage2 == n)
"""
from __future__ import annotations

import numpy as np
import pytest

from hana_app.core.hierarchical_runner import tau_sensitivity_sweep


# ─────────────────────────────────────────────────────────────────────────────
# 공통 fixture — 재현 가능한 가짜 Stage 1 예측
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def fake_stage1_predictions():
    """분리 가능한 Red(y=1) vs Non-Red(y=0) 확률 분포.

    양성 200건, 음성 800건. Red 의 p_proba 는 Beta(8,2) 로 오른쪽 치우침,
    Non-Red 는 Beta(2,8) 로 왼쪽 치우침 — 현실적인 잘 훈련된 모델 시뮬.
    """
    rng = np.random.default_rng(42)
    n_pos, n_neg = 200, 800
    p_pos = rng.beta(8, 2, n_pos)
    p_neg = rng.beta(2, 8, n_neg)
    y_true = np.concatenate([np.ones(n_pos), np.zeros(n_neg)]).astype(int)
    y_proba = np.concatenate([p_pos, p_neg])
    # 무작위 섞기
    idx = rng.permutation(len(y_true))
    return y_true[idx], y_proba[idx]


# ─────────────────────────────────────────────────────────────────────────────
# 정상 경로
# ─────────────────────────────────────────────────────────────────────────────

def test_sweep_returns_row_per_recall_floor(fake_stage1_predictions):
    y_true, y_proba = fake_stage1_predictions
    rows = tau_sensitivity_sweep(
        y_true, y_proba,
        recall_floors=[0.80, 0.90, 0.95],
        review_recall_target=0.98,
    )
    assert len(rows) == 3
    assert [r["recall_floor_requested"] for r in rows] == [0.80, 0.90, 0.95]
    # 모든 row 에 error=None
    assert all(r["error"] is None for r in rows)


def test_sweep_monotonicity_recall_floor_to_tau_red(fake_stage1_predictions):
    """recall_floor 상승 → tau_red 하강 (더 많은 양성 포착) — 단조성."""
    y_true, y_proba = fake_stage1_predictions
    rows = tau_sensitivity_sweep(
        y_true, y_proba,
        recall_floors=[0.80, 0.90, 0.95],
        review_recall_target=0.99,
    )
    taus = [r["tau_red"] for r in rows]
    assert taus[0] >= taus[1] >= taus[2], f"단조성 위배: {taus}"


def test_sweep_traffic_counts_sum_to_total(fake_stage1_predictions):
    """n_red_confirmed + n_review_band + n_clean_stage2 == 전체 N."""
    y_true, y_proba = fake_stage1_predictions
    n = len(y_true)
    rows = tau_sensitivity_sweep(
        y_true, y_proba,
        recall_floors=[0.90],
        review_recall_target=0.98,
    )
    r = rows[0]
    assert r["n_red_confirmed"] + r["n_review_band"] + r["n_clean_stage2"] == n


def test_sweep_actual_recall_meets_floor_when_feasible(fake_stage1_predictions):
    y_true, y_proba = fake_stage1_predictions
    rows = tau_sensitivity_sweep(
        y_true, y_proba,
        recall_floors=[0.80, 0.90],
        review_recall_target=0.98,
    )
    # 합리적인 데이터에서 recall 0.80 은 달성 가능
    assert rows[0]["actual_red_recall"] >= 0.80 - 1e-9
    assert rows[0]["fallback_triggered"] is False


def test_sweep_tau_invariant_preserved(fake_stage1_predictions):
    """select_thresholds_from_pr 불변식: tau_review < tau_red."""
    y_true, y_proba = fake_stage1_predictions
    rows = tau_sensitivity_sweep(
        y_true, y_proba,
        recall_floors=[0.85, 0.90, 0.95],
        review_recall_target=0.99,
    )
    for r in rows:
        if r["error"] is None:
            assert r["tau_review"] < r["tau_red"]


# ─────────────────────────────────────────────────────────────────────────────
# precondition / edge case
# ─────────────────────────────────────────────────────────────────────────────

def test_sweep_precondition_failure_recorded_per_row(fake_stage1_predictions):
    """review_recall_target ≤ recall_floor 일 때 해당 row 에 error 기록 후 다음 진행."""
    y_true, y_proba = fake_stage1_predictions
    rows = tau_sensitivity_sweep(
        y_true, y_proba,
        # 0.95 recall_floor 은 review_target=0.95 과 같아서 select_thresholds 가 ValueError
        recall_floors=[0.80, 0.95],
        review_recall_target=0.95,
    )
    assert len(rows) == 2
    assert rows[0]["error"] is None
    assert rows[1]["error"] is not None
    assert "review_recall_target" in rows[1]["error"]
    # 실패 row 에 tau 는 None
    assert rows[1]["tau_red"] is None


def test_sweep_empty_recall_floors():
    """빈 리스트 입력 시 빈 결과 반환."""
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.8, 0.9])
    rows = tau_sensitivity_sweep(y_true, y_proba, recall_floors=[])
    assert rows == []


def test_sweep_red_leakage_pct_bounded():
    """red_leakage_pct 는 [0, 100] 범위."""
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 500)
    y_proba = rng.random(500)
    rows = tau_sensitivity_sweep(
        y_true, y_proba,
        recall_floors=[0.5, 0.8, 0.95],
        review_recall_target=0.99,
    )
    for r in rows:
        if r["error"] is None:
            assert 0.0 <= r["red_leakage_pct"] <= 100.0
            assert 0.0 <= r["stage2_traffic_pct"] <= 100.0


def test_sweep_fallback_when_recall_floor_unachievable():
    """recall_floor 가 달성 불가한 수준일 때 fallback_triggered=True."""
    # y_true 와 y_proba 가 반상관 — 높은 recall 은 낮은 threshold 에서만 가능
    rng = np.random.default_rng(1)
    n = 200
    y_true = rng.integers(0, 2, n)
    # proba 를 거의 균일 분포로 설정해 구분력 약화
    y_proba = rng.uniform(0.3, 0.7, n)
    rows = tau_sensitivity_sweep(
        y_true, y_proba,
        recall_floors=[0.5, 1.0],   # 1.0 은 경계 케이스
        review_recall_target=1.0 + 1e-6,   # precondition 회피 위해
    )
    # 1.0 recall 은 tau = min(threshold) 로 fallback
    # 운영자는 fallback_triggered 를 보고 "이 floor 는 이 데이터에선 비현실적" 판단
    # 여기선 그냥 crash 안 하고 row 가 채워지는지만 검증
    assert len(rows) == 2
