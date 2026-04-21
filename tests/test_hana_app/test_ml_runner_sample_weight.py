"""XGBoost 4분류 sample_weight 계산 회귀 테스트.

XGBClassifier 는 class_weights 파라미터를 무시하므로 4분류 불균형 데이터에서
fit() 에 sample_weight 를 직접 전달해야 가중치가 적용된다.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hana_app.core.ml_runner import _xgb_multiclass_sample_weight


def test_sample_weight_none_for_binary_target():
    """이진 분류는 scale_pos_weight 로 처리되므로 None 반환."""
    y = np.array([0, 0, 0, 1, 1])
    sw = _xgb_multiclass_sample_weight(
        target="risk_binary", y_train=y,
        cost_sensitive=False, cost_fp=1.0, cost_fn=5.0,
    )
    assert sw is None


def test_sample_weight_balanced_for_multiclass():
    """4분류 + cost_sensitive=False → balanced 가중치.

    소수 클래스(Green 1건)가 다수 클래스(Yellow 100건)보다 훨씬 큰 가중치.
    """
    # Normal=0, Green=1, Yellow=2, Red=3
    y = np.array([0] * 50 + [1] * 1 + [2] * 100 + [3] * 10)
    sw = _xgb_multiclass_sample_weight(
        target="risk_label", y_train=y,
        cost_sensitive=False, cost_fp=1.0, cost_fn=5.0,
    )
    assert sw is not None
    assert len(sw) == len(y)
    # Green(1개)의 가중치가 Yellow(100개)보다 월등히 큼
    green_weight = sw[y == 1][0]
    yellow_weight = sw[y == 2][0]
    assert green_weight > yellow_weight * 50  # 최소 50배 이상


def test_sample_weight_cost_sensitive_for_multiclass():
    """4분류 + cost_sensitive=True → {0: fp, 1: fp*1.5, 2: fn*0.7, 3: fn}."""
    y = np.array([0, 1, 2, 3])
    sw = _xgb_multiclass_sample_weight(
        target="risk_label", y_train=y,
        cost_sensitive=True, cost_fp=1.0, cost_fn=10.0,
    )
    # 각 클래스에 정확히 지정된 비용 반영
    np.testing.assert_allclose(sw, [1.0, 1.5, 7.0, 10.0])


def test_sample_weight_xgboost_fit_accepts_it():
    """실제 XGBoost fit() 이 sample_weight 를 수용하는지 smoke test."""
    from xgboost import XGBClassifier

    rng = np.random.default_rng(42)
    X = rng.random((100, 3))
    y = np.array([0] * 30 + [1] * 5 + [2] * 50 + [3] * 15)
    sw = _xgb_multiclass_sample_weight(
        target="risk_label", y_train=y,
        cost_sensitive=False, cost_fp=1.0, cost_fn=5.0,
    )
    clf = XGBClassifier(
        n_estimators=5, max_depth=3, objective="multi:softprob",
        num_class=4, verbosity=0,
    )
    # 이 호출이 에러 없이 완료되면 통과 (sample_weight 가 실제로 적용됨)
    clf.fit(X, y, sample_weight=sw)
    # 예측 가능해야 함
    pred = clf.predict(X)
    assert pred.shape == (100,)
