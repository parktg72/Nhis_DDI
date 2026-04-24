"""Stage 2 6-class sample_weight 단위 테스트."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hana_app.core.hierarchical_runner import (
    STAGE2_LABELS,
    encode_stage2_labels,
    _stage2_sample_weight,
)


def test_balanced_minority_has_higher_weight():
    """희소 Y_MIX 가 다수 No_Alert 보다 훨씬 큰 가중치."""
    labels = ["Y_MIX"] * 1 + ["No_Alert"] * 100
    y, _enc = encode_stage2_labels(labels)
    sw = _stage2_sample_weight(y, cost_sensitive=False)
    y_mix_idx = encode_stage2_labels(["Y_MIX"])[0][0]
    no_alert_idx = encode_stage2_labels(["No_Alert"])[0][0]
    y_mix_w = sw[y == y_mix_idx][0]
    no_alert_w = sw[y == no_alert_idx][0]
    assert y_mix_w > no_alert_w * 50


def test_cost_sensitive_multiplies_balanced_by_ratio():
    """각 클래스 1건씩 → balanced=1.0 → sw = cost_ratio 그대로."""
    labels = list(STAGE2_LABELS)
    y, _enc = encode_stage2_labels(labels)
    cost_ratio = {
        "Y_MIX": 3.0, "Y_DDI_MAJOR": 2.5, "Y_DDI_MOD": 1.0,
        "Y_DUP": 1.0, "Y_FRAG": 0.8, "No_Alert": 0.5,
    }
    sw = _stage2_sample_weight(y, cost_sensitive=True, cost_ratio_by_class=cost_ratio)
    expected = np.array([cost_ratio[lbl] for lbl in labels])
    np.testing.assert_allclose(sw, expected)


def test_cost_sensitive_without_ratio_returns_balanced():
    labels = ["Y_MIX", "No_Alert", "No_Alert"]
    y, _enc = encode_stage2_labels(labels)
    sw = _stage2_sample_weight(y, cost_sensitive=True, cost_ratio_by_class=None)
    y_mix_idx = encode_stage2_labels(["Y_MIX"])[0][0]
    assert sw[y == y_mix_idx][0] > sw[y != y_mix_idx][0]


def test_unknown_class_in_ratio_raises():
    """cost_ratio 에 STAGE2_LABELS 에 없는 키 → 명시적 오류."""
    y, _enc = encode_stage2_labels(["Y_MIX", "No_Alert"])
    with pytest.raises(KeyError, match="Y_UNKNOWN"):
        _stage2_sample_weight(
            y, cost_sensitive=True,
            cost_ratio_by_class={"Y_UNKNOWN": 2.0},
        )


def test_xgboost_fit_accepts_stage2_sample_weight():
    """XGBoost 6-class fit 이 sample_weight 수용."""
    from xgboost import XGBClassifier

    rng = np.random.default_rng(42)
    labels = ["Y_MIX"] * 5 + ["Y_DDI_MAJOR"] * 10 + ["Y_DDI_MOD"] * 30 \
             + ["Y_DUP"] * 10 + ["Y_FRAG"] * 15 + ["No_Alert"] * 30
    y, _enc = encode_stage2_labels(labels)
    X = rng.random((len(y), 3))
    sw = _stage2_sample_weight(y, cost_sensitive=False)
    clf = XGBClassifier(
        n_estimators=5, max_depth=3,
        objective="multi:softprob", num_class=len(STAGE2_LABELS),
        verbosity=0,
    )
    clf.fit(X, y, sample_weight=sw)
    pred = clf.predict(X)
    assert pred.shape == (len(y),)
