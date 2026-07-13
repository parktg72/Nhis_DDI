"""train_model 입력 validation 회귀 테스트.

UI 또는 호출자의 오입력이 학습 파이프라인 깊숙한 곳에서 크래시하기 전에
명확한 ValueError 로 조기 차단되는지 검증.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hana_app.core.ml_runner import _build_misclassified_cases, train_model


@pytest.mark.parametrize("bad_model", ["foo", "XGBoost", "", None])
def test_rejects_invalid_model_name(bad_model):
    with pytest.raises(ValueError, match="model_name"):
        train_model(df=None, model_name=bad_model, target="risk_binary")


@pytest.mark.parametrize("bad_target", ["foo", "risk", "binary", ""])
def test_rejects_invalid_target(bad_target):
    with pytest.raises(ValueError, match="target"):
        train_model(df=None, model_name="xgboost", target=bad_target)


@pytest.mark.parametrize("bad_test_size", [0.0, 1.0, -0.1, 1.5, 10.0])
def test_rejects_invalid_test_size(bad_test_size):
    with pytest.raises(ValueError, match="test_size"):
        train_model(
            df=None, model_name="xgboost", target="risk_binary",
            test_size=bad_test_size,
        )


@pytest.mark.parametrize("bad_cv", [0, 1, -5])
def test_rejects_invalid_cv_folds(bad_cv):
    with pytest.raises(ValueError, match="cv_folds"):
        train_model(
            df=None, model_name="xgboost", target="risk_binary",
            cv_folds=bad_cv,
        )


def test_rejects_negative_sampling_size():
    with pytest.raises(ValueError, match="sampling_size"):
        train_model(
            df=None, model_name="xgboost", target="risk_binary",
            sampling_size=-100,
        )


def test_rejects_zero_cost():
    with pytest.raises(ValueError, match="cost_fp and cost_fn"):
        train_model(
            df=None, model_name="xgboost", target="risk_binary",
            cost_fp=0.0, cost_fn=5.0,
        )


def test_build_misclassified_cases_excludes_identifiers():
    import pandas as pd

    x_test = pd.DataFrame([
        {
            "patient_id": "SECRET001",
            "drug_count": 12,
            "ddi_major": 2,
            "age": 91,
            "sex_m": 1,
            "has_high_risk_drug": 1,
        },
        {
            "patient_id": "SECRET002",
            "drug_count": 3,
            "ddi_major": 0,
            "age": 67,
            "sex_m": 0,
            "has_high_risk_drug": 0,
        },
    ])
    cases = _build_misclassified_cases(
        x_test=x_test,
        y_true=[3, 1],
        y_pred=[2, 3],
        y_proba=[0.49, 0.92],
        class_names={3: "Red", 2: "Yellow", 1: "Green"},
    )

    assert len(cases) == 2
    payload = str(cases)
    assert "SECRET" not in payload
    assert "patient_id" not in payload
    assert "age" not in payload
    assert "sex_m" not in payload
    assert cases[0]["actual"] == "Red"
    assert cases[0]["predicted"] == "Yellow"
    assert cases[0]["features"]["drug_count"] == 12
    assert cases[0]["features"]["ddi_major"] == 2
    assert cases[0]["features"]["has_high_risk_drug"] == 1
