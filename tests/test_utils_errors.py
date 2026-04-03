import pytest
from utils import CohortStepError


def test_cohort_step_error_stores_attributes():
    cause = ValueError("table not found")
    err = CohortStepError(step=3, step_name="투약 분류", cause=cause)
    assert err.step == 3
    assert err.step_name == "투약 분류"
    assert err.cause is cause


def test_cohort_step_error_message_contains_step_info():
    cause = RuntimeError("duckdb error")
    err = CohortStepError(step=2, step_name="당뇨 청구 식별", cause=cause)
    assert "2단계" in str(err)
    assert "당뇨 청구 식별" in str(err)
    assert "duckdb error" in str(err)


def test_cohort_step_error_is_exception():
    err = CohortStepError(step=1, step_name="기본 인구", cause=ValueError("x"))
    assert isinstance(err, Exception)
