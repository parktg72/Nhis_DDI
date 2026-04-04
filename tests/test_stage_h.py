"""
tests/test_stage_h.py - Stage H 가드 적용 범위 확대 테스트
"""

import pytest
import pandas as pd
from unittest.mock import patch
from utils import format_error_for_user, InsufficientDataError
from statistical_analysis import StatisticalAnalyzer, SamplingInfo


def test_insufficient_data_error_default_kind_is_rows():
    """kind 미지정 시 기본값 'rows' 여야 한다 (하위호환)."""
    exc = InsufficientDataError(valid_rows=5, min_rows=30)
    assert exc.kind == "rows"


def test_insufficient_data_error_kind_events():
    """kind='events' 로 생성 가능해야 한다."""
    exc = InsufficientDataError(valid_rows=3, min_rows=10, kind="events")
    assert exc.kind == "events"


def test_format_error_rows_kind_mentions_min_valid_rows():
    """rows 종류 에러는 MIN_VALID_ROWS 설정을 안내해야 한다."""
    exc = InsufficientDataError(valid_rows=5, min_rows=30, kind="rows")
    msg = format_error_for_user(exc)
    assert "MIN_VALID_ROWS" in msg, f"MIN_VALID_ROWS 언급 없음: {msg!r}"
    assert "MIN_EVENTS" not in msg, f"잘못된 설정 키 MIN_EVENTS 언급: {msg!r}"


def test_format_error_events_kind_mentions_min_events():
    """events 종류 에러는 MIN_EVENTS 설정을 안내해야 한다."""
    exc = InsufficientDataError(valid_rows=3, min_rows=10, kind="events")
    msg = format_error_for_user(exc)
    assert "MIN_EVENTS" in msg, f"MIN_EVENTS 언급 없음: {msg!r}"
    assert "MIN_VALID_ROWS" not in msg, f"잘못된 설정 키 MIN_VALID_ROWS 언급: {msg!r}"


def _make_analyzer_with_df(df):
    """미리 준비된 df 를 _cached_df 로 주입한 분석기."""
    analyzer = StatisticalAnalyzer.__new__(StatisticalAnalyzer)
    analyzer.results = {}
    analyzer._cached_df = df
    analyzer._sampling_info = SamplingInfo(applied=False, total_rows=len(df), sampled_rows=len(df))
    return analyzer


def test_run_subgroup_respects_min_valid_rows_from_config():
    """run_subgroup 이 하드코딩 100 대신 MIN_VALID_ROWS 를 사용한다.

    MIN_VALID_ROWS=30 으로 패치하면 남성 30건 서브그룹이 실행되어야 함.
    하드코딩 100 이라면 30건은 skip 되어 결과에 sex_male 이 없음.
    CoxPHFitter 는 목 처리하여 합성 데이터 수렴 실패를 방지한다.
    """
    from unittest.mock import MagicMock
    n = 50
    events = [1] * 6 + [0] * 24 + [0] * 20  # 6 male events (rows 0-29), 0 female events
    df = pd.DataFrame({
        'follow_up_years': [float(i % 5 + 1) for i in range(n)],
        'dementia_event': events,
        'exposure_group': ['T2DM_OHA'] * n,
        'is_t1dm': [0] * n,
        'is_t2dm_oha': [1] * n,
        'is_t2dm_insulin': [0] * n,
        'is_t2dm_nomed': [0] * n,
        'male': [1] * 30 + [0] * 20,
        'age_at_index': [50.0 + (i % 25) for i in range(n)],
        'cci_score': [i % 4 for i in range(n)],
        'age_group': ['55-64'] * n,
    })
    analyzer = _make_analyzer_with_df(df)

    # Mock CoxPHFitter so synthetic data does not cause convergence errors.
    mock_summary = pd.DataFrame(
        {'exp(coef)': [1.0], 'exp(coef) lower 95%': [0.9], 'exp(coef) upper 95%': [1.1], 'p': [0.5]},
        index=['is_t2dm_oha'],
    )
    mock_cph = MagicMock()
    mock_cph.summary = mock_summary

    with patch('statistical_analysis.STUDY_SETTINGS', {'MIN_VALID_ROWS': 30, 'MIN_EVENTS': 10, 'SAMPLING_SEED': 42}), \
         patch('statistical_analysis.CoxPHFitter', return_value=mock_cph):
        result = analyzer.run_subgroup(df_prepared=df)
    assert 'sex_male' in result, \
        f"MIN_VALID_ROWS=30 인데도 sex_male(30건) 서브그룹이 skip 됨 — 하드코딩 100 사용 중. result keys: {list(result.keys())}"
