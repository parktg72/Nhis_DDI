"""Stage N: run_interaction 가드 + Cox 침묵 실패 + 임계값 설정화 테스트"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from statistical_analysis import StatisticalAnalyzer, SamplingInfo


def _make_analyzer(df):
    """테스트용 StatisticalAnalyzer — _cached_df 직접 주입."""
    analyzer = StatisticalAnalyzer.__new__(StatisticalAnalyzer)
    analyzer._cached_df = df
    analyzer._sampling_info = SamplingInfo(applied=False, total_rows=len(df), sampled_rows=len(df))
    analyzer.results = {}
    analyzer.db_path = ':memory:'
    return analyzer


def test_run_interaction_returns_none_when_too_few_rows():
    """run_interaction: MIN_VALID_ROWS 미만이면 None 반환 (Cox 시도 안 함)."""
    n = 5  # MIN_VALID_ROWS=30 보다 훨씬 적음
    df = pd.DataFrame({
        'exposure_group': ['T1DM'] * n,
        'is_t1dm': [1] * n,
        'dm_duration_cat': ['<5yr'] * n,
        'age_at_index': [50.0] * n,
        'male': [1] * n,
        'income_q': [5] * n,
        'cci_score': [0] * n,
        'follow_up_years': [1.0] * n,
        'dementia_event': [0] * n,
    })
    analyzer = _make_analyzer(df)
    with patch('statistical_analysis.STUDY_SETTINGS',
               {'MIN_VALID_ROWS': 30, 'MIN_EVENTS': 10, 'SAMPLING_SEED': 42}):
        result = analyzer.run_interaction(df_prepared=df)
    assert result is None, \
        f"MIN_VALID_ROWS 미만 데이터에서 run_interaction 이 None 을 반환해야 함: {result}"


def test_run_interaction_returns_none_when_too_few_events():
    """run_interaction: MIN_EVENTS 미만이면 None 반환."""
    n = 50
    df = pd.DataFrame({
        'exposure_group': ['T1DM'] * n,
        'is_t1dm': [1] * n,
        'dm_duration_cat': ['<5yr'] * n,
        'age_at_index': [50.0] * n,
        'male': [1] * n,
        'income_q': [5] * n,
        'cci_score': [0] * n,
        'follow_up_years': [1.0] * n,
        'dementia_event': [0] * (n - 2) + [1, 1],  # 이벤트 2건 (MIN_EVENTS=10 미만)
    })
    analyzer = _make_analyzer(df)
    with patch('statistical_analysis.STUDY_SETTINGS',
               {'MIN_VALID_ROWS': 30, 'MIN_EVENTS': 10, 'SAMPLING_SEED': 42}):
        result = analyzer.run_interaction(df_prepared=df)
    assert result is None, \
        f"MIN_EVENTS 미만 이벤트에서 run_interaction 이 None 을 반환해야 함: {result}"
