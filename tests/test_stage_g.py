"""
tests/test_stage_g.py - Stage G 분석 견고성 강화 테스트
"""

import pytest
import duckdb
import pandas as pd
from unittest.mock import patch
from utils import format_error_for_user, InsufficientDataError
from statistical_analysis import StatisticalAnalyzer, SamplingInfo


def _make_analyzer_with_conn(conn):
    class MockStorage:
        def get_row_count(self, t):
            return conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    class MockDM:
        storage = MockStorage()
        def query(self, sql):
            return conn.execute(sql).df()
        def execute(self, sql):
            conn.execute(sql)
    analyzer = StatisticalAnalyzer.__new__(StatisticalAnalyzer)
    analyzer.dm = MockDM()
    analyzer.results = {}
    analyzer._cached_df = None
    analyzer._sampling_info = SamplingInfo(applied=False, total_rows=0, sampled_rows=0)
    return analyzer


def test_format_error_for_user_insufficient_data_error():
    """InsufficientDataError 가 사용자 친화적 메시지로 변환되어야 한다."""
    exc = InsufficientDataError(valid_rows=10, min_rows=30)
    msg = format_error_for_user(exc)
    assert "10" in msg or "30" in msg or "최소" in msg, \
        f"InsufficientDataError 전용 메시지 없음: {msg!r}"
    # ValueError 일반 분기("입력값 오류:")로 떨어지면 안 됨
    assert "입력값 오류" not in msg, \
        f"InsufficientDataError 가 일반 ValueError 로 처리됨: {msg!r}"


def test_check_min_rows_raises_on_small_df():
    """_check_min_rows() 가 기준 미달 DataFrame 에서 InsufficientDataError 를 발생시킨다."""
    analyzer = StatisticalAnalyzer.__new__(StatisticalAnalyzer)
    analyzer.results = {}
    small_df = pd.DataFrame({'a': range(5)})
    with pytest.raises(InsufficientDataError):
        analyzer._check_min_rows(small_df, context="테스트")


def test_check_min_rows_passes_on_sufficient_df():
    """_check_min_rows() 가 기준 이상 DataFrame 에서 예외 없이 반환한다."""
    analyzer = StatisticalAnalyzer.__new__(StatisticalAnalyzer)
    analyzer.results = {}
    ok_df = pd.DataFrame({'a': range(30)})
    analyzer._check_min_rows(ok_df, context="테스트")  # 예외 없음
