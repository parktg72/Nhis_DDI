"""
tests/test_stage_i.py - Stage I 가드 마무리 폴리시 테스트
"""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from statistical_analysis import StatisticalAnalyzer, SamplingInfo
from utils import format_error_for_user, InsufficientDataError


def _make_analyzer_with_df(df):
    analyzer = StatisticalAnalyzer.__new__(StatisticalAnalyzer)
    analyzer.results = {}
    analyzer._cached_df = df
    analyzer._sampling_info = SamplingInfo(applied=False, total_rows=len(df), sampled_rows=len(df))
    return analyzer


def test_run_psm_skip_reason_uses_format_error_for_user():
    """run_psm skip reason 이 format_error_for_user 메시지를 포함해야 한다.

    MIN_VALID_ROWS=30 으로 패치, 10건 df → skip 됨.
    reason 에 'MIN_VALID_ROWS' 가 포함되어야 한다 (format_error_for_user 경유).
    현재 str(e) 사용 시 reason 에는 원본 예외 메시지만 포함됨.
    """
    n = 10
    df = pd.DataFrame({
        'follow_up_years': [1.0] * n,
        'dementia_event': [0] * n,
        'ad_event': [0] * n,
        'vad_event': [0] * n,
        'exposure_group': (['T1DM'] * 5) + (['T2DM_OHA'] * 5),
        'is_t1dm': [1] * 5 + [0] * 5,
        'is_t2dm_oha': [0] * 5 + [1] * 5,
        'is_t2dm_insulin': [0] * n,
        'is_t2dm_nomed': [0] * n,
        'male': [1] * n,
        'age_at_index': [60.0] * n,
        'income_q': [3.0] * n,
        'comor_hypertension': [0] * n,
        'comor_dyslipidemia': [0] * n,
        'dm_duration_years': [5.0] * n,
    })
    analyzer = StatisticalAnalyzer.__new__(StatisticalAnalyzer)
    analyzer.results = {}
    with patch('statistical_analysis.STUDY_SETTINGS',
               {'MIN_VALID_ROWS': 30, 'MIN_EVENTS': 10, 'MIN_SUBGROUP_EVENTS': 5, 'SAMPLING_SEED': 42}):
        result = analyzer.run_psm(df_prepared=df)
    assert result.get('skipped') is True
    reason = result.get('reason', '')
    # format_error_for_user 경유 시 "유효 데이터 부족:" 형식, str(e) 경유 시 "유효 행 수(" 형식
    assert '유효 데이터 부족' in reason, \
        f"skip reason 에 '유효 데이터 부족' 없음 — format_error_for_user 미사용(str(e) 사용 중): {reason!r}"


def test_run_subgroup_respects_min_subgroup_events():
    """run_subgroup 이 하드코딩 5 대신 MIN_SUBGROUP_EVENTS 를 사용한다.

    MIN_SUBGROUP_EVENTS=10 으로 패치하면 이벤트 5건인 서브그룹이 skip 되어야 한다.
    CoxPHFitter 를 statistical_analysis 모듈 수준에서 패치하여 성공 반환 — 이벤트 임계값만 검사.
    """
    n = 50
    # 이벤트 5건, 남성/여성 충분 (30/20), MIN_VALID_ROWS=30 통과
    df = pd.DataFrame({
        'follow_up_years': [1.0] * n,
        'dementia_event': [1] * 5 + [0] * (n - 5),
        'exposure_group': ['T2DM_OHA'] * 25 + ['T1DM'] * 25,
        'is_t1dm': [0] * 25 + [1] * 25,
        'is_t2dm_oha': [1] * 25 + [0] * 25,
        'is_t2dm_insulin': [0] * n,
        'is_t2dm_nomed': [0] * n,
        'male': [1] * 30 + [0] * 20,
        'age_at_index': [60.0] * n,
        'cci_score': [1] * n,
        'age_group': ['55-64'] * n,
    })
    analyzer = _make_analyzer_with_df(df)

    mock_summary = pd.DataFrame(
        {'exp(coef)': [1.1], 'exp(coef) lower 95%': [0.9],
         'exp(coef) upper 95%': [1.3], 'p': [0.3]},
        index=['is_t1dm']
    )

    # MIN_SUBGROUP_EVENTS=4 이면 이벤트 5건 서브그룹이 실행돼야 함 → result 비지 않음
    with patch('statistical_analysis.STUDY_SETTINGS',
               {'MIN_VALID_ROWS': 30, 'MIN_EVENTS': 10, 'MIN_SUBGROUP_EVENTS': 4, 'SAMPLING_SEED': 42}):
        with patch('statistical_analysis.CoxPHFitter') as mock_cph:
            mock_cph.return_value.fit.return_value = None
            mock_cph.return_value.summary = mock_summary
            mock_cph.return_value.concordance_index_ = 0.6
            result_runs = analyzer.run_subgroup(df_prepared=df)

    # MIN_SUBGROUP_EVENTS=10 이면 이벤트 5건 서브그룹은 skip 되어야 함 → result 비어야 함
    with patch('statistical_analysis.STUDY_SETTINGS',
               {'MIN_VALID_ROWS': 30, 'MIN_EVENTS': 10, 'MIN_SUBGROUP_EVENTS': 10, 'SAMPLING_SEED': 42}):
        with patch('statistical_analysis.CoxPHFitter') as mock_cph:
            mock_cph.return_value.fit.return_value = None
            mock_cph.return_value.summary = mock_summary
            mock_cph.return_value.concordance_index_ = 0.6
            result_skips = analyzer.run_subgroup(df_prepared=df)

    assert len(result_runs) > 0, \
        f"MIN_SUBGROUP_EVENTS=4 인데 이벤트 5건 서브그룹이 실행 안 됨 (하드코딩 5 사용 중일 수 있음)"
    assert len(result_skips) == 0, \
        f"MIN_SUBGROUP_EVENTS=10 인데 이벤트 5건 서브그룹이 skip 안 됨: {list(result_skips.keys())}"


def test_run_competing_risks_dementia_event_no_duplicate_column_error():
    """outcome='dementia_event' 일 때 need_cols 중복으로 인한 오류 없이 실행돼야 한다.

    Stage H 즉시 반영: dict.fromkeys 로 need_cols 중복 제거.
    이 테스트는 회귀 방지 — 중복 dedup 이 제거되면 여기서 실패한다.
    """
    n = 35
    df = pd.DataFrame({
        'follow_up_years': [1.0] * n,
        'dementia_event': [1] * 5 + [0] * (n - 5),
        'competing_death_event': [0] * n,
        'is_t1dm': [0] * n,
        'is_t2dm_oha': [1] * n,
        'is_t2dm_insulin': [0] * n,
        'is_t2dm_nomed': [0] * n,
        'age_at_index': [60.0] * n,
        'male': [1] * n,
    })
    analyzer = _make_analyzer_with_df(df)
    with patch('statistical_analysis.STUDY_SETTINGS',
               {'MIN_VALID_ROWS': 30, 'MIN_EVENTS': 10, 'MIN_SUBGROUP_EVENTS': 5, 'SAMPLING_SEED': 42}):
        with patch('gpu_accelerator.is_gpu_enabled', return_value=False):
            with patch('gpu_accelerator.compute_cif_gpu', return_value=None):
                # 예외 없이 실행되어야 함 (중복 컬럼 IndexError 방지)
                result = analyzer.run_competing_risks(df_prepared=df)
    # dementia_event 키가 결과에 있어야 함
    assert 'dementia_event' in result, \
        f"outcome='dementia_event' 결과 없음 (중복 컬럼 버그 재발 의심): {list(result.keys())}"
