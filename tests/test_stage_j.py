"""
tests/test_stage_j.py - Stage J: CIF per-group 이벤트 가드 + MIN_SUBGROUP_EVENTS 테스트
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch
from statistical_analysis import StatisticalAnalyzer, SamplingInfo


def _make_analyzer_with_df(df):
    analyzer = StatisticalAnalyzer.__new__(StatisticalAnalyzer)
    analyzer.results = {}
    analyzer._cached_df = df
    analyzer._sampling_info = SamplingInfo(applied=False, total_rows=len(df), sampled_rows=len(df))
    return analyzer


def test_cif_skips_group_with_zero_events():
    """CIF per-group 루프가 이벤트 0건 그룹을 skip 해야 한다.

    T1DM: 15행, 0 이벤트 → MIN_SUBGROUP_EVENTS=3 → CIF skip 되어야 함
    T2DM_OHA: 25행, 5 이벤트 → CIF 포함되어야 함
    현재 코드는 행 수만 확인하므로 T1DM 도 포함됨 — 이 테스트는 수정 전 FAIL.
    """
    n = 40
    df = pd.DataFrame({
        'follow_up_years': [1.0] * n,
        'dementia_event': [0] * 15 + [1] * 5 + [0] * 20,  # T1DM=0건, T2DM_OHA=5건
        'competing_death_event': [0] * n,
        'is_t1dm': [1] * 15 + [0] * 25,
        'is_t2dm_oha': [0] * 15 + [1] * 25,
        'is_t2dm_insulin': [0] * n,
        'is_t2dm_nomed': [0] * n,
        'age_at_index': [60.0] * n,
        'male': [1] * n,
    })
    analyzer = _make_analyzer_with_df(df)
    with patch('statistical_analysis.STUDY_SETTINGS',
               {'MIN_VALID_ROWS': 10, 'MIN_EVENTS': 3, 'MIN_SUBGROUP_EVENTS': 3, 'SAMPLING_SEED': 42}):
        with patch('gpu_accelerator.is_gpu_enabled', return_value=False):
            result = analyzer.run_competing_risks(df_prepared=df)
    cif = result.get('dementia_event', {}).get('cif_by_group', {})
    assert 'T1DM' not in cif, \
        f"이벤트 0건 T1DM 이 CIF 에 포함됨 — 이벤트 수 가드 미적용: {list(cif.keys())}"
    assert 'T2DM_OHA' in cif, \
        f"이벤트 5건 T2DM_OHA 가 CIF 에서 누락됨: {list(cif.keys())}"


def test_cif_respects_min_subgroup_events_threshold():
    """MIN_SUBGROUP_EVENTS 를 임계값 위아래로 패치해 CIF 포함/skip 전환을 검증한다.

    T2DM_OHA: 25행, 4 이벤트
    MIN_SUBGROUP_EVENTS=3 → 포함 (4 >= 3)
    MIN_SUBGROUP_EVENTS=5 → skip  (4 < 5)
    """
    n = 40
    df = pd.DataFrame({
        'follow_up_years': [1.0] * n,
        'dementia_event': [0] * 15 + [1] * 4 + [0] * 21,  # T1DM=0건, T2DM_OHA=4건
        'competing_death_event': [0] * n,
        'is_t1dm': [1] * 15 + [0] * 25,
        'is_t2dm_oha': [0] * 15 + [1] * 25,
        'is_t2dm_insulin': [0] * n,
        'is_t2dm_nomed': [0] * n,
        'age_at_index': [60.0] * n,
        'male': [1] * n,
    })
    analyzer = _make_analyzer_with_df(df)

    # MIN_SUBGROUP_EVENTS=3 → T2DM_OHA (4건) 포함
    with patch('statistical_analysis.STUDY_SETTINGS',
               {'MIN_VALID_ROWS': 10, 'MIN_EVENTS': 3, 'MIN_SUBGROUP_EVENTS': 3, 'SAMPLING_SEED': 42}):
        with patch('gpu_accelerator.is_gpu_enabled', return_value=False):
            result_runs = analyzer.run_competing_risks(df_prepared=df)

    # MIN_SUBGROUP_EVENTS=5 → T2DM_OHA (4건) skip
    with patch('statistical_analysis.STUDY_SETTINGS',
               {'MIN_VALID_ROWS': 10, 'MIN_EVENTS': 3, 'MIN_SUBGROUP_EVENTS': 5, 'SAMPLING_SEED': 42}):
        with patch('gpu_accelerator.is_gpu_enabled', return_value=False):
            result_skips = analyzer.run_competing_risks(df_prepared=df)

    cif_runs = result_runs.get('dementia_event', {}).get('cif_by_group', {})
    cif_skips = result_skips.get('dementia_event', {}).get('cif_by_group', {})
    assert 'T2DM_OHA' in cif_runs, \
        f"MIN_SUBGROUP_EVENTS=3 인데 이벤트 4건 T2DM_OHA 가 CIF 에서 누락됨: {list(cif_runs.keys())}"
    assert 'T2DM_OHA' not in cif_skips, \
        f"MIN_SUBGROUP_EVENTS=5 인데 이벤트 4건 T2DM_OHA 가 CIF 에 포함됨: {list(cif_skips.keys())}"


def test_cif_non_dm_skips_group_with_zero_events():
    """NON_DM CIF 블록이 이벤트 0건일 때 skip 해야 한다.

    NON_DM: 15행, 0 이벤트 → MIN_SUBGROUP_EVENTS=3 → CIF skip 되어야 함
    T2DM_OHA: 35행, 5 이벤트 → CIF 포함되어야 함
    Stage J 에서 추가된 NON_DM 이벤트 수 가드의 회귀 방지 테스트.
    """
    n = 50
    # 행 0-14: is_t1dm=0, is_t2dm_oha=0, ... → NON_DM (15행, 0 이벤트)
    # 행 15-49: is_t2dm_oha=1 → T2DM_OHA (35행, 5 이벤트)
    df = pd.DataFrame({
        'follow_up_years': [1.0] * n,
        'dementia_event': [0] * 15 + [1] * 5 + [0] * 30,  # NON_DM=0건, T2DM_OHA=5건
        'competing_death_event': [0] * n,
        'is_t1dm': [0] * n,
        'is_t2dm_oha': [0] * 15 + [1] * 35,
        'is_t2dm_insulin': [0] * n,
        'is_t2dm_nomed': [0] * n,
        'age_at_index': [60.0] * n,
        'male': [1] * n,
    })
    analyzer = _make_analyzer_with_df(df)
    with patch('statistical_analysis.STUDY_SETTINGS',
               {'MIN_VALID_ROWS': 10, 'MIN_EVENTS': 3, 'MIN_SUBGROUP_EVENTS': 3, 'SAMPLING_SEED': 42}):
        with patch('gpu_accelerator.is_gpu_enabled', return_value=False):
            result = analyzer.run_competing_risks(df_prepared=df)
    cif = result.get('dementia_event', {}).get('cif_by_group', {})
    assert 'NON_DM' not in cif, \
        f"이벤트 0건 NON_DM 이 CIF 에 포함됨 — NON_DM 이벤트 수 가드 미적용: {list(cif.keys())}"
    assert 'T2DM_OHA' in cif, \
        f"이벤트 5건 T2DM_OHA 가 CIF 에서 누락됨: {list(cif.keys())}"


def test_cif_non_dm_respects_min_subgroup_events_threshold():
    """NON_DM CIF 블록이 MIN_SUBGROUP_EVENTS 임계값을 정확히 적용한다.

    NON_DM: 15행, 4 이벤트
    MIN_SUBGROUP_EVENTS=3 → 포함 (4 >= 3)
    MIN_SUBGROUP_EVENTS=5 → skip  (4 < 5)
    """
    n = 50
    # 행 0-14: NON_DM (15행) — 이 중 4건 이벤트
    # 행 15-49: T2DM_OHA (35행, 5 이벤트)
    df = pd.DataFrame({
        'follow_up_years': [1.0] * n,
        'dementia_event': [1] * 4 + [0] * 11 + [1] * 5 + [0] * 30,  # NON_DM=4건, T2DM_OHA=5건
        'competing_death_event': [0] * n,
        'is_t1dm': [0] * n,
        'is_t2dm_oha': [0] * 15 + [1] * 35,
        'is_t2dm_insulin': [0] * n,
        'is_t2dm_nomed': [0] * n,
        'age_at_index': [60.0] * n,
        'male': [1] * n,
    })
    analyzer = _make_analyzer_with_df(df)

    # MIN_SUBGROUP_EVENTS=3 → NON_DM (4건) 포함
    with patch('statistical_analysis.STUDY_SETTINGS',
               {'MIN_VALID_ROWS': 10, 'MIN_EVENTS': 3, 'MIN_SUBGROUP_EVENTS': 3, 'SAMPLING_SEED': 42}):
        with patch('gpu_accelerator.is_gpu_enabled', return_value=False):
            result_runs = analyzer.run_competing_risks(df_prepared=df)

    # MIN_SUBGROUP_EVENTS=5 → NON_DM (4건) skip
    with patch('statistical_analysis.STUDY_SETTINGS',
               {'MIN_VALID_ROWS': 10, 'MIN_EVENTS': 3, 'MIN_SUBGROUP_EVENTS': 5, 'SAMPLING_SEED': 42}):
        with patch('gpu_accelerator.is_gpu_enabled', return_value=False):
            result_skips = analyzer.run_competing_risks(df_prepared=df)

    cif_runs = result_runs.get('dementia_event', {}).get('cif_by_group', {})
    cif_skips = result_skips.get('dementia_event', {}).get('cif_by_group', {})
    assert 'NON_DM' in cif_runs, \
        f"MIN_SUBGROUP_EVENTS=3 인데 이벤트 4건 NON_DM 이 CIF 에서 누락됨: {list(cif_runs.keys())}"
    assert 'NON_DM' not in cif_skips, \
        f"MIN_SUBGROUP_EVENTS=5 인데 이벤트 4건 NON_DM 이 CIF 에 포함됨: {list(cif_skips.keys())}"
