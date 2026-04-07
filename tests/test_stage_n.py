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

def test_run_cox_raises_when_all_models_fail():
    """run_cox: 모든 모델 피팅 실패 시 RuntimeError 발생 (침묵 실패 방지)."""
    import pytest as _pytest
    n = 50
    df = pd.DataFrame({
        'exposure_group': ['T1DM'] * 25 + ['T2DM_OHA'] * 25,
        'is_t1dm':        [1] * 25 + [0] * 25,
        'is_t2dm_oha':    [0] * 25 + [1] * 25,
        'is_t2dm_insulin':[0] * n,
        'is_t2dm_nomed':  [0] * n,
        'age_at_index':   [50.0] * n,
        'male':           [1] * n,
        'income_q':       [5] * n,
        'comor_hypertension':  [0] * n,
        'comor_dyslipidemia':  [0] * n,
        'comor_depression':    [0] * n,
        'comp_retinopathy':    [0] * n,
        'comp_nephropathy':    [0] * n,
        'comp_neuropathy':     [0] * n,
        'comor_ischemic_stroke':   [0] * n,
        'comor_hemorrhagic_stroke':[0] * n,
        'comor_ihd':           [0] * n,
        'comor_atrial_fib':    [0] * n,
        'comor_heart_failure': [0] * n,
        'comp_hypoglycemia':   [0] * n,
        'follow_up_years':     [1.0] * n,
        'dementia_event':      [1] * 15 + [0] * 35,
    })
    analyzer = _make_analyzer(df)
    with patch('statistical_analysis.STUDY_SETTINGS',
               {'MIN_VALID_ROWS': 10, 'MIN_EVENTS': 10, 'SAMPLING_SEED': 42,
                'PH_ALPHA': 0.05}):
        with patch('statistical_analysis.CoxPHFitter') as mock_cox_cls:
            mock_cox_cls.return_value.fit.side_effect = ValueError("강제 실패")
            with _pytest.raises(RuntimeError, match="Cox 회귀 분석"):
                analyzer.run_cox(df_prepared=df)


def test_psm_caliper_respects_study_settings():
    """PSM caliper 는 STUDY_SETTINGS['PSM_CALIPER'] 를 사용해야 한다."""
    from config import STUDY_SETTINGS
    assert 'PSM_CALIPER' in STUDY_SETTINGS, \
        "STUDY_SETTINGS 에 PSM_CALIPER 키가 없음"
    assert 'PSM_SMD_THRESHOLD' in STUDY_SETTINGS, \
        "STUDY_SETTINGS 에 PSM_SMD_THRESHOLD 키가 없음"
    assert 'PH_ALPHA' in STUDY_SETTINGS, \
        "STUDY_SETTINGS 에 PH_ALPHA 키가 없음"
    assert STUDY_SETTINGS['PH_ALPHA'] == 0.05
    assert STUDY_SETTINGS['PSM_CALIPER'] == 0.2
    assert STUDY_SETTINGS['PSM_SMD_THRESHOLD'] == 0.1


def _make_cox_df(n=50):
    """Cox 테스트용 최소 DataFrame 생성."""
    return pd.DataFrame({
        'exposure_group':      ['T1DM'] * (n // 2) + ['T2DM_OHA'] * (n // 2),
        'is_t1dm':             [1] * (n // 2) + [0] * (n // 2),
        'is_t2dm_oha':         [0] * (n // 2) + [1] * (n // 2),
        'is_t2dm_insulin':     [0] * n,
        'is_t2dm_nomed':       [0] * n,
        'age_at_index':        [50.0] * n,
        'male':                [1] * n,
        'income_q':            [5] * n,
        'comor_hypertension':  [0] * n,
        'comor_dyslipidemia':  [0] * n,
        'comor_depression':    [0] * n,
        'comp_retinopathy':    [0] * n,
        'comp_nephropathy':    [0] * n,
        'comp_neuropathy':     [0] * n,
        'comor_ischemic_stroke':   [0] * n,
        'comor_hemorrhagic_stroke':[0] * n,
        'comor_ihd':           [0] * n,
        'comor_atrial_fib':    [0] * n,
        'comor_heart_failure': [0] * n,
        'comp_hypoglycemia':   [0] * n,
        'follow_up_years':     [1.0] * n,
        'dementia_event':      [1] * 15 + [0] * (n - 15),
    })


def test_run_cox_skips_model_when_exposure_ph_violated(caplog):
    """H-2: 노출변수가 PH 가정 위반 시 해당 모델만 스킵하고 경고 로그를 남긴다."""
    import logging
    df = _make_cox_df()
    analyzer = _make_analyzer(df)

    # 3개 모델 중 model1만 is_t1dm PH 위반, model2/model3은 정상
    ph_summary_violated = pd.DataFrame({'p': [0.01, 0.5]}, index=['is_t1dm', 'age_at_index'])
    ph_summary_ok = pd.DataFrame({'p': [0.9, 0.8]}, index=['is_t1dm', 'age_at_index'])
    ph_mock_violated = MagicMock()
    ph_mock_violated.summary = ph_summary_violated
    ph_mock_ok = MagicMock()
    ph_mock_ok.summary = ph_summary_ok

    call_count = {'n': 0}

    def ph_side_effect(cph, df_model, time_transform):
        call_count['n'] += 1
        return ph_mock_violated if call_count['n'] == 1 else ph_mock_ok

    with patch('statistical_analysis.STUDY_SETTINGS',
               {'MIN_VALID_ROWS': 10, 'MIN_EVENTS': 5, 'SAMPLING_SEED': 42,
                'PH_ALPHA': 0.05}):
        with patch('statistical_analysis.CoxPHFitter') as mock_cls, \
             patch('statistical_analysis.proportional_hazard_test',
                   side_effect=ph_side_effect):
            mock_cls.return_value.fit.return_value = None
            mock_cls.return_value.summary = pd.DataFrame()
            mock_cls.return_value.concordance_index_ = 0.6
            with caplog.at_level(logging.WARNING, logger='statistical_analysis'):
                result = analyzer.run_cox(df_prepared=df)

    # model1은 PH 위반으로 스킵, model2+model3은 결과에 포함
    assert 'model1_age_sex' not in result, "PH 위반 모델은 결과에서 제외"
    assert 'model2_socio' in result, "정상 모델은 결과에 포함"
    assert 'model3_full' in result, "정상 모델은 결과에 포함"
    assert any('PH 가정 위반' in r.message for r in caplog.records), \
        "PH 위반 경고 로그 필요"


def test_sampling_seed_out_of_range_raises():
    """M3: SAMPLING_SEED 0-99 범위 초과 시 ValueError 발생."""
    import pytest as _pytest
    from statistical_analysis import StatisticalAnalyzer

    mock_dm = MagicMock()
    mock_dm.storage.get_row_count.return_value = 1000  # total > max_rows → 샘플링 분기
    group_df = pd.DataFrame({'exposure_group': ['T1DM', 'NON_DM'], 'cnt': [300, 700]})
    mock_dm.query.return_value = group_df

    analyzer = StatisticalAnalyzer.__new__(StatisticalAnalyzer)
    analyzer.dm = mock_dm
    analyzer._cached_df = None
    analyzer._sampling_info = None
    analyzer.results = {}

    with patch('statistical_analysis.STUDY_SETTINGS',
               {'MIN_VALID_ROWS': 10, 'MIN_EVENTS': 5, 'SAMPLING_SEED': 100,
                'PH_ALPHA': 0.05}):
        with patch('statistical_analysis.mem_manager') as mock_mem:
            mock_mem.get_safe_analysis_rows.return_value = 100  # 강제로 total > max_rows
            with _pytest.raises(ValueError, match="SAMPLING_SEED"):
                analyzer._load_data()


def test_run_psm_pooled_sd_zero_returns_skipped_dict():
    """M-1: pooled_sd = 0 이면 run_psm이 skipped dict를 반환하고 self.results에 저장한다."""
    import numpy as np
    from statistical_analysis import StatisticalAnalyzer, SamplingInfo

    n = 4
    df = pd.DataFrame({
        'exposure_group': ['T1DM', 'T1DM', 'T2DM_OHA', 'T2DM_OHA'],
        'is_t1dm':         [1, 1, 0, 0],
        'is_t2dm_oha':     [0, 0, 1, 1],
        'is_t2dm_insulin': [0, 0, 0, 0],
        'is_t2dm_nomed':   [0, 0, 0, 0],
        'age_at_index':    [50.0, 51.0, 55.0, 56.0],
        'male':            [1, 1, 1, 1],
        'income_q':        [5, 5, 5, 5],
        'comor_hypertension':   [0, 0, 0, 0],
        'comor_dyslipidemia':   [0, 0, 0, 0],
        'dm_duration_years':    [3.0, 3.0, 3.0, 3.0],
        'follow_up_years':      [1.0, 1.0, 1.0, 1.0],
        'dementia_event':       [1, 0, 0, 0],
        'ad_event':             [0, 0, 0, 0],
        'vad_event':            [0, 0, 0, 0],
    })
    analyzer = StatisticalAnalyzer.__new__(StatisticalAnalyzer)
    analyzer._cached_df = df
    analyzer._sampling_info = SamplingInfo(applied=False, total_rows=n, sampled_rows=n)
    analyzer.results = {}
    analyzer.db_path = ':memory:'

    mock_lr = MagicMock()
    mock_lr.fit = MagicMock()
    # 같은 PS 값 → var()=0 → pooled_sd=0
    mock_lr.predict_proba = MagicMock(
        return_value=np.array([[0.1, 0.9], [0.1, 0.9], [0.9, 0.1], [0.9, 0.1]])
    )

    with patch('statistical_analysis.STUDY_SETTINGS',
               {'MIN_VALID_ROWS': 1, 'MIN_EVENTS': 1, 'SAMPLING_SEED': 42,
                'PSM_RATIO': 1, 'PSM_CALIPER': 0.2, 'PSM_SMD_THRESHOLD': 0.1,
                'PH_ALPHA': 0.05}):
        with patch('gpu_accelerator.is_gpu_enabled', return_value=False), \
             patch('gpu_accelerator.get_logistic_regression', return_value=mock_lr):
            result = analyzer.run_psm(df_prepared=df)

    assert result is not None, "run_psm 이 None 대신 skipped dict를 반환해야 한다"
    assert result.get('skipped') is True, f"skipped=True 기대, 실제: {result}"
    assert 'reason' in result, "reason 키 필요"
    assert analyzer.results.get('psm') == result, "self.results['psm']에 저장되어야 한다"


def test_run_interaction_saves_skipped_dict_when_no_duration_col():
    """M-2: dm_duration_cat 컬럼 없으면 skipped dict가 self.results['interaction']에 저장된다."""
    n = 5
    df = pd.DataFrame({
        'exposure_group': ['T1DM'] * n,
        'is_t1dm': [1] * n,
        # dm_duration_cat 컬럼 없음
        'age_at_index': [50.0] * n,
        'male': [1] * n,
        'follow_up_years': [1.0] * n,
        'dementia_event': [0] * n,
    })
    analyzer = _make_analyzer(df)
    with patch('statistical_analysis.STUDY_SETTINGS',
               {'MIN_VALID_ROWS': 1, 'MIN_EVENTS': 1, 'SAMPLING_SEED': 42}):
        result = analyzer.run_interaction(df_prepared=df)

    assert result is None
    stored = analyzer.results.get('interaction')
    assert stored is not None, "self.results['interaction']에 skipped dict 저장 필요"
    assert stored.get('skipped') is True
    assert 'reason' in stored


def test_run_interaction_saves_skipped_dict_when_insufficient_data():
    """M-2: 데이터 부족 시 skipped dict가 self.results['interaction']에 저장된다."""
    n = 5
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

    assert result is None
    stored = analyzer.results.get('interaction')
    assert stored is not None, "self.results['interaction']에 skipped dict 저장 필요"
    assert stored.get('skipped') is True
    assert 'reason' in stored


def test_load_settings_raises_on_invalid_sampling_seed(tmp_path):
    """M-3: load_settings 호출 시 SAMPLING_SEED 범위 초과는 ValueError 발생."""
    import json
    import pytest as _pytest
    from config import load_settings, STUDY_SETTINGS

    settings_file = tmp_path / 'test_settings.json'
    data = {'STUDY_SETTINGS': {'SAMPLING_SEED': 150}}
    settings_file.write_text(json.dumps(data), encoding='utf-8')

    with _pytest.raises(ValueError, match="SAMPLING_SEED"):
        load_settings(path=str(settings_file))


def test_load_settings_accepts_valid_sampling_seed(tmp_path):
    """M-3: 유효한 SAMPLING_SEED(0-99)는 정상 로드된다."""
    import json
    import pytest as _pytest
    from config import load_settings, STUDY_SETTINGS

    settings_file = tmp_path / 'test_settings.json'
    data = {'STUDY_SETTINGS': {'SAMPLING_SEED': 50}}
    settings_file.write_text(json.dumps(data), encoding='utf-8')

    result = load_settings(path=str(settings_file))
    assert result is True
    assert STUDY_SETTINGS['SAMPLING_SEED'] == 50
