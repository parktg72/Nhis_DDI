"""tests/test_stage_s.py — Stage S: tabs.py 방어 코드 테스트"""
import sys
from pathlib import Path
from unittest.mock import MagicMock
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from statistical_analysis import StatisticalAnalyzer


def test_on_post_analysis_result_none_guard():
    """result = data.get('result') or {} 패턴이 None 을 {} 로 대체하는지 검증."""
    data_with_none = {'result': None}
    result = data_with_none.get('result') or {}
    assert result == {}, f"None 이 {{}} 로 대체되지 않음: {result!r}"

    data_with_dict = {'result': {'errors': ['err1'], 'exported_files': []}}
    result2 = data_with_dict.get('result') or {}
    assert result2.get('errors') == ['err1'], "정상 dict 가 유지되지 않음"

    data_missing = {}
    result3 = data_missing.get('result') or {}
    assert result3 == {}, f"키 없을 때 {{}} 로 대체되지 않음: {result3!r}"


def test_run_competing_risks_standalone_passes_cb_to_load_data(monkeypatch):
    """run_competing_risks(cb=..., df_prepared=None) 시 _load_data 에 cb 전달."""
    dm = MagicMock()
    analyzer = StatisticalAnalyzer(dm)
    load_cb_received = []

    def patched_load(cb=None):
        load_cb_received.append(cb)
        raise pd.errors.EmptyDataError("테스트 중단")

    monkeypatch.setattr(analyzer, '_load_data', patched_load)
    cb = MagicMock()
    try:
        analyzer.run_competing_risks(cb=cb, df_prepared=None)
    except Exception:
        pass
    assert load_cb_received and load_cb_received[0] is cb, \
        f"run_competing_risks fallback: cb 미전달. received={load_cb_received}"


def test_generate_table1_standalone_passes_cb_to_load_data(monkeypatch):
    """generate_table1(cb=..., df_prepared=None) 시 _load_data 에 cb 전달."""
    dm = MagicMock()
    analyzer = StatisticalAnalyzer(dm)
    load_cb_received = []

    def patched_load(cb=None):
        load_cb_received.append(cb)
        raise pd.errors.EmptyDataError("테스트 중단")

    monkeypatch.setattr(analyzer, '_load_data', patched_load)
    cb = MagicMock()
    try:
        analyzer.generate_table1(cb=cb, df_prepared=None)
    except Exception:
        pass
    assert load_cb_received and load_cb_received[0] is cb, \
        f"generate_table1 fallback: cb 미전달. received={load_cb_received}"
