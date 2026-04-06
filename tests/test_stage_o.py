"""Stage O: TEMP_DIRECTORY fallback, 설정 파일 권한, _check_min_rows 위치 테스트"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import patch, MagicMock


def test_temp_directory_none_uses_base_dir(tmp_path):
    """TEMP_DIRECTORY=None 이면 _BASE_DIR 기준 경로가 사용된다."""
    import db_connector as dc

    storage = dc.DuckDBStorage(str(tmp_path / 'test.duckdb'))

    fake_settings = {
        'TEMP_DIRECTORY': None,
        'MEMORY_LIMIT': '1GB',
        'THREADS': 1,
    }
    with patch('db_connector.DUCKDB_SETTINGS', fake_settings):
        with patch('db_connector.os.makedirs') as mock_makedirs:
            with patch('db_connector.duckdb.connect') as mock_conn:
                mock_conn.return_value.execute = MagicMock()
                try:
                    storage.connect()
                except Exception:
                    pass
                called_path = mock_makedirs.call_args[0][0]
                assert str(dc._BASE_DIR) in called_path, \
                    f"TEMP_DIRECTORY=None 인데 _BASE_DIR 경로를 사용하지 않음: {called_path}"


def test_temp_directory_explicit_path_is_respected(tmp_path):
    """TEMP_DIRECTORY 가 명시된 경우 그 경로를 그대로 사용한다."""
    import db_connector as dc

    storage = dc.DuckDBStorage(str(tmp_path / 'test.duckdb'))
    explicit_path = str(tmp_path / 'custom_temp')

    fake_settings = {
        'TEMP_DIRECTORY': explicit_path,
        'MEMORY_LIMIT': '1GB',
        'THREADS': 1,
    }
    with patch('db_connector.DUCKDB_SETTINGS', fake_settings):
        with patch('db_connector.os.makedirs') as mock_makedirs:
            with patch('db_connector.duckdb.connect') as mock_conn:
                mock_conn.return_value.execute = MagicMock()
                try:
                    storage.connect()
                except Exception:
                    pass
                called_path = mock_makedirs.call_args[0][0]
                assert called_path == explicit_path, \
                    f"명시 경로가 무시됨: {called_path} != {explicit_path}"
