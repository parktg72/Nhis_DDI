"""db_connector.py 청크 적재 시 넓은 Decimal 스키마 고정 테스트."""

import sys
from decimal import Decimal
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db_connector import (
    DUCKDB_WIDE_INTEGER_DECIMAL,
    DuckDBStorage,
    _build_chunk_select_sql,
    _prepare_chunk_for_duckdb,
)


def test_prepare_chunk_for_duckdb_marks_integral_decimal_for_wide_decimal_cast():
    df = pd.DataFrame({
        'claim_id': [Decimal('999999'), Decimal('1031900'), None],
    })

    converted = _prepare_chunk_for_duckdb(df.copy())

    assert str(converted['claim_id'].dtype) == 'object'
    assert converted['claim_id'].iloc[0] == '999999'
    assert converted['claim_id'].iloc[1] == '1031900'
    assert pd.isna(converted['claim_id'].iloc[2])
    assert converted.attrs['duckdb_type_overrides'] == {
        'claim_id': DUCKDB_WIDE_INTEGER_DECIMAL,
    }


def test_chunk_insert_does_not_freeze_decimal_6_0_schema(tmp_path):
    storage = DuckDBStorage(str(tmp_path / 'decimal_chunks.duckdb'))
    storage.connect()

    first_chunk = _prepare_chunk_for_duckdb(pd.DataFrame({
        'claim_id': [Decimal('999999')],
    }))
    storage.conn.register('_temp_chunk', first_chunk)
    storage.execute(f"CREATE TABLE claims AS {_build_chunk_select_sql(first_chunk, '_temp_chunk')}")
    storage.conn.unregister('_temp_chunk')

    second_chunk = _prepare_chunk_for_duckdb(pd.DataFrame({
        'claim_id': [Decimal('1031900')],
    }))
    storage.conn.register('_temp_chunk', second_chunk)
    storage.execute(f"INSERT INTO claims {_build_chunk_select_sql(second_chunk, '_temp_chunk')}")
    storage.conn.unregister('_temp_chunk')

    col_type = storage.execute("""
        SELECT data_type
        FROM information_schema.columns
        WHERE table_name = 'claims' AND column_name = 'claim_id'
    """).fetchone()[0]
    values = storage.execute("SELECT claim_id FROM claims ORDER BY claim_id").fetchall()

    assert col_type == DUCKDB_WIDE_INTEGER_DECIMAL
    assert values == [(Decimal('999999'),), (Decimal('1031900'),)]

    storage.close()
