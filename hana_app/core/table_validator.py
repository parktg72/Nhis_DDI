"""HANA 테이블·컬럼 매핑 검증 헬퍼.

Page 1 wizard에서 호출되며, Streamlit 없이도 단위 테스트 가능하도록
순수 Python 함수로 작성한다.
"""
from __future__ import annotations

from hana_app.core.db import _assert_safe_identifier


def check_column_mapping(
    actual_cols: list[str],
    expected_map: dict[str, str],
) -> dict[str, list[str]]:
    """논리명 → 실제 DB 컬럼명 매핑을 검증한다.

    Args:
        actual_cols: DB에서 조회한 실제 컬럼명 목록.
        expected_map: {논리명: 기대 DB 컬럼명} 딕셔너리.

    Returns:
        {"ok": [일치한 논리명 목록], "missing": [불일치 논리명 목록]}
    """
    actual_set = set(actual_cols)
    ok: list[str] = []
    missing: list[str] = []
    for logical_name, db_col in expected_map.items():
        if db_col in actual_set:
            ok.append(logical_name)
        else:
            missing.append(logical_name)
    return {"ok": ok, "missing": missing}


def validate_all_identifiers(column_map: dict[str, str]) -> None:
    """column_map의 모든 키·값이 HANA 안전 식별자인지 검증한다.

    _assert_safe_identifier()를 통과하지 못하는 항목이 있으면 ValueError 발생.
    저장 직전 일괄 호출해 SQL 인젝션 방어선으로 사용한다.

    Args:
        column_map: {논리명: DB 컬럼명} 딕셔너리.

    Raises:
        ValueError: 안전하지 않은 식별자가 포함된 경우.
    """
    for logical_name, db_col in column_map.items():
        _assert_safe_identifier(logical_name, "논리명")
        _assert_safe_identifier(db_col, "DB 컬럼명")
