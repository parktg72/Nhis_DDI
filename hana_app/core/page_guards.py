"""Streamlit 페이지 진입 가드 헬퍼.

각 페이지 상단에서 호출해 사전 조건을 검증하고,
조건 미충족 시 Streamlit 렌더링을 중단한다.
"""
from __future__ import annotations


def check_hana_validated(cfg: dict) -> bool:
    """HANA 설정이 검증 완료 상태인지 확인한다.

    Args:
        cfg: load_config()가 반환한 설정 딕셔너리.

    Returns:
        True — validated=True이고 호스트 불일치 없음.
        False — validated=False 또는 호스트 불일치.
    """
    from hana_app.core.config import is_hana
    if not is_hana(cfg):
        return True  # HANA 아닌 경우 검증 불필요
    if not cfg.get("validated"):
        return False
    validated_host = cfg.get("validated_host", "")
    current_host = cfg.get("connection", {}).get("host", "")
    if validated_host and validated_host != current_host:
        return False
    return True


def get_validation_error(cfg: dict) -> str | None:
    """검증 실패 이유를 반환한다. 문제 없으면 None.

    Args:
        cfg: load_config()가 반환한 설정 딕셔너리.

    Returns:
        오류 메시지 문자열 또는 None.
    """
    from hana_app.core.config import is_hana
    if not is_hana(cfg):
        return None
    if not cfg.get("validated"):
        return "HANA 테이블 검증이 완료되지 않았습니다."
    validated_host = cfg.get("validated_host", "")
    current_host = cfg.get("connection", {}).get("host", "")
    if validated_host and validated_host != current_host:
        return (
            f"검증된 호스트({validated_host})와 "
            f"현재 연결 호스트({current_host})가 다릅니다."
        )
    return None
