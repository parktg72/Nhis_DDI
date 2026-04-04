"""
tests/test_stage_g.py - Stage G 분석 견고성 강화 테스트
"""

import pytest
from utils import format_error_for_user, InsufficientDataError


def test_format_error_for_user_insufficient_data_error():
    """InsufficientDataError 가 사용자 친화적 메시지로 변환되어야 한다."""
    exc = InsufficientDataError(valid_rows=10, min_rows=30)
    msg = format_error_for_user(exc)
    assert "10" in msg or "30" in msg or "최소" in msg, \
        f"InsufficientDataError 전용 메시지 없음: {msg!r}"
    # ValueError 일반 분기("입력값 오류:")로 떨어지면 안 됨
    assert "입력값 오류" not in msg, \
        f"InsufficientDataError 가 일반 ValueError 로 처리됨: {msg!r}"
