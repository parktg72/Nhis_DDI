"""
tests/test_stage_h.py - Stage H 가드 적용 범위 확대 테스트
"""

import pytest
import pandas as pd
from unittest.mock import patch
from utils import format_error_for_user, InsufficientDataError


def test_insufficient_data_error_default_kind_is_rows():
    """kind 미지정 시 기본값 'rows' 여야 한다 (하위호환)."""
    exc = InsufficientDataError(valid_rows=5, min_rows=30)
    assert exc.kind == "rows"


def test_insufficient_data_error_kind_events():
    """kind='events' 로 생성 가능해야 한다."""
    exc = InsufficientDataError(valid_rows=3, min_rows=10, kind="events")
    assert exc.kind == "events"


def test_format_error_rows_kind_mentions_min_valid_rows():
    """rows 종류 에러는 MIN_VALID_ROWS 설정을 안내해야 한다."""
    exc = InsufficientDataError(valid_rows=5, min_rows=30, kind="rows")
    msg = format_error_for_user(exc)
    assert "MIN_VALID_ROWS" in msg, f"MIN_VALID_ROWS 언급 없음: {msg!r}"
    assert "MIN_EVENTS" not in msg, f"잘못된 설정 키 MIN_EVENTS 언급: {msg!r}"


def test_format_error_events_kind_mentions_min_events():
    """events 종류 에러는 MIN_EVENTS 설정을 안내해야 한다."""
    exc = InsufficientDataError(valid_rows=3, min_rows=10, kind="events")
    msg = format_error_for_user(exc)
    assert "MIN_EVENTS" in msg, f"MIN_EVENTS 언급 없음: {msg!r}"
    assert "MIN_VALID_ROWS" not in msg, f"잘못된 설정 키 MIN_VALID_ROWS 언급: {msg!r}"
