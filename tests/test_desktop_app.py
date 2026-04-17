"""desktop_app.py 단위 테스트.

Windows-specific 부분(CREATE_NO_WINDOW, 배치 스크립트)은 단위 테스트 불가.
헬스체크/로그 경로/포트 감지 등 플랫폼 독립 로직만 검증한다.
"""
import os
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest import mock

import pytest

# desktop_app.py 는 프로젝트 루트에 있다
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import desktop_app as da  # noqa: E402


def test_port_open_false_for_free_port():
    """점유되지 않은 포트는 False 반환."""
    # 임시 포트 얻기
    with socket.socket() as s:
        s.bind(("localhost", 0))
        free_port = s.getsockname()[1]
    # 바인딩 해제 후 즉시 체크
    assert da._port_open(free_port, timeout=0.1) is False


def test_port_open_true_for_listening_port():
    """LISTEN 중인 포트는 True 반환."""
    srv = socket.socket()
    srv.bind(("localhost", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    try:
        assert da._port_open(port, timeout=0.5) is True
    finally:
        srv.close()


def test_resolve_python_falls_back_to_sys_executable(tmp_path, monkeypatch):
    """venv 없으면 sys.executable 반환."""
    monkeypatch.setattr(da, "ROOT", tmp_path)
    assert da._resolve_python() == sys.executable
