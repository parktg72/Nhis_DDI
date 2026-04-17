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


class _HealthHandler(BaseHTTPRequestHandler):
    """/_stcore/health 에 'ok' 응답하는 mock 서버."""
    response_body = b"ok"
    response_code = 200

    def do_GET(self):
        if self.path == "/_stcore/health":
            self.send_response(self.response_code)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(self.response_body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *a, **kw):
        pass  # 테스트 로그 오염 방지


@pytest.fixture
def mock_health_server():
    """임시 HTTP 서버 기동 후 port 반환."""
    def _start(body: bytes = b"ok", code: int = 200):
        _HealthHandler.response_body = body
        _HealthHandler.response_code = code
        httpd = HTTPServer(("localhost", 0), _HealthHandler)
        port = httpd.server_address[1]
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        return httpd, port

    servers = []
    def factory(body: bytes = b"ok", code: int = 200):
        httpd, port = _start(body, code)
        servers.append(httpd)
        return port

    yield factory

    for httpd in servers:
        httpd.shutdown()


def test_is_our_streamlit_ok_response(mock_health_server, monkeypatch):
    """/_stcore/health 가 'ok' 응답 → True."""
    port = mock_health_server(body=b"ok", code=200)
    monkeypatch.setattr(da, "PORT", port)
    monkeypatch.setattr(da, "HEALTH_URL", f"http://localhost:{port}/_stcore/health")
    assert da._is_our_streamlit(timeout=1.0) is True


def test_is_our_streamlit_wrong_response(mock_health_server, monkeypatch):
    """/_stcore/health 가 'ok' 가 아닌 응답 → False (다른 프로세스)."""
    port = mock_health_server(body=b"hello from nginx", code=200)
    monkeypatch.setattr(da, "PORT", port)
    monkeypatch.setattr(da, "HEALTH_URL", f"http://localhost:{port}/_stcore/health")
    assert da._is_our_streamlit(timeout=1.0) is False


def test_is_our_streamlit_partial_match_rejected(mock_health_server, monkeypatch):
    """'not ok' 같은 부분 문자열 응답은 False (엄격 비교 회귀 방지)."""
    port = mock_health_server(body=b"not ok", code=200)
    monkeypatch.setattr(da, "PORT", port)
    monkeypatch.setattr(da, "HEALTH_URL", f"http://localhost:{port}/_stcore/health")
    assert da._is_our_streamlit(timeout=1.0) is False


def test_is_our_streamlit_no_response(monkeypatch):
    """아무도 듣지 않는 포트 → False."""
    # 예약 후 해제하여 "비어있음이 보장된 포트" 확보
    with socket.socket() as s:
        s.bind(("localhost", 0))
        port = s.getsockname()[1]
    monkeypatch.setattr(da, "PORT", port)
    monkeypatch.setattr(da, "HEALTH_URL", f"http://localhost:{port}/_stcore/health")
    assert da._is_our_streamlit(timeout=0.5) is False


def test_log_dir_uses_localappdata(tmp_path, monkeypatch):
    """LOCALAPPDATA 환경변수가 있으면 그 아래에 hana_desktop/logs 배치."""
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    # 모듈 재로드하여 최상단 LOG_DIR 재계산
    import importlib
    importlib.reload(da)
    assert da.LOG_DIR == tmp_path / "hana_desktop" / "logs"


def test_log_dir_falls_back_to_root(tmp_path, monkeypatch):
    """LOCALAPPDATA 가 없으면 ROOT 기준으로 경로 계산 (에러 나지 않음)."""
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.setattr(da, "ROOT", tmp_path)
    import importlib
    importlib.reload(da)
    assert da.LOG_DIR.parent.name == "hana_desktop"


def test_log_file_open_failure_falls_back_to_stderr(monkeypatch, capsys):
    """로그 디렉터리 생성 실패 시 LOG_FILE=None, log_fh=stderr, 경고 출력."""
    def bad_mkdir(*args, **kwargs):
        raise OSError("permission denied (test)")
    monkeypatch.setattr(Path, "mkdir", bad_mkdir)
    import importlib
    importlib.reload(da)
    assert da.LOG_FILE is None
    assert da.log_fh is sys.stderr
    captured = capsys.readouterr()
    assert "로그 파일 생성 실패" in captured.err
