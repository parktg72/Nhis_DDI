"""
NHIS DDI Risk Classification System - Desktop App Launcher
pywebview: Streamlit server runs in background, displayed in native window
Fallback: opens browser if pywebview is not installed
"""
from __future__ import annotations

import atexit
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT     = Path(__file__).parent
APP_FILE = ROOT / "hana_app" / "app.py"
PORT     = 8502
URL      = f"http://localhost:{PORT}"
TITLE    = "NHIS DDI Risk Classification System"
HEALTH_URL = f"http://localhost:{PORT}/_stcore/health"


def _resolve_python() -> str:
    for venv_name in (".venv_hana", ".venv", "venv"):
        p = ROOT / venv_name / "Scripts" / "python.exe"
        if p.exists():
            return str(p)
    return sys.executable


PYTHON_BIN = _resolve_python()


def _port_open(port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection(("localhost", port), timeout=timeout):
            return True
    except OSError:
        return False


def _is_our_streamlit(timeout: float = 2.0) -> bool:
    """8502 포트 점유자가 진짜 우리 Streamlit 인지 /_stcore/health 로 확인.

    임의 프로세스가 8502 를 쓰고 있을 때 잘못 연결되는 것을 방지한다.
    Streamlit 은 정확히 'ok' 200 을 반환하므로 strip+equality 로 엄격 비교
    (부분 문자열 매칭은 'not ok'/'tokyo' 같은 오탐 가능).
    """
    try:
        with urllib.request.urlopen(HEALTH_URL, timeout=timeout) as r:
            return r.status == 200 and r.read().strip().lower() == b"ok"
    except (urllib.error.URLError, socket.timeout, ConnectionError, OSError):
        return False


def _wait_ready(port: int, timeout: int = 60) -> bool:
    for _ in range(timeout * 2):
        if _port_open(port):
            return True
        time.sleep(0.5)
    return False


def _start_streamlit() -> subprocess.Popen:
    flags = 0
    if sys.platform == "win32":
        flags = subprocess.CREATE_NO_WINDOW
    return subprocess.Popen(
        [
            PYTHON_BIN, "-m", "streamlit", "run", str(APP_FILE),
            f"--server.port={PORT}",
            "--server.headless=true",
            "--browser.gatherUsageStats=false",
            "--theme.base=light",
            "--theme.primaryColor=#1f77b4",
        ],
        creationflags=flags,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _run_webview(proc: subprocess.Popen | None) -> None:
    import webview

    def on_closed():
        if proc and proc.poll() is None:
            proc.terminate()

    window = webview.create_window(
        TITLE,
        URL,
        width=1440,
        height=900,
        min_size=(960, 640),
    )
    window.events.closed += on_closed
    webview.start()


def _run_browser(proc: subprocess.Popen | None) -> None:
    import webbrowser
    print("[WARNING] pywebview not installed. Opening browser instead.")
    print(f"[INFO]    Run:  pip install pywebview")
    webbrowser.open(URL)
    if proc:
        try:
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()


def main() -> None:
    if not APP_FILE.exists():
        print(f"[ERROR] App file not found: {APP_FILE}")
        sys.exit(1)

    already_running = _port_open(PORT)
    proc = None

    if not already_running:
        print("Starting Streamlit server...")
        proc = _start_streamlit()
        if not _wait_ready(PORT, timeout=60):
            print("[ERROR] Server failed to start within 60 seconds.")
            if proc:
                proc.terminate()
            sys.exit(1)
        print("Server ready.")

    try:
        _run_webview(proc)
    except ImportError:
        _run_browser(proc)
    finally:
        if proc and proc.poll() is None:
            proc.terminate()


if __name__ == "__main__":
    main()
