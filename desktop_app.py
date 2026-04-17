"""
NHIS DDI Risk Classification System - Desktop App Launcher
pywebview: Streamlit server runs in background, displayed in native window
pywebview 미설치 시 명확한 에러로 종료 (exit 2). 브라우저 fallback 없음 —
인트라넷 브라우저 3시간 자동 종료로부터 세션을 보호하기 위한 설계.
로그: %LOCALAPPDATA%\\hana_desktop\\logs\\desktop.log
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


# 모듈 레벨 로그 핸들 (ImportError 분기에서도 안전하게 참조되도록 최상위에 둔다)
LOG_DIR = Path(os.environ.get("LOCALAPPDATA", str(ROOT))) / "hana_desktop" / "logs"
try:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE: Path | None = LOG_DIR / "desktop.log"
    log_fh = LOG_FILE.open("a", encoding="utf-8", errors="replace")
    atexit.register(log_fh.close)
except OSError as e:
    print(f"[WARN] 로그 파일 생성 실패({e}) — stderr 로 대체", file=sys.stderr)
    LOG_FILE = None
    log_fh = sys.stderr


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


def _wait_ready(port: int, timeout: int = 90, proc: subprocess.Popen | None = None) -> bool:
    """포트가 열릴 때까지 0.5초 간격 폴링.

    proc 이 전달되면 서브프로세스 조기 종료(poll() != None) 감지 시 즉시
    False 반환 — 크래시 시 90초 허송 방지.
    """
    for _ in range(timeout * 2):
        if proc is not None and proc.poll() is not None:
            return False
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
            "--server.address=localhost",
            "--server.headless=true",
            "--browser.gatherUsageStats=false",
            "--theme.base=light",
            "--theme.primaryColor=#1f77b4",
        ],
        creationflags=flags,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
    )


def _run_webview(proc: subprocess.Popen | None) -> None:
    import webview

    def on_closed():
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

    window = webview.create_window(
        TITLE,
        URL,
        width=1440,
        height=900,
        min_size=(960, 640),
    )
    window.events.closed += on_closed
    webview.start()


def main() -> None:
    if not APP_FILE.exists():
        print(f"[ERROR] App file not found: {APP_FILE}", file=sys.stderr)
        sys.exit(1)

    # 포트 점유 시 내 Streamlit 인지 확인
    already_running = False
    if _port_open(PORT):
        if not _is_our_streamlit():
            print(f"[ERROR] 포트 {PORT}가 Streamlit 이외의 프로세스에 점유됨.", file=sys.stderr)
            print(f"[INFO]  종료 후 재시도하거나 PORT 충돌 원인을 확인하세요.", file=sys.stderr)
            sys.exit(3)
        already_running = True
        print(f"[INFO] 기존 Streamlit 재사용 (localhost:{PORT})")

    proc = None
    if not already_running:
        print("Starting Streamlit server...")
        proc = _start_streamlit()
        if not _wait_ready(PORT, timeout=90, proc=proc):
            if proc and proc.poll() is not None:
                print(f"[ERROR] Streamlit 프로세스가 조기 종료됨 (exit={proc.returncode}).", file=sys.stderr)
            else:
                print("[ERROR] Server failed to start within 90 seconds.", file=sys.stderr)
            if LOG_FILE:
                print("[INFO]  로그 마지막 20줄:", file=sys.stderr)
                try:
                    with LOG_FILE.open("r", encoding="utf-8", errors="replace") as f:
                        from collections import deque
                        tail = deque(f, maxlen=20)
                    for line in tail:
                        print(f"  {line.rstrip()}", file=sys.stderr)
                except OSError:
                    pass
            if proc and proc.poll() is None:
                proc.terminate()
            sys.exit(1)
        print("Server ready.")

    try:
        _run_webview(proc)
    except ImportError:
        print("[ERROR] pywebview 미설치.", file=sys.stderr)
        print("[INFO]  install_312.bat venv 를 다시 실행하거나 install_pywebview.bat 를 실행하세요.", file=sys.stderr)
        if LOG_FILE:
            print(f"[INFO]  로그: {LOG_FILE}", file=sys.stderr)
        if proc and proc.poll() is None:
            proc.terminate()
        sys.exit(2)
    finally:
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    main()
