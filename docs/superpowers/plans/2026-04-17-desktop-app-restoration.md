# 데스크탑 앱(pywebview) 복구 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 폐쇄망 Windows에서 인트라넷 브라우저 3시간 자동 종료로 끊기던 Streamlit 세션을 pywebview 데스크탑 창으로 복구하고 `run_desktop.bat` 더블클릭만으로 동작하게 만든다.

**Architecture:** 기존 `desktop_app.py`(pywebview + Streamlit subprocess)를 복구. 포트 분리(웹 8501 / 데스크탑 8502) + `_stcore/health` 헬스체크 + 모듈 레벨 로그 상수 + 사일런트 fallback 제거. `install_312.bat` 에 pywebview 오프라인 설치 통합.

**Tech Stack:** Python 3.12 · Streamlit · pywebview 6.1 · urllib.request · pytest · Windows 배치 스크립트.

**Spec 참조:** `docs/superpowers/specs/2026-04-17-desktop-app-design.md`

---

## File Structure

- **CREATE** `tests/test_desktop_app.py` — pytest 단위 테스트
- **MODIFY** `desktop_app.py` — 포트 변경, 헬스체크, 로그 리다이렉트, 에러 처리
- **MODIFY** `run_desktop.bat` — UTF-8, 사전 점검, 조건부 pause
- **MODIFY** `install_312.bat` — pywebview 설치 단계 통합
- **MODIFY** `install_pywebview.bat` — `install_312.bat`과 집합 일치 + 선택사항 주석
- **MODIFY** `web-user-guide.md` — 데스크탑 모드 사용법 섹션
- **CREATE/MERGE** `.claude/settings.local.json` — auto-approve 규칙 (기존 있으면 deep-merge)

각 파일의 책임은 독립적으로 변경/테스트 가능하며, 커밋도 분리한다.

---

## Task 1: proxy_tools 실제 의존성 확인 (선조사)

**Files:** 없음 (조사만)

- [ ] **Step 1: Mac 개발 환경에서 pywebview 6.1 의 의존성 조회**

Run:
```bash
cd /Users/aidept/ptg_at_train/MODE_11_hana
.venv_macos/bin/python -m pip show pywebview 2>/dev/null | grep -i requires
# 또는 pip 미설치 시
python3 -m pip install --dry-run pywebview==6.1 2>&1 | grep -i proxy
```
Expected: `proxy_tools` 가 Requires 또는 종속 설치 목록에 나타나면 유지, 없으면 `install_312.bat` / `install_pywebview.bat` 에서 제외.

- [ ] **Step 2: 결정 기록**

파일 `docs/superpowers/plans/2026-04-17-desktop-app-restoration.md` 의 Task 7/8 의 설치 커맨드를 Step 1 결과에 따라:
- **필요하면**: `pywebview proxy_tools` 유지
- **불필요하면**: `pywebview` 단독으로 Task 7/8 수정 후 진행

결정 메모를 커밋 메시지에 남길 것: `chore: proxy_tools 의존성 확인 결과 반영`

- [ ] **Step 3: 이 Task는 커밋 생략** (조사만; 수정 발생 시 Task 7/8 커밋에 포함)

---

## Task 2: 테스트 파일 스캐폴딩 + 기존 함수 baseline 테스트

**Files:**
- Create: `tests/test_desktop_app.py`

- [ ] **Step 1: 테스트 파일 생성 (실패하도록 먼저 기록)**

File: `tests/test_desktop_app.py`

```python
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
```

- [ ] **Step 2: 테스트 실행하여 기존 함수 동작 확인**

Run:
```bash
cd /Users/aidept/ptg_at_train/MODE_11_hana
.venv_macos/bin/python -m pytest tests/test_desktop_app.py -v
```
Expected: 3 passed (기존 desktop_app.py 의 `_port_open`, `_resolve_python` 은 이미 존재).

- [ ] **Step 3: Commit**

```bash
git add tests/test_desktop_app.py
git commit -m "test: desktop_app.py baseline — _port_open/_resolve_python 회귀 방지"
```

---

## Task 3: `_is_our_streamlit()` TDD — 헬스체크 신규 함수

**Files:**
- Modify: `tests/test_desktop_app.py` (테스트 추가)
- Modify: `desktop_app.py` (함수 신규)

- [ ] **Step 1: 헬스체크 테스트 먼저 작성 (실패 기대)**

File: `tests/test_desktop_app.py` — 파일 하단에 추가

```python
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
    monkeypatch.setattr(da, "HEALTH_URL", f"http://localhost:{port}/_stcore/health")
    assert da._is_our_streamlit(timeout=1.0) is False


def test_is_our_streamlit_no_response(monkeypatch):
    """아무도 듣지 않는 포트 → False."""
    # 예약 후 해제하여 "비어있음이 보장된 포트" 확보
    with socket.socket() as s:
        s.bind(("localhost", 0))
        port = s.getsockname()[1]
    monkeypatch.setattr(da, "HEALTH_URL", f"http://localhost:{port}/_stcore/health")
    assert da._is_our_streamlit(timeout=0.5) is False
```

- [ ] **Step 2: 테스트 실행 → 실패 확인**

Run:
```bash
.venv_macos/bin/python -m pytest tests/test_desktop_app.py::test_is_our_streamlit_ok_response -v
```
Expected: FAIL — `AttributeError: module 'desktop_app' has no attribute '_is_our_streamlit'` 또는 `HEALTH_URL`.

- [ ] **Step 3: `desktop_app.py` 에 헬스체크 함수 추가**

File: `desktop_app.py` — 기존 `_port_open` 함수 아래, `_wait_ready` 위에 추가.

먼저 import 섹션 보강:

```python
from __future__ import annotations

import atexit
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
```

그리고 `PORT = 8501` 줄을 `PORT = 8502` 로 변경하고 바로 아래에:

```python
HEALTH_URL = f"http://localhost:{PORT}/_stcore/health"


def _is_our_streamlit(timeout: float = 2.0) -> bool:
    """8502 포트 점유자가 진짜 우리 Streamlit 인지 /_stcore/health 로 확인.

    임의 프로세스가 8502 를 쓰고 있을 때 잘못 연결되는 것을 방지한다.
    """
    try:
        with urllib.request.urlopen(HEALTH_URL, timeout=timeout) as r:
            return r.status == 200 and b"ok" in r.read().lower()
    except Exception:
        return False
```

- [ ] **Step 4: 테스트 실행 → 통과 확인**

Run:
```bash
.venv_macos/bin/python -m pytest tests/test_desktop_app.py -v
```
Expected: 6 passed (기존 3 + 신규 3).

- [ ] **Step 5: Commit**

```bash
git add tests/test_desktop_app.py desktop_app.py
git commit -m "feat: _is_our_streamlit() 헬스체크 — 8502 포트 오연결 방지"
```

---

## Task 4: 로그 파일 모듈 레벨 상수 + stderr fallback

**Files:**
- Modify: `tests/test_desktop_app.py`
- Modify: `desktop_app.py`

- [ ] **Step 1: 로그 경로 계산 테스트 작성 (실패 기대)**

File: `tests/test_desktop_app.py` — 파일 하단에 추가

```python
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
```

- [ ] **Step 2: 테스트 실행 → 실패 확인**

Run:
```bash
.venv_macos/bin/python -m pytest tests/test_desktop_app.py::test_log_dir_uses_localappdata -v
```
Expected: FAIL — `AttributeError: module 'desktop_app' has no attribute 'LOG_DIR'`.

- [ ] **Step 3: 모듈 레벨 로그 상수를 `desktop_app.py` 에 추가**

File: `desktop_app.py` — 기존 상수 블록(`ROOT`, `APP_FILE`, `PORT`...) 바로 아래에 추가. `_resolve_python()` 함수 **정의 후** PYTHON_BIN 상수 정의 **전**에 둔다 (의존 없음).

```python
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
```

- [ ] **Step 4: 테스트 실행 → 통과 확인**

Run:
```bash
.venv_macos/bin/python -m pytest tests/test_desktop_app.py -v
```
Expected: 9 passed (기존 6 + 신규 3).

- [ ] **Step 5: Commit**

```bash
git add tests/test_desktop_app.py desktop_app.py
git commit -m "feat: 로그 파일 모듈 레벨 초기화 + stderr fallback + atexit close"
```

---

## Task 5: `desktop_app.py` 나머지 수정 — 포트/헬스체크 main 통합/로그 리다이렉트/ImportError 에러/타임아웃 90초

**Files:**
- Modify: `desktop_app.py`

- [ ] **Step 1: `_start_streamlit()` — 로그 리다이렉트 + server.address**

Replace 기존 `_start_streamlit` 함수 전체로:

```python
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
```

- [ ] **Step 2: `_wait_ready` 타임아웃 60 → 90초**

Replace `_wait_ready` 정의 첫 줄:
```python
def _wait_ready(port: int, timeout: int = 90) -> bool:
```

- [ ] **Step 3: `_run_webview()` — on_closed 견고화**

Replace `_run_webview` 내부의 `on_closed` 정의:

```python
def on_closed():
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
```

- [ ] **Step 4: `_run_browser()` 함수 완전 제거**

File: `desktop_app.py` — 함수 `_run_browser` 블록 전체 삭제.

- [ ] **Step 5: `main()` 전면 교체 — 헬스체크 + ImportError 에러 exit**

Replace 기존 `main()` 함수 전체:

```python
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
        if not _wait_ready(PORT, timeout=90):
            print(f"[ERROR] Server failed to start within 90 seconds.", file=sys.stderr)
            if LOG_FILE:
                print(f"[INFO]  로그 마지막 20줄:", file=sys.stderr)
                try:
                    tail = LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()[-20:]
                    for line in tail:
                        print(f"  {line}", file=sys.stderr)
                except OSError:
                    pass
            if proc:
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
```

- [ ] **Step 6: 기존 테스트 유지 + 통과 확인**

Run:
```bash
.venv_macos/bin/python -m pytest tests/test_desktop_app.py -v
```
Expected: 9 passed (기능 변경은 `main()` 내부 통합일 뿐, 기존 테스트는 그대로 유효).

- [ ] **Step 7: `desktop_app.py` 구문 점검**

Run:
```bash
.venv_macos/bin/python -c "import ast; ast.parse(open('desktop_app.py').read()); print('AST OK')"
```
Expected: `AST OK`

- [ ] **Step 8: Commit**

```bash
git add desktop_app.py
git commit -m "feat: desktop_app.py 핵심 수정 — 포트 8502 + 헬스체크 통합 + 로그 리다이렉트 + ImportError 에러로 fallback 제거"
```

---

## Task 6: `run_desktop.bat` 수정 — UTF-8 + 사전점검 + 조건부 pause

**Files:**
- Modify: `run_desktop.bat`

- [ ] **Step 1: 파일 전체 교체**

File: `run_desktop.bat` — 전체 내용을 아래로 대체.

```bat
@echo off
chcp 65001 >nul
setlocal

set ROOT=%~dp0
set PYTHON_BIN=

if exist "%ROOT%.venv_hana\Scripts\python.exe" set PYTHON_BIN=%ROOT%.venv_hana\Scripts\python.exe
if not defined PYTHON_BIN if exist "%ROOT%.venv\Scripts\python.exe" set PYTHON_BIN=%ROOT%.venv\Scripts\python.exe
if not defined PYTHON_BIN if exist "%ROOT%venv\Scripts\python.exe"  set PYTHON_BIN=%ROOT%venv\Scripts\python.exe

if not defined PYTHON_BIN (
    for /f "tokens=*" %%i in ('where python 2^>nul') do (
        if not defined PYTHON_BIN set PYTHON_BIN=%%i
    )
)

if not defined PYTHON_BIN (
    echo [ERROR] Python not found. Run install_312.bat venv first.
    pause
    exit /b 1
)

REM 사전점검: streamlit
"%PYTHON_BIN%" -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] streamlit not installed. Run install_312.bat venv first.
    pause
    exit /b 1
)

REM 사전점검: pywebview
"%PYTHON_BIN%" -c "import webview" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] pywebview not installed. Run install_312.bat venv first (or install_pywebview.bat).
    pause
    exit /b 1
)

echo Python : %PYTHON_BIN%
echo Script : %ROOT%desktop_app.py
echo.
echo Starting app...

"%PYTHON_BIN%" "%ROOT%desktop_app.py"
if errorlevel 1 (
    echo.
    echo [FAILED] desktop_app.py exited with error. See above for details.
    pause
)

endlocal
```

- [ ] **Step 2: 구문 확인 (Mac 에서는 실행 불가하나 grep 으로 확인 가능)**

Run:
```bash
grep -n "chcp 65001" /Users/aidept/ptg_at_train/MODE_11_hana/run_desktop.bat
grep -n "if errorlevel 1" /Users/aidept/ptg_at_train/MODE_11_hana/run_desktop.bat
```
Expected: `chcp 65001` 1회, `if errorlevel 1` 최소 4회(Python 없음/streamlit 없음/webview 없음/app 실패).

- [ ] **Step 3: Commit**

```bash
git add run_desktop.bat
git commit -m "feat: run_desktop.bat — UTF-8 chcp + 사전점검 3단계 + 조건부 pause"
```

---

## Task 7: `install_312.bat` 수정 — pywebview 설치 단계 통합

**Files:**
- Modify: `install_312.bat`

- [ ] **Step 1: 4단계(Streamlit 설치) 직후에 pywebview 설치 블록 삽입**

File: `install_312.bat` — 아래 문자열을 찾아 그 뒤에 블록 삽입.

찾기 (find):
```
%PYTHON_BIN% -m pip install --no-index %FIND_LINKS% ^
    streamlit altair watchdog matplotlib statsmodels duckdb
if errorlevel 1 (
    echo [오류] Streamlit 설치 실패
    echo        packages_win\py312 에 streamlit wheel 이 있는지 확인하세요.
    pause
    exit /b 1
)
```

바로 그 뒤에 삽입 (add):
```bat

REM ── 4.2단계: 데스크탑 앱 (pywebview) ─────────────────────────
echo.
echo [4.2/5] 데스크탑 앱 (pywebview) 설치...
%PYTHON_BIN% -m pip install --no-index %FIND_LINKS% pywebview proxy_tools
if errorlevel 1 echo [경고] pywebview 설치 실패 — run_desktop.bat 미지원 (hana_app\run.bat 은 정상)

```

**주의**: Task 1 에서 `proxy_tools` 가 불필요하다고 판정되면 `pywebview proxy_tools` → `pywebview` 로 변경.

- [ ] **Step 2: 5단계 검증 섹션에 pywebview OK 출력 추가**

찾기 (find):
```
echo [웹앱]
%PYTHON_BIN% -c "import streamlit; print('  Streamlit', streamlit.__version__, 'OK')" 2>nul || (echo   [실패] Streamlit & set FAIL=1)
%PYTHON_BIN% -c "import fastapi, uvicorn; print('  FastAPI/uvicorn OK')" 2>nul || (echo   [실패] FastAPI & set FAIL=1)
```

바로 뒤에 삽입:
```bat

echo [데스크탑]
%PYTHON_BIN% -c "import webview" 2>nul && echo   pywebview OK || (echo   [실패] pywebview & set FAIL=1)
if not exist "%ProgramFiles(x86)%\Microsoft\EdgeWebView\Application" echo   [경고] Edge WebView2 Runtime 미감지 (run_desktop.bat 실패 가능)

```

- [ ] **Step 3: 파일이 UTF-8 CRLF 로 유지되는지 확인**

Run:
```bash
file /Users/aidept/ptg_at_train/MODE_11_hana/install_312.bat
```
Expected: `UTF-8 Unicode text, with CRLF line terminators` 또는 유사.

- [ ] **Step 4: Commit**

```bash
git add install_312.bat
git commit -m "feat: install_312.bat — pywebview 오프라인 설치 4.2단계 통합 + 검증"
```

---

## Task 8: `install_pywebview.bat` 수정 — 집합 일치 + 선택사항 주석

**Files:**
- Modify: `install_pywebview.bat`

- [ ] **Step 1: 파일 전체 교체**

File: `install_pywebview.bat` — 기존 내용을 아래로 대체.

```bat
@echo off
REM ============================================================
REM pywebview 단독 오프라인 설치 (선택사항)
REM 표준 경로: install_312.bat venv (pywebview 포함)
REM 이 스크립트는 레거시/진단용이며 install_312.bat 과 집합을 일치시킨다.
REM ============================================================
setlocal

set ROOT=%~dp0
set PYTHON_BIN=

if exist "%ROOT%.venv_hana\Scripts\python.exe" set PYTHON_BIN=%ROOT%.venv_hana\Scripts\python.exe
if not defined PYTHON_BIN if exist "%ROOT%.venv\Scripts\python.exe" set PYTHON_BIN=%ROOT%.venv\Scripts\python.exe
if not defined PYTHON_BIN if exist "%ROOT%venv\Scripts\python.exe"  set PYTHON_BIN=%ROOT%venv\Scripts\python.exe
if not defined PYTHON_BIN set PYTHON_BIN=python

set PKG_DIR=%ROOT%packages_win\py312

echo Python  : %PYTHON_BIN%
echo Packages: %PKG_DIR%
echo.

if not exist "%PKG_DIR%\pywebview-6.1-py3-none-any.whl" (
    echo [ERROR] pywebview package not found in %PKG_DIR%
    echo Run download_pywebview.bat on an internet-connected machine first.
    pause
    exit /b 1
)

"%PYTHON_BIN%" -m pip install pywebview proxy_tools --no-index --find-links="%PKG_DIR%"

if errorlevel 1 (
    echo.
    echo [ERROR] Installation failed.
    pause
    exit /b 1
)

echo.
"%PYTHON_BIN%" -c "import webview; print('pywebview OK')"
echo.
echo Done. Run run_desktop.bat to launch the app.

endlocal
pause
```

**주의**: Task 1 결과에 따라 `pywebview proxy_tools` → `pywebview` 로 변경.

- [ ] **Step 2: Commit**

```bash
git add install_pywebview.bat
git commit -m "chore: install_pywebview.bat — install_312.bat 과 설치 집합 일치 + 선택사항 명시"
```

---

## Task 9: `.claude/settings.local.json` auto-approve 설정

**Files:**
- Create or Merge: `.claude/settings.local.json`

- [ ] **Step 1: 기존 파일 확인**

Run:
```bash
cat /Users/aidept/ptg_at_train/MODE_11_hana/.claude/settings.local.json 2>/dev/null || echo "NOT_EXIST"
```

- [ ] **Step 2A: 파일이 없으면 신규 작성**

File: `.claude/settings.local.json`

```json
{
  "permissions": {
    "allow": [
      "Bash(python -m pip install*)",
      "Bash(install_312.bat*)",
      "Bash(./install_312.bat*)",
      "Bash(install_pywebview.bat*)",
      "Bash(./install_pywebview.bat*)",
      "Bash(run_desktop.bat*)",
      "Bash(./run_desktop.bat*)",
      "Bash(./hana_app/run.bat*)",
      "Edit(install_312.bat)",
      "Edit(desktop_app.py)",
      "Edit(run_desktop.bat)",
      "Edit(install_pywebview.bat)",
      "Edit(docs/superpowers/**)"
    ],
    "deny": [
      "Bash(rm -rf*)",
      "Bash(git push --force*)",
      "Bash(rmdir /S*)"
    ]
  }
}
```

- [ ] **Step 2B: 파일이 이미 있으면 Python 한 줄로 deep-merge**

Run (기존 파일이 있는 경우에만):
```bash
cd /Users/aidept/ptg_at_train/MODE_11_hana
python3 - <<'PY'
import json, pathlib
p = pathlib.Path(".claude/settings.local.json")
current = json.loads(p.read_text())
additions = {
    "permissions": {
        "allow": [
            "Bash(python -m pip install*)",
            "Bash(install_312.bat*)",
            "Bash(./install_312.bat*)",
            "Bash(install_pywebview.bat*)",
            "Bash(./install_pywebview.bat*)",
            "Bash(run_desktop.bat*)",
            "Bash(./run_desktop.bat*)",
            "Bash(./hana_app/run.bat*)",
            "Edit(install_312.bat)",
            "Edit(desktop_app.py)",
            "Edit(run_desktop.bat)",
            "Edit(install_pywebview.bat)",
            "Edit(docs/superpowers/**)",
        ],
        "deny": ["Bash(rm -rf*)", "Bash(git push --force*)", "Bash(rmdir /S*)"],
    }
}
perms = current.setdefault("permissions", {})
for k in ("allow", "deny"):
    existing = perms.setdefault(k, [])
    for item in additions["permissions"][k]:
        if item not in existing:
            existing.append(item)
p.write_text(json.dumps(current, indent=2, ensure_ascii=False) + "\n")
print("MERGED")
PY
```

- [ ] **Step 3: JSON 유효성 확인**

Run:
```bash
python3 -c "import json; print(json.loads(open('/Users/aidept/ptg_at_train/MODE_11_hana/.claude/settings.local.json').read()))"
```
Expected: 딕셔너리 출력, 에러 없음.

- [ ] **Step 4: Commit**

```bash
git add .claude/settings.local.json
git commit -m "chore: .claude/settings.local.json — 반복 루틴 auto-approve 규칙"
```

---

## Task 10: `web-user-guide.md` — 데스크탑 모드 섹션 추가

**Files:**
- Modify: `web-user-guide.md`

- [ ] **Step 1: 기존 구조 확인하여 삽입 위치 결정**

Run:
```bash
grep -n "^##" /Users/aidept/ptg_at_train/MODE_11_hana/web-user-guide.md | head -20
```
삽입 대상 섹션: 파일 맨 끝(혹은 "실행" 관련 섹션 뒤)에 `## 데스크탑 모드` 추가.

- [ ] **Step 2: 파일 끝에 섹션 추가**

File: `web-user-guide.md` — 파일 끝에 아래 추가.

```markdown

## 데스크탑 모드 (run_desktop.bat)

### 언제 사용하나

회사 인트라넷 정책이 **3시간 미사용 시 브라우저를 자동 종료**하여 웹앱 분석 세션이 끊기는 경우. 데스크탑 모드는 pywebview 임베디드 창에서 동작하므로 브라우저 자동 종료 대상이 아니다.

### 실행

1. `run_desktop.bat` 더블클릭
2. 자동으로 `.venv_hana` 감지 → Streamlit 서버 8502 포트 기동 → pywebview 창 표시
3. 웹 모드(`hana_app\run.bat`, 포트 8501)와 **동시 실행 가능** (포트가 다르므로 충돌 없음)

### 종료

창 X 버튼 클릭 → Streamlit 서브프로세스 자동 종료.

### 로그 위치

`%LOCALAPPDATA%\hana_desktop\logs\desktop.log`

예: `C:\Users\<사용자>\AppData\Local\hana_desktop\logs\desktop.log`

### 8502 포트 점유 시

- **내 Streamlit 이 이미 떠 있는 경우**: 헬스체크 통과 → 기존 서버 재사용 (자동).
- **다른 프로세스가 점유**: 명확한 에러 후 exit 3. `netstat -ano | findstr :8502` 로 확인 후 해당 프로세스 종료.

### 설치 / 재설치

- 최초 설치 또는 Python 재설치 후: `install_312.bat venv` 한 번 실행(모든 의존성 일괄 설치).
- pywebview 만 별도: `install_pywebview.bat` (레거시/진단용).

### WebView2 Runtime

pywebview 는 Windows 에서 **Edge WebView2 Runtime** 을 필요로 한다. 사내 표준 이미지에 포함되어 있지 않다면 설치 담당자에게 문의.
```

- [ ] **Step 3: Commit**

```bash
git add web-user-guide.md
git commit -m "docs: web-user-guide — 데스크탑 모드(run_desktop.bat) 섹션 추가"
```

---

## Task 11: 전체 회귀 테스트 + 최종 점검

**Files:** 없음 (검증만)

- [ ] **Step 1: 전체 pytest 실행**

Run:
```bash
cd /Users/aidept/ptg_at_train/MODE_11_hana
.venv_macos/bin/python -m pytest tests/test_desktop_app.py -v
```
Expected: 9 passed.

- [ ] **Step 2: 관련 테스트 영향 없는지 스모크**

Run:
```bash
.venv_macos/bin/python -m pytest tests/ -x --ignore=tests/test_integration -q 2>&1 | tail -20
```
Expected: 기존 테스트 회귀 없음 (실패 시 원인 분석). 데스크탑 변경이 hana_app 내부를 건드리지 않았으므로 다른 테스트에 영향 없어야 함.

- [ ] **Step 3: 변경 파일 최종 목록 확인**

Run:
```bash
git log --name-status main.. 2>&1 | head -40
# 또는 아직 푸시 전이면 스펙 커밋 이후로
git log --name-status ca70c1c.. 2>&1
```
Expected 파일 목록:
- `desktop_app.py` (modified)
- `run_desktop.bat` (modified)
- `install_312.bat` (modified)
- `install_pywebview.bat` (modified)
- `.claude/settings.local.json` (created or modified)
- `tests/test_desktop_app.py` (created)
- `web-user-guide.md` (modified)

- [ ] **Step 4: `/advisor` 검증 호출**

CLAUDE.md 플로우의 마지막 단계: "구현 → /advisor verify → gstack commit". 이 시점에 advisor 에게 구현 결과 검증 요청.

- [ ] **Step 5: Windows 폐쇄망 수동 테스트 체크리스트 전달**

스펙 내 "수동 체크리스트" 항목을 사용자에게 전달:
- install_312.bat venv 재실행 후 `pywebview OK` 확인
- run_desktop.bat 더블클릭 → pywebview 창 표시
- run.bat(8501) 과 run_desktop.bat(8502) 동시 실행
- HANA 연결 + 분석 수행
- **3시간+ 방치 후 session_state 유지 (핵심 회귀)**
- 창 X → Python 프로세스 종료 확인
- `desktop.log` 로그 기록 확인

- [ ] **Step 6: Push 전 마지막 점검**

Run:
```bash
git status
git log --oneline ca70c1c..HEAD
```
예상 커밋 개수: 9~10개 (Task 2~10 각 1커밋, 파일 없음 Task 1/11 제외).

---

## 체크리스트 매핑 (Spec → Task)

| Spec 항목 | Task |
|---|---|
| A. install_312.bat 수정 | 7 |
| B. desktop_app.py PORT=8502 | 3 |
| B. desktop_app.py imports 추가 | 3, 4, 5 |
| B. LOG_FILE 모듈 레벨 + atexit | 4 |
| B. 헬스체크 `_is_our_streamlit` | 3 |
| B. 로그 리다이렉트 | 5 |
| B. on_closed 견고화 | 5 |
| B. `_run_browser` 제거 + ImportError 에러 | 5 |
| B. 90초 타임아웃 + 마지막 20줄 | 5 |
| B. `--server.address=localhost` | 5 |
| C. run_desktop.bat UTF-8 + 사전점검 + 조건부 pause | 6 |
| D. `.claude/settings.local.json` | 9 |
| E. 문서 업데이트 | 10 |
| 구현 시 확인 체크리스트 (proxy_tools, 집합 일치, __version__) | 1, 7, 8 |
| 자동 테스트(_port_open, _resolve_python, 로그 경로, _is_our_streamlit) | 2, 3, 4 |
