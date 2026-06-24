@echo off
chcp 65001 >nul
setlocal
set "PYTHONUTF8=1"

REM 최소 배치: 직접 Python 실행과 동등한 경로.
REM 사전 subprocess 체크(import streamlit/webview)를 제거 — 사내 엔드포인트
REM 보안이 짧은 Python 서브프로세스 스폰 + 출력 리다이렉트 패턴을 차단하여
REM "인터넷 보안 설정으로 인해 하나 이상의 파일을 복사 할 수 없습니다" 팝업
REM 을 띄우던 원인. 의존성 검증은 desktop_app.py 내부로 이동.

set PYTHON_BIN=
if exist "%~dp0.venv_hana\Scripts\python.exe" set PYTHON_BIN=%~dp0.venv_hana\Scripts\python.exe
if not defined PYTHON_BIN if exist "%~dp0.venv\Scripts\python.exe" set PYTHON_BIN=%~dp0.venv\Scripts\python.exe
if not defined PYTHON_BIN if exist "%~dp0venv\Scripts\python.exe" set PYTHON_BIN=%~dp0venv\Scripts\python.exe

if not defined PYTHON_BIN (
    echo [ERROR] Python venv not found. Run install_312.bat venv first.
    pause
    exit /b 1
)

"%PYTHON_BIN%" "%~dp0desktop_app.py"
if errorlevel 1 pause

endlocal
