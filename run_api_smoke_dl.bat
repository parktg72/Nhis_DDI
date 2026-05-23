@echo off
chcp 65001 >nul
setlocal

set "ROOT=%~dp0"
set "PYTHONUTF8=1"
set "DDI_SMOKE_HISTORY_PROVIDER=1"
if not "%~1"=="" set "ADMIN_API_KEY=%~1"
set "PORT=8000"
if not "%~2"=="" set "PORT=%~2"
if not defined MODEL_DIR set "MODEL_DIR=%ROOT%models"

set "PYTHON_BIN="
if exist "%ROOT%.venv_hana\Scripts\python.exe" set "PYTHON_BIN=%ROOT%.venv_hana\Scripts\python.exe"
if not defined PYTHON_BIN if exist "%ROOT%.venv\Scripts\python.exe" set "PYTHON_BIN=%ROOT%.venv\Scripts\python.exe"
if not defined PYTHON_BIN set "PYTHON_BIN=python"

if not defined ADMIN_API_KEY (
    echo [ERROR] ADMIN_API_KEY 환경변수가 필요합니다. 첫 번째 인자로도 전달할 수 있습니다.
    echo [USAGE] run_api_smoke_dl.bat [ADMIN_API_KEY] [PORT]
    exit /b 1
)

netstat /an 2>nul | findstr /C:":%PORT% " | findstr LISTENING >nul
if not errorlevel 1 (
    echo [WARNING] 포트 %PORT% 이 이미 사용 중입니다. 기존 서버 또는 다른 프로세스를 확인하세요.
)

echo [WARNING] DDI_SMOKE_HISTORY_PROVIDER=1 -- smoke DL 검증 전용입니다.
echo [INFO] API: http://127.0.0.1:%PORT%
echo [INFO] MODEL_DIR: %MODEL_DIR%
echo [INFO] Python: %PYTHON_BIN%
echo [INFO] 종료: Ctrl+C
"%PYTHON_BIN%" -m uvicorn serving.main:app --host 127.0.0.1 --port %PORT%
endlocal
