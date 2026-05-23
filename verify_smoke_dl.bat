@echo off
chcp 65001 >nul
setlocal

set "ROOT=%~dp0"
set "PYTHONUTF8=1"
set "BASE_URL=%~1"
if "%BASE_URL%"=="" set "BASE_URL=http://127.0.0.1:8000"
if not "%~2"=="" set "ADMIN_API_KEY=%~2"
if not defined MODEL_DIR set "MODEL_DIR=%ROOT%models"
set "EXTRA_ARGS="
if /I "%~3"=="--require-dl-prediction" set "EXTRA_ARGS=--require-dl-prediction"
if /I "%~3"=="--skip-validation" set "EXTRA_ARGS=--skip-validation"
if /I "%~4"=="--require-dl-prediction" set "EXTRA_ARGS=%EXTRA_ARGS% --require-dl-prediction"
if /I "%~4"=="--skip-validation" set "EXTRA_ARGS=%EXTRA_ARGS% --skip-validation"

set "PYTHON_BIN="
if exist "%ROOT%.venv_hana\Scripts\python.exe" set "PYTHON_BIN=%ROOT%.venv_hana\Scripts\python.exe"
if not defined PYTHON_BIN if exist "%ROOT%.venv\Scripts\python.exe" set "PYTHON_BIN=%ROOT%.venv\Scripts\python.exe"
if not defined PYTHON_BIN set "PYTHON_BIN=python"

if not defined ADMIN_API_KEY (
    echo [ERROR] ADMIN_API_KEY 환경변수가 필요합니다. 두 번째 인자로도 전달할 수 있습니다.
    echo [USAGE] verify_smoke_dl.bat [http://127.0.0.1:8000] [ADMIN_API_KEY] [--require-dl-prediction] [--skip-validation]
    exit /b 1
)

echo [INFO] API: %BASE_URL%
echo [INFO] MODEL_DIR: %MODEL_DIR%
echo [INFO] Python: %PYTHON_BIN%
"%PYTHON_BIN%" -m scripts.ops.verify_smoke_dl "%BASE_URL%" --admin-key "%ADMIN_API_KEY%" --model-dir "%MODEL_DIR%" %EXTRA_ARGS%
if errorlevel 1 (
    echo [ERROR] smoke DL reload/predict 검증 실패
    exit /b 1
)

echo [OK] smoke DL reload/predict 검증 완료
endlocal
