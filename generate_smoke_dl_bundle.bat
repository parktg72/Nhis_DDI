@echo off
chcp 65001 >nul
setlocal

set "ROOT=%~dp0"
set "PYTHONUTF8=1"
set "OUT_DIR=%ROOT%models\dl\smoke"
if not "%~1"=="" set "OUT_DIR=%~1"

set "PYTHON_BIN="
if exist "%ROOT%.venv_hana\Scripts\python.exe" set "PYTHON_BIN=%ROOT%.venv_hana\Scripts\python.exe"
if not defined PYTHON_BIN if exist "%ROOT%.venv\Scripts\python.exe" set "PYTHON_BIN=%ROOT%.venv\Scripts\python.exe"
if not defined PYTHON_BIN set "PYTHON_BIN=python"

echo [INFO] smoke DL bundle 생성 경로: %OUT_DIR%
echo [INFO] Python: %PYTHON_BIN%
"%PYTHON_BIN%" -m scripts.datasets.smoke_dl_bundle "%OUT_DIR%" --run-id smoke-deploy --schema-version dl.v1.smoke --lookback-days 365
if errorlevel 1 (
    echo [ERROR] smoke DL bundle 생성 실패
    echo [ERROR] 먼저 install_312.bat venv 로 .venv_hana 및 CUDA DL package 설치를 확인하세요.
    exit /b 1
)

echo [OK] smoke DL bundle 생성 완료: %OUT_DIR%
echo [NEXT] /admin/reload/dl bundle_dir=%OUT_DIR%
endlocal
