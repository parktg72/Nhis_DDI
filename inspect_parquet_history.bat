@echo off
chcp 65001 >nul
setlocal

set "ROOT=%~dp0"
set "PYTHONUTF8=1"

if "%~1"=="" (
    echo [ERROR] parquet 파일 경로가 필요합니다.
    echo [USAGE] inspect_parquet_history.bat ^<parquet_path^> [patient_id]
    exit /b 1
)

set "PARQUET_PATH=%~1"

set "PYTHON_BIN="
if exist "%ROOT%.venv_hana\Scripts\python.exe" set "PYTHON_BIN=%ROOT%.venv_hana\Scripts\python.exe"
if not defined PYTHON_BIN if exist "%ROOT%.venv\Scripts\python.exe" set "PYTHON_BIN=%ROOT%.venv\Scripts\python.exe"
if not defined PYTHON_BIN set "PYTHON_BIN=python"

set "EXTRA_ARGS="
if not "%~2"=="" set "EXTRA_ARGS=--patient-id %~2"

echo [INFO] path: %PARQUET_PATH%
echo [INFO] Python: %PYTHON_BIN%
"%PYTHON_BIN%" -m scripts.ops.inspect_parquet_history "%PARQUET_PATH%" %EXTRA_ARGS%
if errorlevel 1 (
    echo [ERROR] parquet 점검 실패
    exit /b 1
)

echo [OK] 점검 완료
endlocal
