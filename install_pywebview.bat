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
