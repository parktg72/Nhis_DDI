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
