@echo off
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
    echo [ERROR] Python not found. Run install_all.bat first.
    pause
    exit /b 1
)

"%PYTHON_BIN%" -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] streamlit not installed. Run install_all.bat first.
    pause
    exit /b 1
)

echo Python : %PYTHON_BIN%
echo Script : %ROOT%desktop_app.py
echo.
echo Starting app...

"%PYTHON_BIN%" "%ROOT%desktop_app.py"

endlocal
