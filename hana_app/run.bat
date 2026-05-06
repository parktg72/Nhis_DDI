@echo off
chcp 65001 >nul
REM ============================================================
REM NHIS 다재약물 DDI 위험도 분류 - 웹앱 실행 스크립트
REM 폐쇄망 Windows 환경에서 실행
REM
REM 사용법:
REM   hana_app\run.bat              (기본 8501 포트)
REM   hana_app\run.bat 8080         (포트 지정)
REM   hana_app\run.bat 8501 venv    (가상환경 사용)
REM ============================================================

setlocal EnableDelayedExpansion

set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..
set PORT=8501
set APP_FILE=%SCRIPT_DIR%app.py

if not "%1"=="" set PORT=%1

REM Python 바이너리 결정 (가상환경 자동 감지)
REM 우선순위: .venv_hana > .venv > .venv312 > 시스템 python
REM (.venv 는 install_all.bat / packages_win/install.bat / uv venv 가 생성하는
REM  표준 이름. .venv_hana 는 install_312.bat 가 생성하는 별칭. run_desktop.bat
REM  과 일관성 유지.)
set PYTHON_BIN=python
if exist "%PROJECT_ROOT%\.venv_hana\Scripts\python.exe" (
    set PYTHON_BIN=%PROJECT_ROOT%\.venv_hana\Scripts\python.exe
    echo 가상환경 사용: .venv_hana
) else if exist "%PROJECT_ROOT%\.venv\Scripts\python.exe" (
    set PYTHON_BIN=%PROJECT_ROOT%\.venv\Scripts\python.exe
    echo 가상환경 사용: .venv
) else if exist "%PROJECT_ROOT%\.venv312\Scripts\python.exe" (
    set PYTHON_BIN=%PROJECT_ROOT%\.venv312\Scripts\python.exe
    echo 가상환경 사용: .venv312
) else (
    echo 시스템 Python 사용 ^(가상환경 미감지^)
    REM 시스템 Python이 3.12인지 확인
    for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set SYS_VER=%%v
    echo !SYS_VER! | findstr /C:"3.12" >nul 2>&1
    if errorlevel 1 (
        echo [경고] Python 3.12 권장. 현재 버전: !SYS_VER!
        echo        install_312.bat venv 를 먼저 실행하세요.
    )
)

echo ==============================================
echo NHIS 다재약물 DDI 위험도 분류 시스템
echo ==============================================
echo URL: http://localhost:%PORT%
echo 종료: Ctrl+C
echo.

REM Streamlit 설치 확인
%PYTHON_BIN% -c "import streamlit" 2>nul
if errorlevel 1 (
    echo [오류] streamlit이 설치되지 않았습니다.
    echo install_312.bat venv 를 먼저 실행하세요.
    pause
    exit /b 1
)

REM hdbcli 설치 확인
%PYTHON_BIN% -c "import hdbcli" 2>nul
if errorlevel 1 (
    echo [경고] hdbcli가 설치되지 않았습니다. HANA DB 연결 기능이 제한됩니다.
    echo install_312.bat venv 를 실행하여 hdbcli를 설치하세요.
)

%PYTHON_BIN% -m streamlit run "%APP_FILE%" ^
    --server.port %PORT% ^
    --server.address localhost ^
    --server.headless true ^
    --browser.gatherUsageStats false ^
    --theme.base light ^
    --theme.primaryColor "#1f77b4"

endlocal
