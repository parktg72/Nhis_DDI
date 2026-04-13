@echo off
chcp 65001 >nul
REM ============================================================
REM Windows 오프라인 설치 스크립트
REM 폐쇄망 Windows 에서 실행
REM
REM packages_win\pyXXX  +  hana\pyXXX  두 폴더를 동시에 검색하여
REM 어느 폴더에 있든 패키지를 찾아 설치합니다.
REM
REM 사용법:
REM   packages_win\install.bat           (자동 감지)
REM   packages_win\install.bat 310       (Python 3.10 지정)
REM   packages_win\install.bat 311 venv  (Python 3.11 + 가상환경)
REM ============================================================

setlocal EnableDelayedExpansion

set SCRIPT_DIR=%~dp0
set REQUIREMENTS=%SCRIPT_DIR%requirements.txt
set PROJECT_ROOT=%SCRIPT_DIR%..

REM Python 버전 감지
for /f "tokens=*" %%i in ('python -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')"') do (
    set PY_VERSION=%%i
)

if not "%1"=="" (
    set PY_VERSION=%1
)

REM 패키지 폴더 경로 (packages_win + hana 두 폴더 모두 참조)
set PKG_DIR=%SCRIPT_DIR%py%PY_VERSION%
set HANA_PKG_DIR=%PROJECT_ROOT%\hana\py%PY_VERSION%

echo ==============================================
echo Windows 오프라인 패키지 설치
echo ==============================================
echo Python 버전  : %PY_VERSION%
echo 기본 패키지  : %PKG_DIR%
echo HANA 패키지  : %HANA_PKG_DIR%
echo.

if not exist "%PKG_DIR%" (
    echo [오류] packages_win\py%PY_VERSION% 디렉토리 없음
    echo download.bat 를 먼저 실행하세요.
    exit /b 1
)

if not exist "%HANA_PKG_DIR%" (
    echo [경고] hana\py%PY_VERSION% 디렉토리 없음 — packages_win 단독 검색으로 진행
    set HANA_PKG_DIR=
)

REM 가상환경 옵션
set CREATE_VENV=0
if "%2"=="venv" set CREATE_VENV=1

if "%CREATE_VENV%"=="1" (
    set VENV_PATH=%PROJECT_ROOT%\.venv
    if not exist "!VENV_PATH!" (
        echo 가상환경 생성 중: !VENV_PATH!
        python -m venv "!VENV_PATH!"
    )
    set PYTHON_BIN=!VENV_PATH!\Scripts\python.exe
    echo 가상환경: !VENV_PATH!
) else (
    set PYTHON_BIN=python
)

echo.
echo ==============================================
echo [1단계] 오프라인 패키지 설치
echo         (packages_win + hana 동시 검색)
echo ==============================================

if "%HANA_PKG_DIR%"=="" (
    REM hana 폴더 없을 때: packages_win 단독
    %PYTHON_BIN% -m pip install ^
        --no-index ^
        --find-links="%PKG_DIR%" ^
        --upgrade ^
        -r "%REQUIREMENTS%"
) else (
    REM 두 폴더 동시 검색
    %PYTHON_BIN% -m pip install ^
        --no-index ^
        --find-links="%PKG_DIR%" ^
        --find-links="%HANA_PKG_DIR%" ^
        --upgrade ^
        -r "%REQUIREMENTS%"
)

echo.
echo ==============================================
echo [2단계] 설치 검증
echo ==============================================

%PYTHON_BIN% -c "import pandas, numpy, yaml, requests; print('  핵심 패키지 OK')" 2>nul || echo   [실패] 핵심 패키지
%PYTHON_BIN% -c "import lxml; print('  lxml OK')" 2>nul || echo   [실패] lxml
%PYTHON_BIN% -c "import xgboost, lightgbm, sklearn; print('  ML 패키지 OK')" 2>nul || echo   [실패] ML 패키지
%PYTHON_BIN% -c "import fastapi, uvicorn; print('  API 패키지 OK')" 2>nul || echo   [실패] API 패키지
%PYTHON_BIN% -c "import hdbcli; print('  hdbcli OK')" 2>nul || echo   [정보] hdbcli 미설치 (hana\install.bat 로 설치)
%PYTHON_BIN% -c "import streamlit; print('  Streamlit OK')" 2>nul || echo   [정보] Streamlit 미설치 (hana\install.bat 로 설치)

echo.
echo ==============================================
echo 설치 완료!
echo.
echo 다음 단계: python scripts\parse_drugbank.py
echo HANA 연동이 필요하면: hana\install.bat %PY_VERSION% venv
echo ==============================================
endlocal
