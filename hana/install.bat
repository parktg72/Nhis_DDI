@echo off
chcp 65001 >nul
REM ============================================================
REM SAP HANA 패키지 폐쇄망 설치 스크립트
REM 폐쇄망 Windows PC에서 실행
REM
REM packages_win\pyXXX  +  hana\pyXXX  두 폴더를 동시에 검색하여
REM 어느 폴더에 있든 패키지를 찾아 설치합니다.
REM
REM 사용법:
REM   hana\install.bat               (Python 버전 자동 감지)
REM   hana\install.bat 311           (Python 3.11 지정)
REM   hana\install.bat 311 venv      (Python 3.11 + 가상환경 생성)
REM ============================================================

setlocal EnableDelayedExpansion

set SCRIPT_DIR=%~dp0
set REQUIREMENTS=%SCRIPT_DIR%requirements.txt
set PROJECT_ROOT=%SCRIPT_DIR%..

REM Python 버전 자동 감지
for /f "tokens=*" %%i in ('python -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')"') do (
    set PY_VERSION=%%i
)

if not "%1"=="" (
    set PY_VERSION=%1
)

REM 패키지 폴더 경로 (hana + packages_win 두 폴더 모두 참조)
set PKG_DIR=%SCRIPT_DIR%py%PY_VERSION%
set WIN_PKG_DIR=%PROJECT_ROOT%\packages_win\py%PY_VERSION%

echo ==============================================
echo SAP HANA 패키지 오프라인 설치
echo ==============================================
echo Python 버전  : %PY_VERSION%
echo HANA 패키지  : %PKG_DIR%
echo 기본 패키지  : %WIN_PKG_DIR%
echo.

if not exist "%PKG_DIR%" (
    echo [오류] hana\py%PY_VERSION% 디렉토리 없음
    echo download.bat 를 먼저 실행하세요.
    exit /b 1
)

if not exist "%WIN_PKG_DIR%" (
    echo [경고] packages_win\py%PY_VERSION% 디렉토리 없음 — hana 단독 검색으로 진행
    set WIN_PKG_DIR=
)

REM 가상환경 옵션
set CREATE_VENV=0
if "%2"=="venv" set CREATE_VENV=1

if "%CREATE_VENV%"=="1" (
    set VENV_PATH=%PROJECT_ROOT%\.venv_hana
    if not exist "!VENV_PATH!" (
        echo 가상환경 생성 중: !VENV_PATH!
        python -m venv "!VENV_PATH!"
    )
    set PYTHON_BIN=!VENV_PATH!\Scripts\python.exe
    echo 가상환경: !VENV_PATH!
) else (
    set PYTHON_BIN=python
)

REM --find-links 옵션 조합
if "%WIN_PKG_DIR%"=="" (
    set FIND_LINKS=--find-links="%PKG_DIR%"
) else (
    set FIND_LINKS=--find-links="%PKG_DIR%" --find-links="%WIN_PKG_DIR%"
)

REM Python 3.12 일 때만 constraints-py312.txt 적용 (dev/prod 패리티)
set CONSTRAINT=
if "%PY_VERSION%"=="312" (
    set CONSTRAINT=--constraint "%PROJECT_ROOT%\constraints-py312.txt"
)

echo.
echo ==============================================
echo [1단계] pip 업그레이드
echo ==============================================
%PYTHON_BIN% -m pip install --no-index %FIND_LINKS% --upgrade pip 2>nul || (
    echo    pip 업그레이드 건너뜀 ^(오프라인 pip 없음^)
)

echo.
echo ==============================================
echo [2단계] HANA 연결 핵심 패키지 설치
echo         (hana + packages_win 동시 검색)
echo ==============================================
%PYTHON_BIN% -m pip install ^
    --no-index ^
    %FIND_LINKS% ^
    %CONSTRAINT% ^
    hdbcli hana-ml

echo.
echo ==============================================
echo [3단계] 전체 패키지 설치
echo         (hana + packages_win 동시 검색)
echo ==============================================
%PYTHON_BIN% -m pip install ^
    --no-index ^
    %FIND_LINKS% ^
    %CONSTRAINT% ^
    --upgrade ^
    -r "%REQUIREMENTS%"

echo.
echo ==============================================
echo [4단계] 설치 검증
echo ==============================================

echo [HANA 연결]
%PYTHON_BIN% -c "import hdbcli; print('  hdbcli', hdbcli.__version__, 'OK')" 2>nul || echo   [실패] hdbcli
%PYTHON_BIN% -c "import hana_ml; print('  hana-ml', hana_ml.__version__, 'OK')" 2>nul || echo   [실패] hana-ml

echo [데이터 처리]
%PYTHON_BIN% -c "import pandas, numpy; print('  pandas', pandas.__version__, '/ numpy', numpy.__version__, 'OK')" 2>nul || echo   [실패] pandas/numpy
%PYTHON_BIN% -c "import scipy, statsmodels; print('  scipy/statsmodels OK')" 2>nul || echo   [실패] scipy/statsmodels

echo [머신러닝]
%PYTHON_BIN% -c "import sklearn, xgboost, lightgbm, shap; print('  sklearn/xgboost/lightgbm/shap OK')" 2>nul || echo   [실패] ML 패키지

echo [웹앱]
%PYTHON_BIN% -c "import streamlit; print('  Streamlit', streamlit.__version__, 'OK')" 2>nul || echo   [실패] Streamlit
%PYTHON_BIN% -c "import plotly; print('  plotly OK')" 2>nul || echo   [실패] plotly

echo [Jupyter]
%PYTHON_BIN% -c "import jupyter; print('  Jupyter OK')" 2>nul || echo   [실패] Jupyter

echo [ETL / API ^(packages_win^)]
%PYTHON_BIN% -c "import lxml; print('  lxml OK')" 2>nul || echo   [정보] lxml 미설치 (packages_win\install.bat 로 추가 설치 가능)
%PYTHON_BIN% -c "import fastapi; print('  fastapi OK')" 2>nul || echo   [정보] fastapi 미설치 (packages_win\install.bat 로 추가 설치 가능)

echo.
echo ==============================================
echo 설치 완료!
echo.
echo HANA 학습 웹앱 실행:
echo   hana_app\run.bat
echo   브라우저: http://localhost:8501
echo.
echo HANA DB 연결 테스트:
echo   python -c "from hdbcli import dbapi; conn = dbapi.connect(address='HOST', port=30015, user='USER', password='PW'); print('연결 성공')"
echo.
echo 전체 시스템(ETL+API+HANA) 설치는:
echo   install_all.bat %PY_VERSION% venv
echo ==============================================
endlocal
