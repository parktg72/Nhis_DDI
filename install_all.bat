@echo off
chcp 65001 >nul
REM ============================================================
REM 전체 시스템 통합 오프라인 설치 스크립트
REM 폐쇄망 Windows PC에서 실행
REM
REM packages_win\pyXXX + hana\pyXXX 두 폴더를 동시에 검색하여
REM DDI 시스템 전체 패키지를 한 번에 설치합니다.
REM
REM 포함 항목:
REM   - DDI ETL / API 서빙 패키지  (packages_win)
REM   - SAP HANA 연결 패키지        (hana)
REM   - 머신러닝 + Streamlit 웹앱   (hana)
REM   - Jupyter 노트북 환경          (hana)
REM
REM 사용법:
REM   install_all.bat                  (자동 감지, 시스템 Python)
REM   install_all.bat 311              (Python 3.11 지정)
REM   install_all.bat 311 venv         (Python 3.11 + 가상환경 .venv)
REM ============================================================

setlocal EnableDelayedExpansion

set PROJECT_ROOT=%~dp0
set WIN_DIR=%PROJECT_ROOT%packages_win
set HANA_DIR=%PROJECT_ROOT%hana

REM Python 버전 자동 감지
for /f "tokens=*" %%i in ('python -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')"') do (
    set PY_VERSION=%%i
)

if not "%1"=="" set PY_VERSION=%1

set WIN_PKG_DIR=%WIN_DIR%\py%PY_VERSION%
set HANA_PKG_DIR=%HANA_DIR%\py%PY_VERSION%

echo ================================================
echo  NHIS 다재약물 DDI 시스템 - 전체 패키지 설치
echo ================================================
echo  Python 버전  : %PY_VERSION%
echo  ETL/API 폴더 : %WIN_PKG_DIR%
echo  HANA/ML 폴더 : %HANA_PKG_DIR%
echo.

REM 폴더 존재 확인
set PKG_MISSING=0
if not exist "%WIN_PKG_DIR%" (
    echo [경고] packages_win\py%PY_VERSION% 없음
    set PKG_MISSING=1
)
if not exist "%HANA_PKG_DIR%" (
    echo [경고] hana\py%PY_VERSION% 없음
    set PKG_MISSING=1
)
if "%PKG_MISSING%"=="1" (
    echo.
    echo 인터넷 연결 환경에서 먼저 실행하세요:
    echo   packages_win\download.bat %PY_VERSION%
    echo   hana\download.bat %PY_VERSION%
    pause
    exit /b 1
)

REM --find-links 이중 경로
set FIND_LINKS=--find-links="%WIN_PKG_DIR%" --find-links="%HANA_PKG_DIR%"

REM 가상환경 설정
set CREATE_VENV=0
if "%2"=="venv" set CREATE_VENV=1

if "%CREATE_VENV%"=="1" (
    set VENV_PATH=%PROJECT_ROOT%.venv
    if not exist "!VENV_PATH!" (
        echo 가상환경 생성 중: !VENV_PATH!
        python -m venv "!VENV_PATH!"
    )
    set PYTHON_BIN=!VENV_PATH!\Scripts\python.exe
    echo 가상환경: !VENV_PATH!
    echo.
) else (
    set PYTHON_BIN=python
)

REM ── 1단계: pip 업그레이드 ──────────────────────────────────
echo [1/5] pip 업그레이드...
%PYTHON_BIN% -m pip install --no-index %FIND_LINKS% --upgrade pip 2>nul || (
    echo       pip 업그레이드 건너뜀
)

REM ── 2단계: 핵심 패키지 (data + ML) ──────────────────────────
echo.
echo [2/5] 핵심 데이터 처리 및 ML 패키지 설치...
%PYTHON_BIN% -m pip install ^
    --no-index ^
    %FIND_LINKS% ^
    numpy pandas pyarrow scipy scikit-learn xgboost lightgbm shap joblib

REM ── 3단계: HANA 연결 ─────────────────────────────────────────
echo.
echo [3/5] SAP HANA 연결 패키지 설치...

REM setuptools 부트스트랩 (pydotplus 소스 빌드에 필요)
%PYTHON_BIN% -m ensurepip --upgrade >nul 2>&1
%PYTHON_BIN% -m pip install --no-index %FIND_LINKS% --upgrade setuptools wheel >nul 2>&1
if errorlevel 1 (
    echo       [참고] setuptools 업그레이드 건너뜀 ^(시스템 버전 사용^)
)

REM hdbcli 는 wheel 이므로 그대로 설치
%PYTHON_BIN% -m pip install ^
    --no-index ^
    %FIND_LINKS% ^
    hdbcli

REM hana-ml 의존성 중 pydotplus 는 .tar.gz 소스 배포판이므로 --no-build-isolation 사용
%PYTHON_BIN% -m pip install ^
    --no-index ^
    %FIND_LINKS% ^
    --no-deps ^
    hana-ml

%PYTHON_BIN% -m pip install ^
    --no-index ^
    %FIND_LINKS% ^
    Deprecated schedule prettytable shapely

%PYTHON_BIN% -m pip install ^
    --no-build-isolation ^
    --no-index ^
    %FIND_LINKS% ^
    pydotplus

REM ── 4단계: 전체 requirements 설치 ───────────────────────────
echo.
echo [4/5] 전체 패키지 설치 (packages_win + hana requirements)...

REM packages_win requirements
%PYTHON_BIN% -m pip install ^
    --no-index ^
    %FIND_LINKS% ^
    --upgrade ^
    -r "%WIN_DIR%\requirements.txt"

REM hana requirements
%PYTHON_BIN% -m pip install ^
    --no-index ^
    %FIND_LINKS% ^
    --upgrade ^
    -r "%HANA_DIR%\requirements.txt"

REM ── 5단계: 검증 ──────────────────────────────────────────────
echo.
echo [5/5] 설치 검증...
echo.

set FAIL=0

echo [HANA 연결]
%PYTHON_BIN% -c "import hdbcli; print('  hdbcli', hdbcli.__version__, 'OK')" 2>nul || (echo   [실패] hdbcli & set FAIL=1)
%PYTHON_BIN% -c "import hana_ml; print('  hana-ml', hana_ml.__version__, 'OK')" 2>nul || (echo   [실패] hana-ml & set FAIL=1)

echo [데이터 처리]
%PYTHON_BIN% -c "import pandas, numpy, pyarrow, scipy; print('  pandas/numpy/pyarrow/scipy OK')" 2>nul || (echo   [실패] 데이터 패키지 & set FAIL=1)
%PYTHON_BIN% -c "import statsmodels; print('  statsmodels OK')" 2>nul || (echo   [실패] statsmodels & set FAIL=1)
%PYTHON_BIN% -c "import lxml; print('  lxml OK')" 2>nul || (echo   [실패] lxml & set FAIL=1)

echo [머신러닝]
%PYTHON_BIN% -c "import sklearn, xgboost, lightgbm, shap; print('  sklearn/xgboost/lightgbm/shap OK')" 2>nul || (echo   [실패] ML 패키지 & set FAIL=1)
%PYTHON_BIN% -c "import catboost; print('  catboost', catboost.__version__, 'OK')" 2>nul || (echo   [실패] catboost & set FAIL=1)

echo [웹앱 / API]
%PYTHON_BIN% -c "import streamlit; print('  Streamlit', streamlit.__version__, 'OK')" 2>nul || (echo   [실패] Streamlit & set FAIL=1)
%PYTHON_BIN% -c "import plotly, matplotlib; print('  plotly/matplotlib OK')" 2>nul || (echo   [실패] 시각화 패키지 & set FAIL=1)
%PYTHON_BIN% -c "import fastapi, uvicorn; print('  FastAPI/uvicorn OK')" 2>nul || (echo   [실패] FastAPI & set FAIL=1)

echo [모니터링]
%PYTHON_BIN% -c "import prometheus_client; print('  prometheus_client OK')" 2>nul || (echo   [실패] prometheus_client & set FAIL=1)

echo [Jupyter]
%PYTHON_BIN% -c "import jupyter; print('  Jupyter OK')" 2>nul || (echo   [실패] Jupyter & set FAIL=1)

echo.
echo ================================================
if "%FAIL%"=="0" (
    echo  모든 패키지 설치 완료!
) else (
    echo  일부 패키지 설치 실패 — 위 [실패] 항목을 확인하세요.
)
echo.
echo  다음 단계:
echo    1. DrugBank 파싱  : python scripts\parse_drugbank.py
echo    2. HANA 웹앱 실행 : hana_app\run.bat %PY_VERSION% venv
echo    3. 브라우저 접속  : http://localhost:8501
echo ================================================
endlocal
