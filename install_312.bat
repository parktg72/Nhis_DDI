@echo off
chcp 65001 >nul
REM ============================================================
REM Python 3.12 전용 오프라인 설치 스크립트
REM 폐쇄망 Windows PC에서 실행
REM
REM 사용법:
REM   install_312.bat          (가상환경 없이 시스템 Python 3.12 사용)
REM   install_312.bat venv     (.venv312 생성 후 설치)
REM ============================================================

setlocal EnableDelayedExpansion

set PROJECT_ROOT=%~dp0
set WIN_PKG_DIR=%PROJECT_ROOT%packages_win\py312
set HANA_PKG_DIR=%PROJECT_ROOT%hana\py312
set WIN_DIR=%PROJECT_ROOT%packages_win
set HANA_DIR=%PROJECT_ROOT%hana

echo ================================================
echo  NHIS 다재약물 DDI 시스템 - Python 3.12 설치
echo ================================================
echo  ETL/API 폴더 : %WIN_PKG_DIR%
echo  HANA/ML 폴더 : %HANA_PKG_DIR%
echo.

REM ── Python 3.12 경로 탐색 ────────────────────────────────────
set PYTHON_BIN=

REM 1순위: PATH 또는 활성화된 venv의 python 이 3.12인지 확인
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PY_VER_CHECK=%%v
echo !PY_VER_CHECK! | findstr /C:"3.12" >/dev/null 2>&1
if not errorlevel 1 (
    set PYTHON_BIN=python
    echo [확인] python 사용: !PY_VER_CHECK!
    goto :python_found
)

REM 2순위: py 런처
py -3.12 --version >/dev/null 2>&1
if not errorlevel 1 (
    set PYTHON_BIN=py -3.12
    echo [확인] py 런처: py -3.12
    goto :python_found
)

REM 3순위: 일반 설치 경로
if exist "C:\Python312\python.exe" (
    set PYTHON_BIN=C:\Python312\python.exe
    echo [확인] C:\Python312\python.exe
    goto :python_found
)
if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
    set PYTHON_BIN=%LOCALAPPDATA%\Programs\Python\Python312\python.exe
    echo [확인] %LOCALAPPDATA%\Programs\Python\Python312
    goto :python_found
)

echo.
echo [오류] Python 3.12 를 찾을 수 없습니다.
echo.
echo 해결 방법:
echo   1. Python 3.12 설치 후 "Add Python to PATH" 체크
echo   2. 또는 가상환경 먼저 활성화 후 실행:
echo        .venv\Scripts\activate
echo        install_312.bat
echo.
pause
exit /b 1

:python_found
echo.

REM ── 패키지 폴더 확인 ─────────────────────────────────────────
set PKG_MISSING=0
if not exist "%WIN_PKG_DIR%\" (
    echo [경고] packages_win\py312 없음
    set PKG_MISSING=1
)
if not exist "%HANA_PKG_DIR%\" (
    echo [경고] hana\py312 없음
    set PKG_MISSING=1
)
if "%PKG_MISSING%"=="1" (
    echo.
    echo 인터넷 환경에서 먼저 다운로드하세요:
    echo   packages_win\download.bat 312
    echo   hana\download.bat 312
    echo.
    pause
    exit /b 1
)

set FIND_LINKS=--find-links="%WIN_PKG_DIR%" --find-links="%HANA_PKG_DIR%"

REM ── 가상환경 ─────────────────────────────────────────────────
if /I "%1"=="venv" (
    set VENV_PATH=%PROJECT_ROOT%.venv312
    if not exist "!VENV_PATH!\" (
        echo 가상환경 생성 중: !VENV_PATH!
        %PYTHON_BIN% -m venv "!VENV_PATH!"
        if errorlevel 1 (
            echo [오류] 가상환경 생성 실패
            pause
            exit /b 1
        )
    ) else (
        echo 기존 가상환경 사용: !VENV_PATH!
    )
    set PYTHON_BIN=!VENV_PATH!\Scripts\python.exe
    echo.
)

REM ── 1단계: pip 업그레이드 ────────────────────────────────────
echo [1/5] pip 업그레이드...
%PYTHON_BIN% -m pip install --no-index %FIND_LINKS% --upgrade pip 2>/dev/null || echo       pip 업그레이드 건너뜀

REM ── 2단계: 핵심 패키지 ───────────────────────────────────────
echo.
echo [2/5] 핵심 데이터 처리 및 ML 패키지...
%PYTHON_BIN% -m pip install --no-index %FIND_LINKS% ^
    numpy pandas pyarrow scipy scikit-learn xgboost lightgbm shap joblib
if errorlevel 1 (
    echo [오류] 핵심 패키지 설치 실패
    pause
    exit /b 1
)

REM ── 3단계: HANA 연결 ─────────────────────────────────────────
echo.
echo [3/5] SAP HANA 연결 패키지...
%PYTHON_BIN% -m pip install --no-index %FIND_LINKS% hdbcli hana-ml
if errorlevel 1 echo [경고] HANA 패키지 실패 (HANA 미사용 시 무시 가능)

REM ── 4단계: requirements 전체 ─────────────────────────────────
echo.
echo [4/5] 전체 패키지 설치 (requirements.txt)...
%PYTHON_BIN% -m pip install --no-index %FIND_LINKS% --upgrade -r "%WIN_DIR%\requirements.txt"
%PYTHON_BIN% -m pip install --no-index %FIND_LINKS% --upgrade -r "%HANA_DIR%\requirements.txt"

REM ── 5단계: 검증 ──────────────────────────────────────────────
echo.
echo [5/5] 설치 검증...
echo.

set FAIL=0

echo [Python 버전]
%PYTHON_BIN% --version

echo.
echo [HANA 연결]
%PYTHON_BIN% -c "import hdbcli; print('  hdbcli', hdbcli.__version__, 'OK')" 2>/dev/null || (echo   [실패] hdbcli & set FAIL=1)
%PYTHON_BIN% -c "import hana_ml; print('  hana-ml', hana_ml.__version__, 'OK')" 2>/dev/null || (echo   [실패] hana-ml & set FAIL=1)

echo [데이터 처리]
%PYTHON_BIN% -c "import pandas, numpy, pyarrow, scipy; print('  pandas/numpy/pyarrow/scipy OK')" 2>/dev/null || (echo   [실패] 데이터 패키지 & set FAIL=1)
%PYTHON_BIN% -c "import statsmodels; print('  statsmodels OK')" 2>/dev/null || (echo   [실패] statsmodels & set FAIL=1)

echo [머신러닝]
%PYTHON_BIN% -c "import sklearn, xgboost, lightgbm, shap; print('  ML 패키지 OK')" 2>/dev/null || (echo   [실패] ML 패키지 & set FAIL=1)

echo [웹앱]
%PYTHON_BIN% -c "import streamlit; print('  Streamlit', streamlit.__version__, 'OK')" 2>/dev/null || (echo   [실패] Streamlit & set FAIL=1)
%PYTHON_BIN% -c "import fastapi, uvicorn; print('  FastAPI/uvicorn OK')" 2>/dev/null || (echo   [실패] FastAPI & set FAIL=1)

echo.
echo ================================================
if "%FAIL%"=="0" (
    echo  모든 패키지 설치 완료!
) else (
    echo  일부 실패 -- 위 [실패] 항목을 확인하세요.
)
echo.
echo  다음 단계:
echo    웹앱 실행 : hana_app\run.bat
echo    브라우저  : http://localhost:8501
echo ================================================
echo.
pause
endlocal
