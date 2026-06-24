@echo off
chcp 65001 >nul
REM ============================================================
REM Unified offline installer for Windows closed-network PCs
REM Searches packages_win\pyXXX and hana\pyXXX together.
REM
REM Includes:
REM   - DDI ETL / API packages        (packages_win)
REM   - SAP HANA connectivity         (hana)
REM   - ML + Streamlit web app        (hana)
REM   - DOCX/chart report packages    (packages_win/hana)
REM   - Jupyter is excluded from production offline install
REM
REM Usage:
REM   install_all.bat                  (auto-detect Python, system env)
REM   install_all.bat 311              (Python 3.11, system env)
REM   install_all.bat 312 venv         (Python 3.12 + .venv)
REM ============================================================

setlocal EnableDelayedExpansion
set "PYTHONUTF8=1"

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

REM Python 3.12 일 때만 constraints-py312.txt 적용 (dev/prod 패리티)
set CONSTRAINT=
if "%PY_VERSION%"=="312" (
    set CONSTRAINT=--constraint "%PROJECT_ROOT%constraints-py312.txt"
)

REM 가상환경 설정
set CREATE_VENV=0
if "%2"=="venv" set CREATE_VENV=1

if "%CREATE_VENV%"=="1" (
    set VENV_PATH=%PROJECT_ROOT%.venv

    REM Existing .venv may be a copied WSL/Linux venv or otherwise broken.
    if exist "!VENV_PATH!\" if not exist "!VENV_PATH!\Scripts\python.exe" (
        echo [경고] 기존 .venv 가 Windows 가상환경이 아니거나 손상되었습니다. 재생성합니다.
        rmdir /S /Q "!VENV_PATH!"
    )
    if exist "!VENV_PATH!\Scripts\python.exe" (
        "!VENV_PATH!\Scripts\python.exe" --version >nul 2>&1
        if errorlevel 1 (
            echo [경고] 기존 .venv Python 실행 실패. 재생성합니다.
            rmdir /S /Q "!VENV_PATH!"
        )
    )

    if not exist "!VENV_PATH!\Scripts\python.exe" (
        echo 가상환경 생성 중: !VENV_PATH!
        if "%PY_VERSION%"=="312" (
            py -3.12 -m venv "!VENV_PATH!" 2>nul || python -m venv "!VENV_PATH!"
        ) else (
            python -m venv "!VENV_PATH!"
        )
        if errorlevel 1 (
            echo [오류] 가상환경 생성 실패
            pause
            exit /b 1
        )
    )

    set PYTHON_BIN=!VENV_PATH!\Scripts\python.exe
    "!PYTHON_BIN!" --version >nul 2>&1
    if errorlevel 1 (
        echo [오류] 가상환경 Python 실행 실패: !PYTHON_BIN!
        pause
        exit /b 1
    )
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
    %CONSTRAINT% ^
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
    %CONSTRAINT% ^
    hdbcli
if errorlevel 1 (
    echo [오류] hdbcli 설치 실패
    pause
    exit /b 1
)

REM hana-ml 런타임 의존성을 먼저 설치해야 resolver 경고가 발생하지 않음
%PYTHON_BIN% -m pip install ^
    --no-index ^
    %FIND_LINKS% ^
    %CONSTRAINT% ^
    jinja2 plotly Deprecated schedule prettytable shapely
if errorlevel 1 (
    echo [오류] hana-ml 런타임 의존성 설치 실패
    pause
    exit /b 1
)

REM pydotplus 는 소스 배포판일 수 있으므로 setuptools/wheel 설치 후 build isolation 없이 설치
%PYTHON_BIN% -m pip install ^
    --no-build-isolation ^
    --no-index ^
    %FIND_LINKS% ^
    %CONSTRAINT% ^
    pydotplus
if errorlevel 1 (
    echo [오류] pydotplus 설치 실패
    pause
    exit /b 1
)

REM hana-ml 본체는 의존성 설치 후 --no-deps 로 고정 설치
%PYTHON_BIN% -m pip install ^
    --no-index ^
    %FIND_LINKS% ^
    %CONSTRAINT% ^
    --no-deps ^
    hana-ml
if errorlevel 1 (
    echo [오류] hana-ml 설치 실패
    pause
    exit /b 1
)

REM ── 4단계: 전체 requirements 설치 ───────────────────────────
echo.
echo [4/5] 전체 패키지 설치 (packages_win + hana requirements)...

REM packages_win requirements
%PYTHON_BIN% -m pip install ^
    --no-index ^
    %FIND_LINKS% ^
    %CONSTRAINT% ^
    --upgrade ^
    -r "%WIN_DIR%\requirements.txt"

REM hana requirements
%PYTHON_BIN% -m pip install ^
    --no-index ^
    %FIND_LINKS% ^
    %CONSTRAINT% ^
    --upgrade ^
    -r "%HANA_DIR%\requirements.txt"

REM Phase 3 딥러닝 학습 패키지 명시 보장 (TabNet/GNN/Transformer 선택용)
%PYTHON_BIN% -m pip install ^
    --no-index ^
    %FIND_LINKS% ^
    %CONSTRAINT% ^
    --upgrade ^
    torch pytorch-tabnet
if errorlevel 1 (
    if /I "%DDI_REQUIRE_PHASE3_DL%"=="1" (
        echo [오류] Phase 3 DL 학습 패키지 설치 실패 ^(DDI_REQUIRE_PHASE3_DL=1^)
        pause
        exit /b 1
    ) else (
        echo [경고] Phase 3 DL 학습 패키지 설치 실패 -- ML-only 실행은 가능
        echo        인터넷 환경에서 download_all.bat %PY_VERSION% 또는 packages_win\download_cuda_cu126.bat 실행 후 다시 설치하세요.
    )
)

REM Python 3.12 CUDA/PyG 운영 DL wheel set (선택)
set CUDA_REQ=%WIN_DIR%\requirements_cuda_cu126.txt
if "%PY_VERSION%"=="312" if exist "%CUDA_REQ%" (
    %PYTHON_BIN% -m pip install --no-index %FIND_LINKS% --upgrade -r "%CUDA_REQ%"
    if errorlevel 1 (
        if /I "%DDI_REQUIRE_CUDA_DL%"=="1" (
            echo [오류] CUDA DL 패키지 설치 실패 ^(DDI_REQUIRE_CUDA_DL=1^)
            pause
            exit /b 1
        ) else (
            echo [경고] CUDA DL 패키지 설치 실패 -- CPU/ML 학습은 가능
        )
    )
)

REM 결과분석 DOCX/그래프 보고서 핵심 패키지 명시 보장
%PYTHON_BIN% -m pip install ^
    --no-index ^
    %FIND_LINKS% ^
    %CONSTRAINT% ^
    --upgrade ^
    python-docx lxml matplotlib Pillow
if errorlevel 1 (
    echo [오류] DOCX/그래프 보고서 패키지 설치 실패
    pause
    exit /b 1
)

REM ── 5단계: 검증 ──────────────────────────────────────────────
echo.
echo [5/5] 설치 검증...
echo.

set FAIL=0

echo [HANA 연결]
%PYTHON_BIN% -c "import hdbcli; print('  hdbcli', hdbcli.__version__, 'OK')" 2>nul || (echo   [실패] hdbcli & set FAIL=1)
%PYTHON_BIN% -c "import hana_ml, jinja2, plotly, pydotplus; print('  hana-ml', hana_ml.__version__, 'OK')" 2>nul || (echo   [실패] hana-ml 의존성 & set FAIL=1)

echo [데이터 처리]
%PYTHON_BIN% -c "import pandas, numpy, pyarrow, scipy; print('  pandas/numpy/pyarrow/scipy OK')" 2>nul || (echo   [실패] 데이터 패키지 & set FAIL=1)
%PYTHON_BIN% -c "import statsmodels; print('  statsmodels OK')" 2>nul || (echo   [실패] statsmodels & set FAIL=1)
%PYTHON_BIN% -c "import lxml; print('  lxml OK')" 2>nul || (echo   [실패] lxml & set FAIL=1)

echo [머신러닝]
%PYTHON_BIN% -c "import sklearn, xgboost, lightgbm, shap; print('  sklearn/xgboost/lightgbm/shap OK')" 2>nul || (echo   [실패] ML 패키지 & set FAIL=1)
%PYTHON_BIN% -c "import catboost; print('  catboost', catboost.__version__, 'OK')" 2>nul || (echo   [실패] catboost & set FAIL=1)
%PYTHON_BIN% -c "import torch, pytorch_tabnet; print('  Phase3 DL training OK: torch', torch.__version__)" 2>nul || (if /I "%DDI_REQUIRE_PHASE3_DL%"=="1" (echo   [실패] Phase3 DL 학습 패키지 ^(DDI_REQUIRE_PHASE3_DL=1^) & set FAIL=1) else echo   [경고] Phase3 DL 학습 패키지 미설치 -- TabNet/GNN/Transformer 선택 시 실패)
%PYTHON_BIN% -c "import torch; print('  torch', torch.__version__, 'cuda=', torch.version.cuda, 'available=', torch.cuda.is_available()); raise SystemExit(0 if torch.cuda.is_available() else 1)" 2>nul || (if /I "%DDI_REQUIRE_CUDA_DL%"=="1" (echo   [실패] CUDA PyTorch ^(DDI_REQUIRE_CUDA_DL=1^) & set FAIL=1) else echo   [경고] CUDA PyTorch 비활성 -- DL 추론 전 CUDA wheel/driver 확인 필요)
%PYTHON_BIN% -c "import torch_geometric, pyg_lib, torch_scatter, torch_sparse, torch_cluster; print('  PyG CUDA companion packages OK')" 2>nul || (if /I "%DDI_REQUIRE_CUDA_DL%"=="1" (echo   [실패] PyG companion packages ^(DDI_REQUIRE_CUDA_DL=1^) & set FAIL=1) else echo   [경고] PyG companion packages 미설치 -- DL GNN 추론 전 wheel set 보강 필요)

echo [웹앱 / API]
%PYTHON_BIN% -c "import streamlit; print('  Streamlit', streamlit.__version__, 'OK')" 2>nul || (echo   [실패] Streamlit & set FAIL=1)
%PYTHON_BIN% -c "import plotly, matplotlib; print('  plotly/matplotlib OK')" 2>nul || (echo   [실패] 시각화 패키지 & set FAIL=1)
%PYTHON_BIN% -c "import fastapi, uvicorn; print('  FastAPI/uvicorn OK')" 2>nul || (echo   [실패] FastAPI & set FAIL=1)

echo [보고서/DOCX]
%PYTHON_BIN% -c "import docx, lxml, matplotlib; from PIL import Image; print('  python-docx/lxml/matplotlib/Pillow OK')" 2>nul || (echo   [실패] DOCX 보고서 패키지 & set FAIL=1)
%PYTHON_BIN% -c "from hana_app.core.report_exporter import DOCX_AVAILABLE, MPL_AVAILABLE; raise SystemExit(0 if DOCX_AVAILABLE and MPL_AVAILABLE else 1)" 2>nul || (echo   [실패] report_exporter DOCX/그래프 backend & set FAIL=1)

echo [모니터링]
%PYTHON_BIN% -c "import prometheus_client; print('  prometheus_client OK')" 2>nul || (echo   [실패] prometheus_client & set FAIL=1)

echo [Jupyter]
%PYTHON_BIN% -c "import jupyter; print('  Jupyter OK')" 2>nul || echo   [정보] Jupyter 미설치 (운영 오프라인 설치 제외)

echo.
echo ================================================
if "%FAIL%"=="0" (
    echo  모든 패키지 설치 완료!
) else (
    echo  일부 패키지 설치 실패 — 위 [실패] 항목을 확인하세요.
    if /I "%DDI_REQUIRE_PHASE3_DL%"=="1" (
        echo  DDI_REQUIRE_PHASE3_DL=1 이므로 설치 검증 실패를 hard fail 처리합니다.
        exit /b 1
    )
    if /I "%DDI_REQUIRE_CUDA_DL%"=="1" (
        echo  DDI_REQUIRE_CUDA_DL=1 이므로 설치 검증 실패를 hard fail 처리합니다.
        exit /b 1
    )
)
echo.
echo  다음 단계:
echo    1. DrugBank 파싱  : python scripts\parse_drugbank.py
echo    2. HANA 웹앱 실행 : cd hana_app 후 run.bat
echo    3. 브라우저 접속  : http://localhost:8501
echo ================================================
endlocal
