@echo off
chcp 65001 >nul
REM ============================================================
REM SAP HANA 연결 + ML 패키지 Windows 오프라인 다운로더
REM 인터넷 연결 환경(Windows)에서 실행
REM
REM 사용법:
REM   hana\download.bat              (전체 버전: 3.9~3.12)
REM   hana\download.bat 311          (Python 3.11만)
REM
REM 다운로드 후 폐쇄망에서 install.bat 실행
REM ============================================================

setlocal EnableDelayedExpansion
set "PYTHONUTF8=1"

set SCRIPT_DIR=%~dp0
set REQUIREMENTS=%SCRIPT_DIR%requirements.txt

REM Python 버전 목록
set PY_VERSIONS=39 310 311 312

REM 특정 버전 지정 시
if not "%1"=="" (
    set PY_VERSIONS=%1
)

echo ==============================================
echo SAP HANA 패키지 Windows 오프라인 다운로드
echo ==============================================
echo 플랫폼: Windows (win_amd64)
echo Python 버전: %PY_VERSIONS%
echo.

for %%V in (%PY_VERSIONS%) do (
    echo ----------------------------------------------
    echo Python %%V 패키지 다운로드
    echo ----------------------------------------------

    set OUT_DIR=%SCRIPT_DIR%py%%V
    if not exist "!OUT_DIR!" mkdir "!OUT_DIR!"

    echo [1단계] 컴파일된 Windows 바이너리 다운로드...
    pip download ^
        --platform win_amd64 ^
        --python-version %%V ^
        --only-binary=:all: ^
        -d "!OUT_DIR!" ^
        -r "%REQUIREMENTS%"

    echo [2단계] 순수 Python 패키지 ^(바이너리 없는 경우 폴백^)...
    pip download ^
        --prefer-binary ^
        -d "!OUT_DIR!" ^
        -r "%REQUIREMENTS%"

    echo [3단계] 빌드 도구 및 누락 패키지 보완 다운로드...
    REM numpy 1.x wheel 이 남아있으면 삭제 (shap>=0.51 은 numpy>=2 필요)
    del /q "!OUT_DIR!\numpy-1*.whl" 2>nul
    pip download ^
        --platform win_amd64 ^
        --python-version %%V ^
        --only-binary=:all: ^
        -d "!OUT_DIR!" ^
        "numpy>=2.0.0"

    pip download ^
        --prefer-binary ^
        -d "!OUT_DIR!" ^
        "setuptools>=65.0.0" ^
        "wheel>=0.40.0" ^
        "openpyxl>=3.1.0" ^
        "keyring>=24.0.0"

    echo 완료: !OUT_DIR!
    echo.
)

echo ==============================================
echo 다운로드 완료!
echo.
echo 전체 시스템(DDI+HANA) 다운로드:
echo   download_all.bat
echo.
echo 폐쇄망 설치:
echo   install_all.bat %PY_VERSIONS% venv    (전체 통합)
echo   hana\install.bat %PY_VERSIONS% venv   (HANA/ML만)
echo ==============================================
endlocal
