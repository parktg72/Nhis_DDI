@echo off
chcp 65001 >nul
REM ============================================================
REM Windows 오프라인 패키지 다운로더
REM 인터넷 연결 환경(Windows)에서 실행
REM
REM 사용법:
REM   packages_win\download.bat
REM   packages_win\download.bat 310     (Python 3.10만)
REM ============================================================

setlocal EnableDelayedExpansion

set SCRIPT_DIR=%~dp0
set REQUIREMENTS=%SCRIPT_DIR%requirements.txt

REM Python 버전 목록
set PY_VERSIONS=39 310 311 312

REM 특정 버전 지정 시
if not "%1"=="" (
    set PY_VERSIONS=%1
)

echo ==============================================
echo Windows 오프라인 패키지 다운로드 시작
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

    REM 컴파일된 Windows 패키지
    pip download ^
        --platform win_amd64 ^
        --python-version %%V ^
        --only-binary=:all: ^
        --no-deps ^
        -d "!OUT_DIR!" ^
        -r "%REQUIREMENTS%" 2>nul

    REM 순수 Python 패키지 (의존성 포함)
    pip download ^
        --prefer-binary ^
        -d "!OUT_DIR!" ^
        -r "%REQUIREMENTS%" 2>&1 | findstr /C:"Saved" /C:"already"

    REM 빌드 도구 및 누락 패키지 보완
    pip download ^
        --prefer-binary ^
        -d "!OUT_DIR!" ^
        "setuptools>=65.0.0" ^
        "wheel>=0.40.0" ^
        "openpyxl>=3.1.0"

    echo 완료: !OUT_DIR!
    echo.
)

REM PySpark 별도
echo ----------------------------------------------
echo PySpark + Delta Lake 다운로드
echo ----------------------------------------------
set SPARK_DIR=%SCRIPT_DIR%spark
if not exist "%SPARK_DIR%" mkdir "%SPARK_DIR%"
pip download --prefer-binary -d "%SPARK_DIR%" "pyspark>=3.5.0" "delta-spark>=3.0.0"
echo    → %SPARK_DIR%

echo.
echo ==============================================
echo 다운로드 완료!
echo.
echo 전체 시스템(HANA 포함) 다운로드:
echo   download_all.bat
echo.
echo 폐쇄망 설치:
echo   install_all.bat %PY_VERSIONS% venv    (전체 통합)
echo   packages_win\install.bat %PY_VERSIONS% venv  (ETL/API만)
echo ==============================================
endlocal
