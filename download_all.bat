@echo off
chcp 65001 >nul
REM ============================================================
REM 전체 시스템 통합 패키지 다운로더 (인터넷 연결 환경에서 실행)
REM
REM packages_win + hana 두 폴더를 한 번에 다운로드합니다.
REM 다운로드 후 전체 폴더를 USB/이동식 미디어에 복사하여
REM 폐쇄망에서 install_all.bat 로 설치합니다.
REM
REM 사용법:
REM   download_all.bat              (전체 버전: 3.9~3.12)
REM   download_all.bat 311          (Python 3.11만)
REM ============================================================

setlocal EnableDelayedExpansion
set "PYTHONUTF8=1"

set PROJECT_ROOT=%~dp0
set PY_VERSIONS=39 310 311 312

if not "%1"=="" set PY_VERSIONS=%1

echo ================================================
echo  NHIS 다재약물 DDI 시스템 - 전체 패키지 다운로드
echo ================================================
echo  플랫폼    : Windows (win_amd64)
echo  Python    : %PY_VERSIONS%
echo  다운로드 폴더:
echo    - packages_win\pyXXX  (ETL / API / 공통)
echo    - hana\pyXXX          (HANA / ML / Streamlit)
echo.
echo  참고: 하위 download.bat 는 첫 번째 인자만 버전으로 받으므로
echo        여러 버전은 여기서 버전별로 순차 호출합니다.
echo.

for %%V in (%PY_VERSIONS%) do (
     echo.
     echo ================================================
     echo  Python %%V 패키지 다운로드
     echo ================================================

     REM ── packages_win 다운로드 ─────────────────────────────────
     echo ------------------------------------------------
     echo  [1/3] packages_win 패키지 다운로드 시작
     echo ------------------------------------------------
     call "%PROJECT_ROOT%packages_win\download.bat" %%V
     if errorlevel 1 (
         echo [오류] packages_win 패키지 다운로드 실패: Python %%V
         exit /b 1
     )

     REM ── hana 다운로드 ─────────────────────────────────────────
     echo.
     echo ------------------------------------------------
     echo  [2/3] hana 패키지 다운로드 시작
     echo ------------------------------------------------
     call "%PROJECT_ROOT%hana\download.bat" %%V
     if errorlevel 1 (
         echo [오류] hana 패키지 다운로드 실패: Python %%V
         exit /b 1
     )

     REM ── 결과분석 DOCX/그래프 필수 패키지 명시 보강 ───────────────
     echo.
     echo ------------------------------------------------
     echo  [3/3] DOCX/그래프 보고서 필수 패키지 보강 다운로드
     echo ------------------------------------------------
     set REPORT_PKG_DIR=%PROJECT_ROOT%packages_win\py%%V
     if not exist "!REPORT_PKG_DIR!" mkdir "!REPORT_PKG_DIR!"
     pip download ^
         --platform win_amd64 ^
         --python-version %%V ^
         --only-binary=:all: ^
         -d "!REPORT_PKG_DIR!" ^
         "python-docx==1.2.0" ^
         "lxml>=4.9.0,<6.0.0" ^
         "matplotlib>=3.7.0" ^
         "Pillow>=10.0.0"
     if errorlevel 1 (
         echo [오류] DOCX/그래프 보고서 필수 패키지 다운로드 실패: Python %%V
         exit /b 1
     )
)

echo.
echo ================================================
echo  다운로드 완료!
echo.
echo  폐쇄망 복사 방법:
echo    이 프로젝트 폴더 전체를 USB/이동식 미디어에 복사하거나
echo    아래 두 폴더만 복사해도 됩니다:
echo      packages_win\
echo      hana\
echo.
echo  폐쇄망 설치:
echo    install_all.bat %PY_VERSIONS% venv
echo ================================================
endlocal
