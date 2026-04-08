@echo off
chcp 65001 > nul
cd /d "%~dp0"
if errorlevel 1 (
    echo [ERROR] 스크립트 폴더로 이동 실패: %~dp0
    pause & exit /b 1
)

REM --- 가상환경 존재 확인 ---
if not exist venv\Scripts\activate.bat (
    echo [ERROR] 가상환경이 없습니다.
    echo         먼저 setup.bat 를 실행하여 패키지를 설치하세요.
    pause & exit /b 1
)

REM --- 가상환경 활성화 ---
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] 가상환경 활성화 실패. setup.bat 를 다시 실행하세요.
    pause & exit /b 1
)

REM --- 앱 실행 ---
echo [INFO] NHIS YOD-DM Analyzer 시작...
python main_app.py
if errorlevel 1 (
    echo.
    echo [ERROR] 앱이 오류로 종료되었습니다.
    echo         로그 파일을 확인하세요: %%LOCALAPPDATA%%\NHIS_YOD_DM_Analyzer\logs\
    pause
)
