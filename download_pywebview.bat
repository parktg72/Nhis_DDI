@echo off
chcp 65001 >nul
REM Download pywebview for Python 3.12 (Windows, offline install)
REM Run this script on a machine WITH internet access
REM Then copy the packages_win\py312 folder to the offline machine

setlocal
set "PYTHONUTF8=1"

set ROOT=%~dp0
set PKG_DIR=%ROOT%packages_win\py312

if not exist "%PKG_DIR%" mkdir "%PKG_DIR%"

echo Downloading pywebview and dependencies to %PKG_DIR%...
echo.

python -m pip download pywebview proxy_tools --dest "%PKG_DIR%" --platform win_amd64 --python-version 3.12 --only-binary=:all:

if errorlevel 1 (
    echo.
    echo [WARN] Binary-only download failed. Trying without platform filter...
    python -m pip download pywebview proxy_tools --dest "%PKG_DIR%"
)

echo.
echo Done. Files saved to: %PKG_DIR%
echo.
echo Next steps:
echo   1. Copy packages_win\py312 folder to the offline machine
echo   2. Run install_pywebview.bat on the offline machine

endlocal
pause
