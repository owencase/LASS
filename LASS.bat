@echo off
setlocal
cd /d "%~dp0"

if exist "LASS.exe" (
    start "" "%~dp0LASS.exe"
    exit /b
)

if not exist ".venv\Scripts\python.exe" (
    echo Preparing the LASS environment for the first run.
    python -m venv .venv
    if errorlevel 1 exit /b 1
)

set "RUNTIME_VERSION=2026-09-14-bigvox-v1"
set "INSTALLED_VERSION="
if exist ".venv\.lass-runtime-version" set /p INSTALLED_VERSION=<".venv\.lass-runtime-version"
if not "%INSTALLED_VERSION%"=="%RUNTIME_VERSION%" (
    echo Synchronizing the LASS runtime dependencies.
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install --no-cache-dir -r requirements.txt
    if errorlevel 1 exit /b 1
    >".venv\.lass-runtime-version" echo %RUNTIME_VERSION%
)

start "" ".venv\Scripts\pythonw.exe" -m src.desktop
endlocal
