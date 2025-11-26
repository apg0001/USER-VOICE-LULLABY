@echo off
setlocal

set "ROOT_DIR=%~dp0"
set "RVC_DIR=%ROOT_DIR%applio"
set "PYTHON_BIN=%RVC_DIR%\env\python.exe"

if not exist "%PYTHON_BIN%" (
    echo Could not find %PYTHON_BIN%.
    echo Please run rvc\run-install.bat first to set up the environment,
    echo or install dependencies via pip install -r requirements.txt.
    exit /b 1
)

pushd "%ROOT_DIR%"
"%PYTHON_BIN%" -m app %*
popd

echo.
pause

