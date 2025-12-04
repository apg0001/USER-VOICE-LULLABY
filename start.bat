@echo off
chcp 65001>nul
set HOST=127.0.0.1
set PORT=8000
set LOG_DIR=logs
set APP_LOG=%LOG_DIR%\debug.log

REM 로그 디렉토리 생성
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

REM 기존 uvicorn 종료
taskkill /IM python.exe /F /FI "WINDOWTITLE eq *uvicorn*" 2>nul
timeout /t 2 /nobreak >nul

REM 백그라운드 실행 + 로그 파일로 리다이렉트
start /B "" cmd /C "uvicorn app.main:app --host %HOST% --port %PORT% --reload > "%APP_LOG%" 2>&1"

echo FastAPI 백그라운드 실행됨: http://%HOST%:%PORT%
echo 로그: %APP_LOG%
pause
