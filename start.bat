@echo off
chcp 65001 >nul

REM 서버 설정 (여기서 변경)
set HOST=127.0.0.1
set PORT=8000
set LOG_DIR=logs
set APP_LOG=%LOG_DIR%\app.log
set ERR_LOG=%LOG_DIR%\error.log

mkdir "%LOG_DIR%" 2>nul

REM 기존 uvicorn 프로세스 종료
for /f "tokens=2" %%i in ('tasklist /FI "IMAGENAME eq python.exe" /FO CSV ^| findstr /I "uvicorn app.main:app --host %HOST% --port %PORT%"') do (
    echo Killing uvicorn server... ^(PID: %%i^)
    taskkill /PID %%i /F >nul 2>&1
)
timeout /t 2 /nobreak >nul

REM 새 uvicorn 시작 (표준 출력은 콘솔로, 파일 로깅은 Python 로깅 설정에 맡김)
powershell -Command "Start-Process uvicorn -ArgumentList 'app.main:app','--host','%HOST%','--port','%PORT%','--reload' -WindowStyle Hidden" 
echo FastAPI server started ^(%HOST%:%PORT%^)
