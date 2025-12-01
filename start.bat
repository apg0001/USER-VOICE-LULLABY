@echo off

REM 서버 설정 (여기서 변경)
set HOST=127.0.0.1
set PORT=8000

mkdir logs 2>nul

REM 기존 uvicorn 프로세스 종료
for /f "tokens=2" %%i in ('tasklist /FI "IMAGENAME eq python.exe" /FO CSV ^| findstr /I "uvicorn app.main:app --host %HOST% --port %PORT%"') do (
    echo Killing uvicorn server... ^(PID: %%i^)
    taskkill /PID %%i /F
)
timeout /t 2 /nobreak >nul

REM 새 uvicorn 시작
powershell -Command "Start-Process uvicorn -ArgumentList 'app.main:app','--host','%HOST%','--port','%PORT%','--reload' -WindowStyle Hidden -RedirectStandardOutput 'logs\\app.log' -RedirectStandardError 'logs\\error.log'"
echo FastAPI server started ^(%HOST%:%PORT%, LOGS: logs\app.log^)
