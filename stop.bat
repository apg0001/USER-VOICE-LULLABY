@echo off

REM 서버 설정 (start.bat와 동일)
set HOST=127.0.0.1
set PORT=8000

REM uvicorn 프로세스 정확히 검색해서 종료
for /f "tokens=2" %%i in ('tasklist /FI "IMAGENAME eq python.exe" /FO CSV ^| findstr /I "uvicorn app.main:app --host %HOST% --port %PORT%"') do (
    echo killing server... ^(PID: %%i, %HOST%:%PORT%^)
    taskkill /PID %%i /F
)

echo server killed ^(%HOST%:%PORT%^).
