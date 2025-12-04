@echo off
setlocal

set HOST=127.0.0.1
set PORT=8000
set "PID="

rem --- Find PID using netstat ---
rem The for loop parses the output of 'netstat -a -n -o'
rem It filters for the listening port and extracts the 5th column (the PID)
for /f "tokens=5" %%a in ('netstat -a -n -o ^| findstr ":%PORT%" ^| findstr "LISTENING"') do (
    set "PID=%%a"
)

rem --- Terminate Process ---
if "%PID%" == "" (
    echo [INFO] No running server found on %HOST%:%PORT%.
    goto End
)

echo [INFO] Terminating process with PID: %PID% on port %PORT%.
taskkill /PID %PID% /T /F

rem --- Verify Termination ---
echo [INFO] Waiting for server to shut down...
timeout /t 2 /nobreak >nul

rem Use netstat again to verify. If findstr finds the line, the server is still running.
netstat -a -n -o | findstr ":%PORT%" | findstr "LISTENING" > NUL
if %ERRORLEVEL% equ 0 (
    echo [ERROR] Failed to stop the server! Port %PORT% is still in use. Manual check required.
) else (
    echo [SUCCESS] Server on %HOST%:%PORT% has been stopped successfully.
)

:End
endlocal