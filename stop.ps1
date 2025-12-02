# 서버 설정 (start.bat과 동일하게)
$HostName = "127.0.0.1"
$Port = 8000

Write-Host "--- Stopping server on $($HostName):$($Port) ---"

# 포트 점유 프로세스 PID 찾기
$Process = Get-NetTCPConnection -LocalPort $Port -State Listen | Select-Object -First 1

if ($Process) {
    $PID = $Process.OwningProcess
    if ($PID -eq 0) {
        Write-Host "[ERROR] Could not determine the process PID (PID is 0)." -ForegroundColor Red
    } else {
        Write-Host "[INFO] Terminating process with PID: $($PID) on port $($Port)." -ForegroundColor Yellow

        # 프로세스 종료
        # -Force: 강제 종료, -Recurse: 자식 프로세스까지 함께 종료
        try {
            Stop-Process -Id $PID -Force -ErrorAction Stop
            Write-Host "[INFO] Waiting for server to shut down..." -ForegroundColor Cyan
            Start-Sleep -Seconds 2

            # 종료 확인
            $StillRunning = Get-NetTCPConnection -LocalPort $Port -State Listen | Select-Object -First 1
            if ($StillRunning) {
                Write-Host "[ERROR] Failed to stop the server! Port $($Port) is still in use. Manual check required." -ForegroundColor Red
            } else {
                Write-Host "[SUCCESS] Server on $($HostName):$($Port) has been stopped successfully." -ForegroundColor Green
            }
        } catch {
            Write-Host "[ERROR] Failed to stop process $($PID): $($_.Exception.Message)" -ForegroundColor Red
        }
    }
} else {
    Write-Host "[INFO] No running server found on $($HostName):$($Port)." -ForegroundColor Yellow
}

Write-Host "--- Script finished ---"
