#!/bin/bash

# 서버 설정 (start.sh와 동일하게)
HOST="127.0.0.1"
PORT="8000"
TARGET_CMD="uvicorn app.main:app --host $HOST --port $PORT"

# 정확히 일치하는 프로세스만 종료
PIDS=$(pgrep -f "$TARGET_CMD")
if [ ! -z "$PIDS" ]; then
    echo "Killing Server... (PID: $PIDS, $HOST:$PORT)"
    pkill -f "$TARGET_CMD"
    sleep 1
    
    # 아직 살아있는 프로세스 강제 종료
    REMAINING_PIDS=$(pgrep -f "$TARGET_CMD")
    if [ ! -z "$REMAINING_PIDS" ]; then
        echo "Force Stop... (PID: $REMAINING_PIDS)"
        pkill -9 -f "$TARGET_CMD"
    fi
    
    echo "Server Killed ($HOST:$PORT)."
else
    echo "No Server ($HOST:$PORT)."
fi
