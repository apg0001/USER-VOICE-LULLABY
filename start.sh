#!/bin/bash

# 서버 설정 (여기서 변경)
HOST="127.0.0.1"
PORT="8000"
UVICORN_CMD="uvicorn app.main:app --host $HOST --port $PORT"

LOG_FILE="logs/app.log"

# 디렉토리 생성
mkdir -p logs

# 기존 uvicorn 프로세스 확인 후 중지
EXISTING_PIDS=$(pgrep -f "$UVICORN_CMD")
if [ ! -z "$EXISTING_PIDS" ]; then
    echo "uvicorn server detected (PID: $EXISTING_PIDS). stop and restart"
    pkill -f "$UVICORN_CMD"
    sleep 2
fi

# uvicorn 백그라운드 실행 + 로그 리다이렉트
nohup $UVICORN_CMD --reload > "$LOG_FILE" 2>&1 &
NEW_PID=$!

echo "FastAPI server started (PID: $NEW_PID, $HOST:$PORT, LOGS: $LOG_FILE)"
echo "Watch Log: tail -f $LOG_FILE"