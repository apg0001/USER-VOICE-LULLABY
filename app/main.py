"""FastAPI 애플리케이션 메인"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .dependencies import get_inference_queue, get_train_queue
from .logging_config import configure_logging, get_logger
from .routers import (
    health_router,
    inference_router,
    jobs_router,
    models_router,
    training_router,
)

# 로거 설정
logger = get_logger(__name__)

# FastAPI 앱 생성
app = FastAPI(
    title="Applio FastAPI Server",
    version="1.0.0",
    description="Minimal training & inference backend without GUI.",
)

# 정적 파일 디렉토리 설정
APP_DIR = Path(__file__).resolve().parent
PUBLIC_DIR = APP_DIR / "public"

if PUBLIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=PUBLIC_DIR), name="static")


# 라우터 등록
app.include_router(health_router)
app.include_router(training_router)
app.include_router(inference_router)
app.include_router(models_router)
app.include_router(jobs_router)


@app.on_event("startup")
async def _on_startup() -> None:
    """애플리케이션 시작 시 초기화"""
    configure_logging()
    train_queue = get_train_queue()
    inference_queue = get_inference_queue()
    await train_queue.start()
    await inference_queue.start()
    logger.info("Application started")


@app.on_event("shutdown")
async def _on_shutdown() -> None:
    """애플리케이션 종료 시 정리"""
    train_queue = get_train_queue()
    inference_queue = get_inference_queue()
    await train_queue.stop()
    await inference_queue.stop()
    logger.info("Application stopped")


@app.get("/ui", include_in_schema=False)
async def serve_ui():
    """UI 정적 페이지 제공"""
    index_path = PUBLIC_DIR / "index.html"
    if not index_path.exists():
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="정적 UI 파일을 찾을 수 없습니다.")
    return FileResponse(index_path)
