"""FastAPI 애플리케이션 메인"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response as StarletteResponse

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


class NoCacheStaticFiles(StaticFiles):
    """캐시 방지 헤더를 추가하는 정적 파일 서빙 클래스"""

    async def __call__(self, scope, receive, send):
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                # 정적 파일 응답에 캐시 방지 헤더 추가
                headers = dict(message.get("headers", []))
                headers[b"cache-control"] = b"no-cache, no-store, must-revalidate"
                headers[b"pragma"] = b"no-cache"
                headers[b"expires"] = b"0"
                message["headers"] = list(headers.items())
            await send(message)

        await super().__call__(scope, receive, send_wrapper)


if PUBLIC_DIR.exists():
    app.mount("/static", NoCacheStaticFiles(directory=PUBLIC_DIR), name="static")


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
    """UI 정적 페이지 제공 (캐시 방지)"""
    index_path = PUBLIC_DIR / "index.html"
    if not index_path.exists():
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="정적 UI 파일을 찾을 수 없습니다.")
    # 캐시 방지 헤더 추가
    return FileResponse(
        index_path,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )
