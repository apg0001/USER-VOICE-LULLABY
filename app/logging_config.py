from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

# 공통 로그 디렉토리/파일 경로
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE_PATH = LOGS_DIR / "app.log"
ERROR_FILE_PATH = LOGS_DIR / "error.log"

_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def _build_rotating_handler(
    path: Path,
    *,
    level: int = logging.INFO,
    max_bytes: int = 100 * 1024 * 1024,
    backups: int = 10,
) -> RotatingFileHandler:
    handler = RotatingFileHandler(
        filename=str(path),
        maxBytes=max_bytes,
        backupCount=backups,
        encoding="utf-8-sig",  # Windows 메모장에서 자동으로 UTF-8 인식하도록 BOM 포함
    )
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    return handler


def configure_logging() -> None:
    """루트 로거에 콘솔 + 파일 핸들러를 구성한다.
    
    모든 로그(INFO, WARNING, ERROR 포함)는 app.log에 저장됩니다.
    ERROR 로그는 추가로 error.log에도 저장됩니다.
    """
    if getattr(configure_logging, "_configured", False):
        return

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(_LOG_FORMAT))

    # app.log 핸들러: 모든 로그(INFO 이상) 저장
    app_handler = _build_rotating_handler(LOG_FILE_PATH, level=logging.INFO)
    
    # error.log 핸들러: ERROR 레벨만 추가로 저장
    error_handler = _build_rotating_handler(
        ERROR_FILE_PATH, level=logging.ERROR, max_bytes=20 * 1024 * 1024, backups=5
    )

    # uvicorn 로거도 app.log에 저장되도록 설정
    # uvicorn의 기본 핸들러를 제거하고 루트 로거로만 전파되도록 설정
    uvicorn_logger = logging.getLogger("uvicorn")
    uvicorn_logger.setLevel(logging.INFO)
    uvicorn_logger.propagate = True  # 루트 로거로 전파
    # uvicorn이 시작되기 전이므로 핸들러가 없을 수 있음
    if uvicorn_logger.handlers:
        uvicorn_logger.handlers = []  # 기본 핸들러 제거
    
    uvicorn_access_logger = logging.getLogger("uvicorn.access")
    uvicorn_access_logger.setLevel(logging.INFO)
    uvicorn_access_logger.propagate = True  # 루트 로거로 전파
    if uvicorn_access_logger.handlers:
        uvicorn_access_logger.handlers = []  # 기본 핸들러 제거

    # 루트 로거에 핸들러 추가
    # 모든 로그(INFO, WARNING, ERROR)는 app.log에 저장됨
    root_logger.addHandler(console_handler)
    root_logger.addHandler(app_handler)
    root_logger.addHandler(error_handler)  # ERROR는 추가로 error.log에도 저장

    configure_logging._configured = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    logger = logging.getLogger(name)
    logger.propagate = True
    return logger


__all__ = [
    "PROJECT_ROOT",
    "LOGS_DIR",
    "LOG_FILE_PATH",
    "ERROR_FILE_PATH",
    "configure_logging",
    "get_logger",
]
