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
    path: Path, *, level: int = logging.INFO, max_bytes: int = 100 * 1024 * 1024, backups: int = 10
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
    """루트 로거에 콘솔 + 파일 핸들러를 구성한다."""
    if getattr(configure_logging, "_configured", False):
        return

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(_LOG_FORMAT))

    app_handler = _build_rotating_handler(LOG_FILE_PATH)
    error_handler = _build_rotating_handler(
        ERROR_FILE_PATH, level=logging.ERROR, max_bytes=20 * 1024 * 1024, backups=5
    )

    # 기존 uvicorn/표준 핸들러는 유지하고, 추가로 파일/콘솔 핸들러만 붙인다.
    # 이렇게 하면 `uvicorn app.main:app` 로 직접 실행해도
    # uvicorn 기본 콘솔 출력 + 우리의 로그 출력이 모두 터미널에 표시된다.
    root_logger.addHandler(console_handler)
    root_logger.addHandler(app_handler)
    root_logger.addHandler(error_handler)

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

