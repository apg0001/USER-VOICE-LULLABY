"""상수 정의"""
from __future__ import annotations

from pathlib import Path

from .logging_config import PROJECT_ROOT

RVC_ROOT = PROJECT_ROOT / "applio"
RVC_LOGS_DIR = RVC_ROOT / "logs"  # 모델 저장 폴더
DEFAULT_OUTPUT_DIR = RVC_ROOT / "outputs"  # 출력 파일 기본 경로

