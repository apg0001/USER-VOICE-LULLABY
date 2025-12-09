"""파일 관리 리포지토리"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional
from uuid import uuid4

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import UploadFile

from app.constants import RVC_ROOT
from app.logging_config import PROJECT_ROOT, get_logger

logger = get_logger(__name__)

DATASET_ROOT = RVC_ROOT / "datasets"
AUDIO_ROOT = DATASET_ROOT / "target_audio"
ALLOWED_ROOT = RVC_ROOT / "outputs"


class FileRepository:
    """파일 관리 리포지토리"""
    
    def __init__(
        self,
        dataset_root: Path | None = None,
        audio_root: Path | None = None,
        allowed_root: Path | None = None,
    ):
        self.dataset_root = dataset_root or DATASET_ROOT
        self.audio_root = audio_root or AUDIO_ROOT
        self.allowed_root = allowed_root or ALLOWED_ROOT
        self._logger = logger
    
    def save_training_files(self, model_id: str, files: list) -> Path:
        """학습용 파일 저장"""
        dataset_path = self.dataset_root / model_id
        os.makedirs(dataset_path, exist_ok=True)
        self._logger.info(f"데이터셋 저장 경로 생성: {dataset_path}")
        
        for idx, file in enumerate(files):
            ext = file.filename.split(".")[-1] if file.filename else "wav"
            file_path = dataset_path / f"audio_{idx+1:03d}.{ext}"
            with open(file_path, "wb") as f:
                content = file.file.read()
                f.write(content)
            self._logger.info(f"파일 저장: {file_path} - {file.filename}")
        
        return dataset_path
    
    def save_inference_audio(self, audio_file) -> Path:
        """추론용 오디오 파일 저장"""
        os.makedirs(self.audio_root, exist_ok=True)
        self._logger.info(f"타깃 오디오 저장 경로 생성: {self.audio_root}")
        
        ext = getattr(audio_file, 'filename', '').split(".")[-1] if getattr(audio_file, 'filename', None) else "wav"
        temp_audio_path = self.audio_root / f"temp_inference_{uuid4().hex}.{ext}"
        
        with open(temp_audio_path, "wb") as f:
            content = audio_file.file.read()
            f.write(content)
        
        self._logger.info(f"임시 오디오 파일 저장 완료: {temp_audio_path}")
        return temp_audio_path
    
    def is_allowed_path(self, path: str) -> bool:
        """허용된 경로인지 확인"""
        requested_path = (self.allowed_root / path).resolve()
        return requested_path.is_relative_to(self.allowed_root)
    
    def get_file_path(self, path: str) -> Path:
        """파일 경로 반환"""
        requested_path = (self.allowed_root / path).resolve()
        if not requested_path.is_file():
            raise FileNotFoundError(f"File not found: {requested_path}")
        if not self.is_allowed_path(path):
            raise ValueError(f"Not allowed path: {path}")
        return requested_path

