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
        try:
            dataset_path = self.dataset_root / model_id
            os.makedirs(dataset_path, exist_ok=True)
            self._logger.debug(f"데이터셋 저장 경로 생성: {dataset_path}")
        except OSError as e:
            self._logger.error(f"데이터셋 디렉토리 생성 실패 | model_id={model_id} | path={dataset_path} | error={e}", exc_info=True)
            raise RuntimeError(f"데이터셋 디렉토리 생성 실패: {str(e)}")
        
        saved_files = []
        for idx, file in enumerate(files):
            try:
                if not file.filename:
                    self._logger.warning(f"파일명이 없는 파일 건너뜀 | index={idx}")
                    continue
                
                ext = file.filename.split(".")[-1] if "." in file.filename else "wav"
                file_path = dataset_path / f"audio_{idx+1:03d}.{ext}"
                
                # 파일 읽기
                try:
                    content = file.file.read()
                    if not content:
                        self._logger.warning(f"빈 파일 건너뜀 | file={file.filename}")
                        continue
                except Exception as e:
                    self._logger.error(f"파일 읽기 실패 | file={file.filename} | error={e}", exc_info=True)
                    raise RuntimeError(f"파일 읽기 실패 ({file.filename}): {str(e)}")
                
                # 파일 쓰기
                try:
                    with open(file_path, "wb") as f:
                        f.write(content)
                    saved_files.append(file_path)
                    self._logger.debug(f"파일 저장 완료 | file_path={file_path} | filename={file.filename} | size={len(content)} bytes")
                except OSError as e:
                    self._logger.error(f"파일 저장 실패 | file_path={file_path} | filename={file.filename} | error={e}", exc_info=True)
                    raise RuntimeError(f"파일 저장 실패 ({file.filename}): {str(e)}")
            except Exception as e:
                self._logger.error(f"파일 처리 중 오류 | index={idx} | filename={file.filename if file else 'unknown'} | error={e}", exc_info=True)
                raise
        
        if not saved_files:
            self._logger.error(f"저장된 파일이 없음 | model_id={model_id}")
            raise ValueError("저장된 파일이 없습니다. 유효한 파일을 업로드해주세요.")
        
        self._logger.debug(f"학습 파일 저장 완료 | model_id={model_id} | saved_count={len(saved_files)} | dataset_path={dataset_path}")
        return dataset_path
    
    def save_inference_audio(self, audio_file) -> Path:
        """추론용 오디오 파일 저장"""
        try:
            os.makedirs(self.audio_root, exist_ok=True)
            self._logger.debug(f"타깃 오디오 저장 경로 생성: {self.audio_root}")
        except OSError as e:
            self._logger.error(f"오디오 디렉토리 생성 실패 | path={self.audio_root} | error={e}", exc_info=True)
            raise RuntimeError(f"오디오 디렉토리 생성 실패: {str(e)}")
        
        try:
            filename = getattr(audio_file, 'filename', None)
            ext = filename.split(".")[-1] if filename and "." in filename else "wav"
            temp_audio_path = self.audio_root / f"temp_inference_{uuid4().hex}.{ext}"
            
            # 파일 읽기
            try:
                content = audio_file.file.read()
                if not content:
                    self._logger.error("빈 오디오 파일 업로드됨")
                    raise ValueError("빈 오디오 파일입니다")
            except Exception as e:
                self._logger.error(f"오디오 파일 읽기 실패 | filename={filename} | error={e}", exc_info=True)
                raise RuntimeError(f"오디오 파일 읽기 실패: {str(e)}")
            
            # 파일 쓰기
            try:
                with open(temp_audio_path, "wb") as f:
                    f.write(content)
                self._logger.debug(f"임시 오디오 파일 저장 완료 | path={temp_audio_path} | filename={filename} | size={len(content)} bytes")
            except OSError as e:
                self._logger.error(f"오디오 파일 저장 실패 | path={temp_audio_path} | error={e}", exc_info=True)
                raise RuntimeError(f"오디오 파일 저장 실패: {str(e)}")
            
            return temp_audio_path
        except Exception as e:
            self._logger.error(f"추론 오디오 파일 저장 중 오류 | error={e}", exc_info=True)
            raise
    
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

