"""모델 리포지토리"""
from __future__ import annotations

import datetime
import shutil
from pathlib import Path
from typing import Optional

from ..logging_config import get_logger
from ..models.responses import ModelInfo

from ..constants import RVC_LOGS_DIR

logger = get_logger(__name__)

# 보호된 디렉토리 목록
PROTECTED_DIRS = {"mute", "mute_spin", "mute_spin-v2", "reference", "zips", "test"}


class ModelRepository:
    """모델 관리 리포지토리"""
    
    def __init__(self, logs_dir: Path | None = None):
        self.logs_dir = logs_dir or RVC_LOGS_DIR
        self._logger = logger
    
    def list_models(self) -> list[ModelInfo]:
        """모델 리스트 조회"""
        if not self.logs_dir.exists():
            return []
        
        models = []
        for model_dir in self.logs_dir.iterdir():
            if not model_dir.is_dir():
                continue
            
            # 보호된 디렉토리 제외
            if model_dir.name in PROTECTED_DIRS:
                continue
            
            model_id = model_dir.name
            pth_files = [f.name for f in model_dir.glob("*.pth")]
            index_files = [f.name for f in model_dir.glob("*.index")]
            
            # .pth 파일이 있는 경우만 모델로 간주
            if pth_files:
                created_at = self._get_created_at(model_dir)
                models.append(
                    ModelInfo(
                        model_id=model_id,
                        model_files=sorted(pth_files),
                        index_files=sorted(index_files),
                        created_at=created_at,
                    )
                )
        
        return sorted(models, key=lambda x: x.created_at or "", reverse=True)
    
    def get_model_dir(self, model_id: str) -> Path:
        """모델 디렉토리 경로 반환"""
        return self.logs_dir / model_id
    
    def model_exists(self, model_id: str) -> bool:
        """모델 존재 여부 확인"""
        model_dir = self.get_model_dir(model_id)
        return model_dir.exists() and model_dir.is_dir()
    
    def is_protected(self, model_id: str) -> bool:
        """보호된 모델인지 확인"""
        return model_id in PROTECTED_DIRS
    
    def delete_model(self, model_id: str) -> None:
        """모델 삭제"""
        if self.is_protected(model_id):
            raise ValueError(f"이 모델은 삭제할 수 없습니다: {model_id}")
        
        model_dir = self.get_model_dir(model_id)
        if not model_dir.exists() or not model_dir.is_dir():
            raise FileNotFoundError(f"모델을 찾을 수 없습니다: {model_id}")
        
        try:
            shutil.rmtree(model_dir)
            self._logger.info(f"모델 삭제 완료: {model_id}")
        except Exception as e:
            self._logger.exception(f"모델 삭제 실패: {model_id}")
            raise RuntimeError(f"모델 삭제 중 오류 발생: {str(e)}") from e
    
    @staticmethod
    def _get_created_at(model_dir: Path) -> Optional[str]:
        """모델 생성 시간 조회"""
        try:
            return datetime.datetime.fromtimestamp(
                model_dir.stat().st_mtime
            ).isoformat()
        except Exception:
            return None

