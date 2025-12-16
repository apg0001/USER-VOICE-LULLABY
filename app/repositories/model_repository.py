"""모델 리포지토리"""
from __future__ import annotations

import datetime
import json
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
            # G_와 D_로 시작하는 파일 제외 (추론에 사용하지 않음)
            pth_files = [f.name for f in model_dir.glob("*.pth") if not (f.name.startswith("G_") or f.name.startswith("D_"))]
            index_files = [f.name for f in model_dir.glob("*.index")]
            
            # .pth 파일이 있는 경우만 모델로 간주
            if pth_files:
                created_at = self._get_created_at(model_dir)
                
                # 모델 정보 JSON 파일 로드
                model_info_json = self._load_model_info(model_dir)
                
                # 절대 경로 생성
                pth_files_absolute = [str((model_dir / f).resolve()) for f in sorted(pth_files)]
                index_files_absolute = [str((model_dir / f).resolve()) for f in sorted(index_files)]
                
                models.append(
                    ModelInfo(
                        model_id=model_id,
                        model_files=sorted(pth_files),
                        index_files=sorted(index_files),
                        created_at=created_at,
                        # UI에서 표시하지 않는 필드들 (주석 처리 - 나중에 쉽게 복구 가능)
                        # model_name=model_info_json.get("model_name") if model_info_json else None,
                        # embedder_model=model_info_json.get("embedder_model") if model_info_json else None,
                        # sample_rate=model_info_json.get("sample_rate") if model_info_json else None,
                        # total_epoch=model_info_json.get("total_epoch") if model_info_json else None,
                        # vocoder=model_info_json.get("vocoder") if model_info_json else None,
                        model_name=None,  # 주석 처리됨
                        embedder_model=None,  # 주석 처리됨
                        sample_rate=None,  # 주석 처리됨
                        total_epoch=None,  # 주석 처리됨
                        vocoder=None,  # 주석 처리됨
                        model_description=model_info_json.get("model_description") if model_info_json else None,
                        model_files_absolute=pth_files_absolute,
                        index_files_absolute=index_files_absolute,
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
            self._logger.debug(f"모델 삭제 완료: {model_id}")
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
    
    @staticmethod
    def _load_model_info(model_dir: Path) -> Optional[dict]:
        """모델 정보 JSON 파일 로드"""
        model_info_path = model_dir / "model_info.json"
        if not model_info_path.exists():
            logger.debug(f"모델 정보 JSON 파일 없음: {model_info_path}")
            return None
        
        try:
            with open(model_info_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                logger.debug(f"모델 정보 JSON 파일 로드 성공: {model_info_path} | keys={list(data.keys()) if data else 'empty'}")
                return data
        except json.JSONDecodeError as e:
            logger.error(f"모델 정보 JSON 파일 파싱 실패: {model_info_path} - {e}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"모델 정보 JSON 파일 로드 실패: {model_info_path} - {e}", exc_info=True)
            return None

