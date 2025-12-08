"""모델 관리 서비스"""
from __future__ import annotations

from ..logging_config import get_logger
from ..models.responses import ModelInfo
from ..repositories.model_repository import ModelRepository

logger = get_logger(__name__)


class ModelService:
    """모델 관리 서비스"""
    
    def __init__(self, repository: ModelRepository | None = None):
        self.repository = repository or ModelRepository()
        self._logger = logger
    
    def list_models(self) -> list[ModelInfo]:
        """모델 리스트 조회"""
        return self.repository.list_models()
    
    def delete_model(self, model_id: str) -> None:
        """모델 삭제"""
        self.repository.delete_model(model_id)

