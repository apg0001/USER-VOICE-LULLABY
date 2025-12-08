"""학습 서비스"""
from __future__ import annotations

from typing import Optional

from ..logging_config import get_logger
from ..models.requests import TrainFilesRequest

# services.py의 함수를 직접 import (순환 참조 방지를 위해 동적 import)
import importlib.util
from pathlib import Path

_services_module_path = Path(__file__).parent.parent / "services.py"
spec = importlib.util.spec_from_file_location("app_services_module", _services_module_path)
app_services_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app_services_module)
_train_model = app_services_module.train_model

logger = get_logger(__name__)


class TrainingService:
    """모델 학습 서비스"""
    
    def __init__(self):
        self._logger = logger
    
    async def train(
        self,
        model_id: str,
        dataset_path: str,
        sample_rate: Optional[int] = None,
        total_epoch: Optional[int] = None,
        batch_size: Optional[int] = None,
        embedder_model: Optional[str] = None,
        overtraining_detector: Optional[bool] = None,
        custom_pretrained: bool = False,
        g_pretrained_path: Optional[str] = None,
        d_pretrained_path: Optional[str] = None,
    ) -> dict:
        """모델 학습 실행"""
        return await _train_model(
            model_name=model_id,
            dataset_path=dataset_path,
            sample_rate=sample_rate,
            total_epoch=total_epoch,
            batch_size=batch_size,
            embedder_model=embedder_model,
            overtraining_detector=overtraining_detector,
            custom_pretrained=custom_pretrained,
            g_pretrained_path=g_pretrained_path,
            d_pretrained_path=d_pretrained_path,
        )
    
    async def train_from_request(self, request: TrainFilesRequest) -> dict:
        """요청 객체로부터 학습 실행"""
        return await self.train(
            model_id=request.model_id,
            dataset_path=request.dataset_path,
            sample_rate=request.sample_rate,
            total_epoch=request.total_epoch,
            batch_size=request.batch_size,
            embedder_model=request.embedder_model,
            overtraining_detector=request.overtraining_detector,
            custom_pretrained=request.custom_pretrained,
            g_pretrained_path=request.g_pretrained_path,
            d_pretrained_path=request.d_pretrained_path,
        )

