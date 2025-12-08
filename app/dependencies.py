"""의존성 주입 설정"""
from __future__ import annotations

from functools import lru_cache

from .monitors.resource_monitor import ResourceMonitor
from .repositories.file_repository import FileRepository
from .repositories.model_repository import ModelRepository
from .repositories.output_repository import OutputRepository
from .services.inference_service import InferenceService
from .services.model_service import ModelService
from .services.output_service import OutputService
from .services.training_service import TrainingService
from .task_queue import AsyncJobQueue


@lru_cache()
def get_resource_monitor() -> ResourceMonitor:
    """리소스 모니터 싱글톤"""
    return ResourceMonitor()


@lru_cache()
def get_model_repository() -> ModelRepository:
    """모델 리포지토리 싱글톤"""
    return ModelRepository()


@lru_cache()
def get_file_repository() -> FileRepository:
    """파일 리포지토리 싱글톤"""
    return FileRepository()


@lru_cache()
def get_training_service() -> TrainingService:
    """학습 서비스 싱글톤"""
    return TrainingService()


@lru_cache()
def get_inference_service() -> InferenceService:
    """추론 서비스 싱글톤"""
    return InferenceService()


@lru_cache()
def get_model_service() -> ModelService:
    """모델 서비스 싱글톤"""
    return ModelService(repository=get_model_repository())


@lru_cache()
def get_output_repository() -> OutputRepository:
    """출력 리포지토리 싱글톤"""
    return OutputRepository()


@lru_cache()
def get_output_service() -> OutputService:
    """출력 서비스 싱글톤"""
    return OutputService(repository=get_output_repository())


# 작업 큐는 전역으로 관리
_train_queue: AsyncJobQueue | None = None
_inference_queue: AsyncJobQueue | None = None


def get_train_queue() -> AsyncJobQueue:
    """학습 큐 싱글톤"""
    global _train_queue
    if _train_queue is None:
        _train_queue = AsyncJobQueue("train")
    return _train_queue


def get_inference_queue() -> AsyncJobQueue:
    """추론 큐 싱글톤"""
    global _inference_queue
    if _inference_queue is None:
        _inference_queue = AsyncJobQueue("inference")
    return _inference_queue

