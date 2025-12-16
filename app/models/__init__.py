"""도메인 모델 정의"""
from .requests import (
    TrainFilesRequest,
    InferenceFilesRequest,
)
from .responses import (
    HealthResponse,
    ResourceInfo,
    QueueStats,
    ModelInfo,
    OutputInfo,
    JobStatusResponse,
)

__all__ = [
    "TrainFilesRequest",
    "InferenceFilesRequest",
    "HealthResponse",
    "ResourceInfo",
    "QueueStats",
    "ModelInfo",
    "OutputInfo",
    "JobStatusResponse",
]

