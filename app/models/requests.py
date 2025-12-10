"""요청 모델 정의"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class TrainFilesRequest(BaseModel):
    """파일 업로드 학습 요청 모델 (내부 사용)"""

    model_id: str
    dataset_path: str
    sample_rate: int
    total_epoch: int
    batch_size: int
    embedder_model: Optional[str] = None
    vocoder: Optional[str] = None
    overtraining_detector: Optional[bool] = None
    custom_pretrained: bool = False
    g_pretrained_path: Optional[str] = None
    d_pretrained_path: Optional[str] = None


class InferenceFilesRequest(BaseModel):
    """파일 업로드 추론 요청 모델 (내부 사용)"""

    input_audio_path: str
    model_path: str
    index_path: Optional[str] = None
    output_dir: str = "outputs"
    volume_envelope: Optional[float] = None
    protect: Optional[float] = None
    f0_autotune: Optional[bool] = None
    f0_autotune_strength: Optional[float] = None
    embedder_model: Optional[str] = None
    index_rate: Optional[float] = None
