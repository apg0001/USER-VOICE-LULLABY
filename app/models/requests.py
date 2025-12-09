"""요청 모델 정의"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class TrainRequest(BaseModel):
    """학습 요청 모델"""

    model_name: str = Field(..., description="로그 디렉토리에 저장될 모델 이름")
    dataset_path: str = Field(..., description="학습에 사용할 데이터셋 폴더 경로")
    sample_rate: Optional[int] = Field(
        None, ge=16000, le=48000, description="샘플레이트 (기본값 48kHz)"
    )
    total_epoch: Optional[int] = Field(
        None, ge=1, le=1000, description="총 학습 epoch (기본값 20)"
    )
    batch_size: Optional[int] = Field(
        None, ge=1, le=32, description="배치 사이즈 (기본값 8)"
    )


class InferenceRequest(BaseModel):
    """추론 요청 모델"""

    input_audio_path: str = Field(..., description="변환할 입력 오디오 경로")
    model_path: str = Field(..., description=".pth 모델 가중치 경로")
    index_path: Optional[str] = Field(
        None, description="선택적 .index 파일 경로 (없으면 자동으로 비활성화)"
    )
    output_dir: str = Field(
        "outputs", description="추론 결과를 저장할 디렉토리 (자동 생성)"
    )


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
