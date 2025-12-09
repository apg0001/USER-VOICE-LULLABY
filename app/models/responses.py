"""응답 모델 정의"""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class QueueStats(BaseModel):
    """큐 통계 모델"""
    name: str
    running: int  # 실행 중 작업 수
    pending: int  # 대기 중 작업 수


class HealthResponse(BaseModel):
    """헬스 체크 응답 모델"""
    status: str
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    queues: dict[str, QueueStats]
    gpus: list[dict] | None = None
    resource_available: bool = Field(..., description="새 작업을 받을 수 있는지 여부")


class ResourceInfo(BaseModel):
    """리소스 정보 모델"""
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    gpu_utilization: list[float] | None = None
    gpu_memory_used: list[float] | None = None
    can_accept_job: bool = Field(..., description="새 작업을 받을 수 있는지 여부")


class ModelInfo(BaseModel):
    """모델 정보 모델"""
    model_id: str = Field(..., description="모델 ID (UUID)")
    model_files: list[str] = Field(..., description=".pth 모델 파일 목록")
    index_files: list[str] = Field(..., description=".index 파일 목록")
    created_at: Optional[str] = Field(None, description="생성 시간")


class OutputInfo(BaseModel):
    """추론 결과 정보 모델"""
    output_id: str = Field(..., description="출력 파일 ID (파일명)")
    file_path: str = Field(..., description="파일 경로 (상대 경로)")
    file_size: int = Field(..., description="파일 크기 (bytes)")
    created_at: Optional[str] = Field(None, description="생성 시간")


class JobStatusResponse(BaseModel):
    """작업 상태 응답 모델"""
    job_id: str
    status: str
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    result: dict | None = None
    error: Optional[str] = None
    progress: Optional[dict] = Field(None, description="작업 진행률 정보 (학습 작업의 경우)")
