"""학습 라우터"""

from typing import List, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from ..dependencies import (
    get_file_repository,
    get_resource_monitor,
    get_train_queue,
    get_training_service,
)
from ..models.requests import TrainRequest, TrainFilesRequest
from ..monitors.resource_monitor import ResourceMonitor
from ..repositories.file_repository import FileRepository
from ..services.training_service import TrainingService
from ..settings import TRAINING_DEFAULTS
from ..task_queue import AsyncJobQueue
from ..logging_config import get_logger

router = APIRouter()
logger = get_logger(__name__)


def _check_and_raise_if_resources_unavailable(monitor: ResourceMonitor) -> None:
    """리소스 가용성 확인 및 예외 발생"""
    status = monitor.get_resource_status()
    if not status.can_accept_job:
        raise HTTPException(
            status_code=503,
            detail="시스템 리소스가 부족하여 새 작업을 받을 수 없습니다. 잠시 후 다시 시도해주세요.",
        )


@router.post("/train")
async def start_training(
    payload: TrainRequest,
    training_service: TrainingService = Depends(get_training_service),
    train_queue: AsyncJobQueue = Depends(get_train_queue),
):
    """학습 시작 요청 처리 (기존 API)"""
    try:
        result = await train_queue.enqueue(
            training_service.train,
            model_id=payload.model_name,
            dataset_path=payload.dataset_path,
            sample_rate=payload.sample_rate,
            total_epoch=payload.total_epoch,
            batch_size=payload.batch_size,
        )
        return {"status": "running", **result}
    except FileNotFoundError as exc:
        logger.error("Train request failed: %s", exc)
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception("Unexpected training error")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/train-files")
async def start_training_files(
    sample_rate: int = Form(TRAINING_DEFAULTS.sample_rate),
    total_epoch: int = Form(TRAINING_DEFAULTS.total_epoch),
    batch_size: int = Form(TRAINING_DEFAULTS.batch_size),
    files: List[UploadFile] = File(...),
    embedder_model: Optional[str] = Form(TRAINING_DEFAULTS.embedder_model),
    vocoder: Optional[str] = Form(TRAINING_DEFAULTS.vocoder),
    overtraining_detector: Optional[bool] = Form(
        TRAINING_DEFAULTS.overtraining_detector
    ),
    custom_pretrained: Optional[bool] = Form(TRAINING_DEFAULTS.custom_pretrained),
    g_pretrained_path: Optional[str] = Form(TRAINING_DEFAULTS.g_pretrained_path),
    d_pretrained_path: Optional[str] = Form(TRAINING_DEFAULTS.d_pretrained_path),
    training_service: TrainingService = Depends(get_training_service),
    train_queue: AsyncJobQueue = Depends(get_train_queue),
    file_repo: FileRepository = Depends(get_file_repository),
    monitor: ResourceMonitor = Depends(get_resource_monitor),
):
    """파일 업로드로 학습 요청 처리"""
    # 모델 ID 자동 생성
    model_id = str(uuid4())

    logger.info(f"파일 업로드 학습 요청 - 모델 ID: {model_id}")

    try:
        # 파일 저장
        dataset_path = file_repo.save_training_files(model_id, files)

        # 리소스 확인
        # _check_and_raise_if_resources_unavailable(monitor)

        # 비동기 작업 등록
        request = TrainFilesRequest(
            model_id=model_id,
            dataset_path=str(dataset_path),
            sample_rate=sample_rate,
            total_epoch=total_epoch,
            batch_size=batch_size,
            embedder_model=embedder_model,
            vocoder=vocoder,
            overtraining_detector=overtraining_detector,
            custom_pretrained=custom_pretrained,
            g_pretrained_path=g_pretrained_path,
            d_pretrained_path=d_pretrained_path,
        )

        job_id = train_queue.enqueue_async(
            training_service.train_from_request,
            request,
        )

        return {"status": "queued", "job_id": job_id, "model_id": model_id}
    except HTTPException:
        raise
    except FileNotFoundError as exc:
        logger.error("Train request failed: %s", exc)
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception("Unexpected training error")
        raise HTTPException(status_code=500, detail=str(exc))
