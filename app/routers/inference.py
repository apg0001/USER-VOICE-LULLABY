"""추론 라우터"""

from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from ..dependencies import (
    get_file_repository,
    get_inference_queue,
    get_inference_service,
    get_resource_monitor,
)
from ..models.requests import InferenceRequest, InferenceFilesRequest
from ..monitors.resource_monitor import ResourceMonitor
from ..repositories.file_repository import FileRepository
from ..services.inference_service import InferenceService
from ..settings import INFERENCE_DEFAULTS
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


@router.post("/inference")
async def start_inference(
    payload: InferenceRequest,
    inference_service: InferenceService = Depends(get_inference_service),
    inference_queue: AsyncJobQueue = Depends(get_inference_queue),
    monitor: ResourceMonitor = Depends(get_resource_monitor),
):
    """추론 시작 요청 처리"""
    try:
        # 리소스 확인
        # _check_and_raise_if_resources_unavailable(monitor)

        # 비동기 작업 등록
        job_id = inference_queue.enqueue_async(
            inference_service.infer,
            input_audio_path=payload.input_audio_path,
            model_path=payload.model_path,
            index_path=payload.index_path,
            output_dir=payload.output_dir,
        )
        return {"status": "queued", "job_id": job_id}
    except HTTPException:
        raise
    except FileNotFoundError as exc:
        logger.error("Inference request failed: %s", exc)
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception("Unexpected inference error")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/inference-files")
async def start_inference_files(
    target_audio: UploadFile = File(...),
    model_path: str = Form(...),
    index_path: Optional[str] = Form(None),
    output_dir: str = Form("outputs"),
    volume_envelope: Optional[float] = Form(INFERENCE_DEFAULTS.volume_envelope),
    protect: Optional[float] = Form(INFERENCE_DEFAULTS.protect),
    f0_autotune: Optional[bool] = Form(INFERENCE_DEFAULTS.f0_autotune),
    f0_autotune_strength: Optional[float] = Form(
        INFERENCE_DEFAULTS.f0_autotune_strength
    ),
    embedder_model: Optional[str] = Form(INFERENCE_DEFAULTS.embedder_model),
    inference_service: InferenceService = Depends(get_inference_service),
    inference_queue: AsyncJobQueue = Depends(get_inference_queue),
    file_repo: FileRepository = Depends(get_file_repository),
    monitor: ResourceMonitor = Depends(get_resource_monitor),
):
    """파일 업로드로 추론 시작 요청 처리"""
    logger.info(f"파일 업로드 추론 요청: {target_audio.filename}, model: {model_path}")

    try:
        # 파일 저장
        temp_audio_path = file_repo.save_inference_audio(target_audio)

        # 리소스 확인
        # _check_and_raise_if_resources_unavailable(monitor)

        # 비동기 작업 등록
        request = InferenceFilesRequest(
            input_audio_path=str(temp_audio_path),
            model_path=model_path,
            index_path=index_path,
            output_dir=output_dir,
            volume_envelope=volume_envelope,
            protect=protect,
            f0_autotune=f0_autotune,
            f0_autotune_strength=f0_autotune_strength,
            embedder_model=embedder_model,
        )

        job_id = inference_queue.enqueue_async(
            inference_service.infer_from_request,
            request,
        )

        return {"status": "queued", "job_id": job_id}
    except HTTPException:
        raise
    except FileNotFoundError as exc:
        logger.error("Inference request failed: %s", exc)
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception("Unexpected inference error")
        raise HTTPException(status_code=500, detail=str(exc))
