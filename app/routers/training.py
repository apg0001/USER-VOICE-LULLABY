"""학습 라우터"""

from typing import List, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from ..dependencies import (
    get_file_repository,
    get_train_queue,
    get_training_service,
)
from ..models.requests import TrainFilesRequest
from ..repositories.file_repository import FileRepository
from ..services.training_service import TrainingService
from ..settings import TRAINING_DEFAULTS
from ..task_queue import AsyncJobQueue
from ..logging_config import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.post("/train")
async def start_training(
    sample_rate: Optional[int] = Form(None),
    total_epoch: Optional[int] = Form(None),
    batch_size: Optional[int] = Form(None),
    files: List[UploadFile] = File(...),
    embedder_model: Optional[str] = Form(None),
    vocoder: Optional[str] = Form(None),
    overtraining_detector: Optional[bool] = Form(None),
    custom_pretrained: Optional[bool] = Form(None),
    g_pretrained_path: Optional[str] = Form(None),
    d_pretrained_path: Optional[str] = Form(None),
    model_description: Optional[str] = Form(None),
    training_service: TrainingService = Depends(get_training_service),
    train_queue: AsyncJobQueue = Depends(get_train_queue),
    file_repo: FileRepository = Depends(get_file_repository),
):
    """파일 업로드로 학습 요청 처리"""
    logger.info(f"학습 요청 수신 | files_count={len(files) if files else 0}")
    
    # 파일 검증
    if not files or len(files) == 0:
        logger.error("학습 요청 실패: 파일이 업로드되지 않았습니다")
        raise HTTPException(status_code=400, detail="최소 하나의 파일을 업로드해야 합니다")
    
    # 모델 ID 자동 생성
    model_id = str(uuid4())
    logger.debug(f"생성된 모델 ID: {model_id}")

    # 기본값 적용 (None인 경우에만)
    sample_rate = sample_rate if sample_rate is not None else TRAINING_DEFAULTS.sample_rate
    total_epoch = total_epoch if total_epoch is not None else TRAINING_DEFAULTS.total_epoch
    batch_size = batch_size if batch_size is not None else TRAINING_DEFAULTS.batch_size
    embedder_model = embedder_model if embedder_model is not None else TRAINING_DEFAULTS.embedder_model
    vocoder = vocoder if vocoder is not None else TRAINING_DEFAULTS.vocoder
    overtraining_detector = overtraining_detector if overtraining_detector is not None else TRAINING_DEFAULTS.overtraining_detector
    custom_pretrained = custom_pretrained if custom_pretrained is not None else TRAINING_DEFAULTS.custom_pretrained
    g_pretrained_path = g_pretrained_path if g_pretrained_path is not None else TRAINING_DEFAULTS.g_pretrained_path
    d_pretrained_path = d_pretrained_path if d_pretrained_path is not None else TRAINING_DEFAULTS.d_pretrained_path
    
    logger.debug(
        f"학습 파라미터 | model_id={model_id} | sample_rate={sample_rate} | "
        f"total_epoch={total_epoch} | batch_size={batch_size} | embedder_model={embedder_model} | "
        f"vocoder={vocoder} | custom_pretrained={custom_pretrained}"
    )

    try:
        # 파일 저장
        logger.debug(f"학습 파일 저장 시작 | model_id={model_id} | files_count={len(files)}")
        dataset_path = file_repo.save_training_files(model_id, files)
        logger.debug(f"학습 파일 저장 완료 | dataset_path={dataset_path}")

        # 비동기 작업 등록 (모든 요청을 큐에 추가, 리소스 확인은 워커에서 수행)
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
            model_description=model_description,
        )

        # 비동기 작업 등록
        try:
            job_id = train_queue.enqueue_async(
                training_service.train_from_request,
                request,
            )
            logger.debug(f"작업 큐에 등록됨 | job_id={job_id}")
        except Exception as e:
            logger.error(f"작업 큐 등록 실패 | model_id={model_id} | error={e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"작업 큐 등록 실패: {str(e)}")
        
        # job에 metadata 저장 (진행률 계산용 및 모델 정보)
        try:
            job = train_queue.get_job_status(job_id)
            if job:
                job.metadata = {
                    "model_name": model_id,
                    "model_id": model_id,
                    "model_description": model_description,
                    "total_epoch": total_epoch,
                }
                logger.debug(f"작업 메타데이터 저장 완료 | job_id={job_id} | model_id={model_id}")
            else:
                logger.warning(f"작업 상태를 찾을 수 없음 | job_id={job_id}")
        except Exception as e:
            logger.warning(f"작업 메타데이터 저장 실패 (무시) | job_id={job_id} | error={e}")

        logger.info(
            f"학습 작업 등록 완료 | job_id={job_id} | model_id={model_id} | "
            f"sample_rate={sample_rate} | total_epoch={total_epoch} | batch_size={batch_size} | "
            f"embedder_model={embedder_model} | vocoder={vocoder} | dataset_path={dataset_path}"
        )

        return {"status": "queued", "job_id": job_id, "model_id": model_id}
    except HTTPException:
        raise
    except ValueError as exc:
        logger.error(f"학습 요청 검증 실패 | model_id={model_id} | error={exc}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(exc))
    except FileNotFoundError as exc:
        logger.error(f"학습 요청 실패: 파일을 찾을 수 없음 | model_id={model_id} | error={exc}", exc_info=True)
        raise HTTPException(status_code=404, detail=str(exc))
    except OSError as exc:
        logger.error(f"학습 요청 실패: 파일 시스템 오류 | model_id={model_id} | error={exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"파일 시스템 오류: {str(exc)}")
    except Exception as exc:
        logger.exception(f"학습 요청 예상치 못한 오류 | model_id={model_id} | error={exc}")
        raise HTTPException(status_code=500, detail=f"내부 서버 오류: {str(exc)}")
