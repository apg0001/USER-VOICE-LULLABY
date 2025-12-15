"""추론 라우터"""

from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from ..dependencies import (
    get_file_repository,
    get_inference_queue,
    get_inference_service,
)
from ..models.requests import InferenceFilesRequest
from ..repositories.file_repository import FileRepository
from ..services.inference_service import InferenceService
from ..settings import INFERENCE_DEFAULTS
from ..task_queue import AsyncJobQueue
from ..logging_config import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.post("/inference")
async def start_inference(
    target_audio: UploadFile = File(...),
    model_path: str = Form(...),
    index_path: Optional[str] = Form(None),
    output_dir: Optional[str] = Form(None),
    pitch: Optional[int] = Form(None),
    volume_envelope: Optional[float] = Form(None),
    protect: Optional[float] = Form(None),
    f0_autotune: Optional[bool] = Form(None),
    f0_autotune_strength: Optional[float] = Form(None),
    embedder_model: Optional[str] = Form(None),
    index_rate: Optional[float] = Form(None),
    clean_audio: Optional[bool] = Form(None),
    clean_strength: Optional[float] = Form(None),
    reverb: Optional[bool] = Form(None),
    reverb_room_size: Optional[float] = Form(None),
    reverb_damping: Optional[float] = Form(None),
    reverb_wet_gain: Optional[float] = Form(None),
    reverb_dry_gain: Optional[float] = Form(None),
    reverb_width: Optional[float] = Form(None),
    reverb_freeze_mode: Optional[float] = Form(None),
    inference_service: InferenceService = Depends(get_inference_service),
    inference_queue: AsyncJobQueue = Depends(get_inference_queue),
    file_repo: FileRepository = Depends(get_file_repository),
):
    """파일 업로드로 추론 시작 요청 처리"""
    logger.info(f"추론 요청 수신 | model_path={model_path} | index_path={index_path}")

    # 입력 검증
    if not target_audio:
        logger.error("추론 요청 실패: 오디오 파일이 업로드되지 않았습니다")
        raise HTTPException(status_code=400, detail="오디오 파일을 업로드해야 합니다")

    if not model_path or not model_path.strip():
        logger.error("추론 요청 실패: 모델 경로가 제공되지 않았습니다")
        raise HTTPException(status_code=400, detail="모델 경로를 제공해야 합니다")

    # 기본값 적용 (None인 경우에만)
    output_dir = output_dir if output_dir is not None else "outputs"
    pitch = pitch if pitch is not None else INFERENCE_DEFAULTS.pitch
    volume_envelope = (
        volume_envelope
        if volume_envelope is not None
        else INFERENCE_DEFAULTS.volume_envelope
    )
    protect = protect if protect is not None else INFERENCE_DEFAULTS.protect
    f0_autotune = (
        f0_autotune if f0_autotune is not None else INFERENCE_DEFAULTS.f0_autotune
    )
    f0_autotune_strength = (
        f0_autotune_strength
        if f0_autotune_strength is not None
        else INFERENCE_DEFAULTS.f0_autotune_strength
    )
    embedder_model = (
        embedder_model
        if embedder_model is not None
        else INFERENCE_DEFAULTS.embedder_model
    )
    index_rate = index_rate if index_rate is not None else INFERENCE_DEFAULTS.index_rate
    clean_audio = (
        clean_audio if clean_audio is not None else INFERENCE_DEFAULTS.clean_audio
    )
    clean_strength = (
        clean_strength
        if clean_strength is not None
        else INFERENCE_DEFAULTS.clean_strength
    )
    reverb = reverb if reverb is not None else INFERENCE_DEFAULTS.reverb
    reverb_room_size = (
        reverb_room_size
        if reverb_room_size is not None
        else INFERENCE_DEFAULTS.reverb_room_size
    )
    reverb_damping = (
        reverb_damping
        if reverb_damping is not None
        else INFERENCE_DEFAULTS.reverb_damping
    )
    reverb_wet_gain = (
        reverb_wet_gain
        if reverb_wet_gain is not None
        else INFERENCE_DEFAULTS.reverb_wet_gain
    )
    reverb_dry_gain = (
        reverb_dry_gain
        if reverb_dry_gain is not None
        else INFERENCE_DEFAULTS.reverb_dry_gain
    )
    reverb_width = (
        reverb_width if reverb_width is not None else INFERENCE_DEFAULTS.reverb_width
    )
    reverb_freeze_mode = (
        reverb_freeze_mode
        if reverb_freeze_mode is not None
        else INFERENCE_DEFAULTS.reverb_freeze_mode
    )

    logger.debug(
        f"추론 파라미터 | model_path={model_path} | index_path={index_path} | "
        f"output_dir={output_dir} | pitch={pitch} | volume_envelope={volume_envelope} | protect={protect} | "
        f"f0_autotune={f0_autotune} | embedder_model={embedder_model} | index_rate={index_rate}"
    )

    try:
        # 파일 저장
        logger.info(f"추론 오디오 파일 저장 시작 | model_path={model_path}")
        temp_audio_path = file_repo.save_inference_audio(target_audio)
        logger.info(f"추론 오디오 파일 저장 완료 | temp_audio_path={temp_audio_path}")

        # 비동기 작업 등록 (모든 요청을 큐에 추가, 리소스 확인은 워커에서 수행)
        request = InferenceFilesRequest(
            input_audio_path=str(temp_audio_path),
            model_path=model_path,
            index_path=index_path,
            output_dir=output_dir,
            pitch=pitch,
            volume_envelope=volume_envelope,
            protect=protect,
            f0_autotune=f0_autotune,
            f0_autotune_strength=f0_autotune_strength,
            embedder_model=embedder_model,
            index_rate=index_rate,
            clean_audio=clean_audio,
            clean_strength=clean_strength,
            reverb=reverb,
            reverb_room_size=reverb_room_size,
            reverb_damping=reverb_damping,
            reverb_wet_gain=reverb_wet_gain,
            reverb_dry_gain=reverb_dry_gain,
            reverb_width=reverb_width,
            reverb_freeze_mode=reverb_freeze_mode,
        )

        # 비동기 작업 등록
        try:
            job_id = inference_queue.enqueue_async(
                inference_service.infer_from_request,
                request,
            )
            logger.debug(f"작업 큐에 등록됨 | job_id={job_id}")
        except Exception as e:
            logger.error(
                f"작업 큐 등록 실패 | model_path={model_path} | error={e}",
                exc_info=True,
            )
            raise HTTPException(status_code=500, detail=f"작업 큐 등록 실패: {str(e)}")

        logger.info(
            f"추론 작업 등록 완료 | job_id={job_id} | model_path={model_path} | "
            f"index_path={index_path} | output_dir={output_dir} | pitch={pitch} | "
            f"volume_envelope={volume_envelope} | protect={protect} | "
            f"f0_autotune={f0_autotune} | embedder_model={embedder_model} | index_rate={index_rate} | "
            f"clean_audio={clean_audio} | clean_strength={clean_strength} | reverb={reverb} | "
            f"reverb_room_size={reverb_room_size} | reverb_damping={reverb_damping} | "
            f"reverb_wet_gain={reverb_wet_gain} | reverb_dry_gain={reverb_dry_gain} | "
            f"reverb_width={reverb_width} | reverb_freeze_mode={reverb_freeze_mode}"
        )

        return {"status": "queued", "job_id": job_id}
    except HTTPException:
        raise
    except ValueError as exc:
        logger.error(
            f"추론 요청 검증 실패 | model_path={model_path} | error={exc}",
            exc_info=True,
        )
        raise HTTPException(status_code=400, detail=str(exc))
    except FileNotFoundError as exc:
        logger.error(
            f"추론 요청 실패: 파일을 찾을 수 없음 | model_path={model_path} | error={exc}",
            exc_info=True,
        )
        raise HTTPException(status_code=404, detail=str(exc))
    except OSError as exc:
        logger.error(
            f"추론 요청 실패: 파일 시스템 오류 | model_path={model_path} | error={exc}",
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=f"파일 시스템 오류: {str(exc)}")
    except Exception as exc:
        logger.exception(
            f"추론 요청 예상치 못한 오류 | model_path={model_path} | error={exc}"
        )
        raise HTTPException(status_code=500, detail=f"내부 서버 오류: {str(exc)}")
