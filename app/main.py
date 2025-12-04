from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Optional
from uuid import uuid4

import psutil
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .logging_config import PROJECT_ROOT, configure_logging, get_logger
from .services import run_inference, train_model
from .settings import INFERENCE_DEFAULTS, TRAINING_DEFAULTS
from .task_queue import AsyncJobQueue

# 공통 경로 설정
ALLOWED_ROOT = PROJECT_ROOT / "applio/output"

# 작업 큐 구성
train_queue = AsyncJobQueue("train")
inference_queue = AsyncJobQueue("inference")

logger_app = get_logger("applio.api")
logger_fastapi = get_logger("fastapi")

# FastAPI 정적 파일 제공 경로 설정
APP_DIR = Path(__file__).resolve().parent
PUBLIC_DIR = APP_DIR / "public"

app = FastAPI(
    title="Applio FastAPI Server",
    version="1.0.0",
    description="Minimal training & inference backend without GUI.",
)

# 정적 파일 디렉토리가 존재하면 /static 경로에 마운트
if PUBLIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=PUBLIC_DIR), name="static")


# 학습 요청용 데이터 모델
class TrainRequest(BaseModel):
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


# 추론 요청용 데이터 모델
class InferenceRequest(BaseModel):
    input_audio_path: str = Field(..., description="변환할 입력 오디오 경로")
    model_path: str = Field(..., description=".pth 모델 가중치 경로")
    index_path: Optional[str] = Field(
        None, description="선택적 .index 파일 경로 (없으면 자동으로 비활성화)"
    )
    output_dir: str = Field(
        "outputs", description="추론 결과를 저장할 디렉토리 (자동 생성)"
    )


class QueueStats(BaseModel):
    name: str
    running: int  # 실행 중 작업 수
    pending: int  # 대기 중 작업 수


class HealthResponse(BaseModel):
    status: str
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    queues: dict[str, QueueStats]
    gpus: list[dict] | None = None


# 설정 객체를 dict로 변환하는 헬퍼 함수
def _serialize_defaults(defaults) -> dict:
    return {k: getattr(defaults, k) for k in defaults.__dataclass_fields__.keys()}


@app.on_event("startup")
async def _on_startup() -> None:
    """애플리케이션 시작 시 로깅 및 작업 큐를 초기화한다."""
    configure_logging()
    await train_queue.start()
    await inference_queue.start()


@app.on_event("shutdown")
async def _on_shutdown() -> None:
    """애플리케이션 종료 시 작업 큐를 정리한다."""
    await train_queue.stop()
    await inference_queue.stop()


# 헬스 체크 엔드포인트
@app.get("/", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    cpu_percent = psutil.cpu_percent(interval=0.0)
    memory_percent = psutil.virtual_memory().percent
    disk_usage = shutil.disk_usage(PROJECT_ROOT)
    disk_percent = disk_usage.used / disk_usage.total * 100 if disk_usage.total else 0.0

    queue_stats = {
        "train": QueueStats(**train_queue.stats()),
        "inference": QueueStats(**inference_queue.stats()),
    }

    # GPU 통계 (NVIDIA NVML이 있으면)
    gpus = None
    try:
        import pynvml

        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        gpu_list = []
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_list.append(
                {
                    "index": i,
                    "name": name,
                    "utilization_percent": float(util.gpu),
                    "memory_used_mb": round(mem.used / (1024 * 1024), 2),
                    "memory_total_mb": round(mem.total / (1024 * 1024), 2),
                }
            )
        pynvml.nvmlShutdown()
        gpus = gpu_list
    except Exception as e:
        logging.error(f"GPU info fetch failed: {e}")
        gpus = None

    return HealthResponse(
        status="ok",
        cpu_percent=cpu_percent,
        memory_percent=memory_percent,
        disk_percent=disk_percent,
        queues=queue_stats,
        gpus=gpus,
    )


# 학습 시작 요청 처리
@app.post("/train")
async def start_training(payload: TrainRequest):
    try:
        result = await train_queue.enqueue(
            train_model,
            model_name=payload.model_name,
            dataset_path=payload.dataset_path,
            sample_rate=payload.sample_rate,
            total_epoch=payload.total_epoch,
            batch_size=payload.batch_size,
        )
        return {"status": "running", **result}
    except FileNotFoundError as exc:
        logger_fastapi.error("Train request failed: %s", exc)
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger_fastapi.exception("Unexpected training error")
        raise HTTPException(status_code=500, detail=str(exc))


# 추론 시작 요청 처리
@app.post("/inference")
async def start_inference(payload: InferenceRequest):
    try:
        result = await inference_queue.enqueue(
            run_inference,
            input_audio_path=payload.input_audio_path,
            model_path=payload.model_path,
            index_path=payload.index_path,
            output_dir=payload.output_dir,
        )
        return {"status": "completed", **result}
    except FileNotFoundError as exc:
        logger_fastapi.error("Inference request failed: %s", exc)
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger_fastapi.exception("Unexpected inference error")
        raise HTTPException(status_code=500, detail=str(exc))


# UI 정적 페이지 제공 (index.html)
@app.get("/ui", include_in_schema=False)
async def serve_ui():
    index_path = PUBLIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="정적 UI 파일을 찾을 수 없습니다.")
    return FileResponse(index_path)


###############################################################################################################

from fastapi import UploadFile, Form, Depends, File
from typing import List

# 프로젝트 루트 기준 RVC 데이터셋 저장 폴더
RVC_ROOT = PROJECT_ROOT / "applio"
DATASET_ROOT = RVC_ROOT / "datasets"


# 파일 업로드로 학습 요청 처리
@app.post("/train-files")
async def start_training2(
    model_name: str = Form(...),
    sample_rate: int = Form(TRAINING_DEFAULTS.sample_rate),
    total_epoch: int = Form(TRAINING_DEFAULTS.total_epoch),
    batch_size: int = Form(TRAINING_DEFAULTS.batch_size),
    files: List[UploadFile] = File(...),
    embedder_model: Optional[str] = Form(
        TRAINING_DEFAULTS.embedder_model,
        description="선택적 모델 종류(contentvec or spin-v2)",
    ),
    overtraining_detector: Optional[str] = Form(
        TRAINING_DEFAULTS.overtraining_detector,
        description="과적합 방지",
    ),
):
    logger_fastapi.info(f"파일 업로드 학습 요청")
    logger_fastapi.info(f"모델명: {model_name}")
    logger_fastapi.info(f"Sample Rate: {sample_rate}")
    logger_fastapi.info(f"Total Epochs: {total_epoch}")
    logger_fastapi.info(f"배치 사이즈: {batch_size}")
    logger_fastapi.info(f"입력 파일 수: {len(files)}")
    logger_fastapi.info(f"Embedder Model: {embedder_model}")
    logger_fastapi.info(f"과적합 방지: {overtraining_detector}")
    try:
        # 모델명 기준 폴더 생성 (datasets 하위)
        dataset_path = DATASET_ROOT / model_name
        os.makedirs(dataset_path, exist_ok=True)
        logger_fastapi.info(f"데이터셋 저장 경로 생성: {dataset_path}")

        # 업로드한 파일들 순차적으로 저장
        for idx, file in enumerate(files):
            ext = file.filename.split(".")[-1]
            file_path = os.path.join(dataset_path, f"audio_{idx+1:03d}.{ext}")
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            logger_fastapi.info(
                f"개별 업로드 파일 저장: {idx}: {file_path}-{file.filename}"
            )
    except Exception as exc:
        logger_fastapi.exception(f"데이터셋 저장 중 오류 발생: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

    try:
        # 저장된 데이터셋 폴더를 이용해 학습 시작
        result = await train_queue.enqueue(
            train_model,
            model_name=model_name,
            dataset_path=str(dataset_path),
            sample_rate=sample_rate,
            total_epoch=total_epoch,
            batch_size=batch_size,
            embedder_model=embedder_model,
            overtraining_detector=overtraining_detector,
        )
        return {"status": "running", **result}
    except FileNotFoundError as exc:
        logger_fastapi.error("Train request failed: %s", exc)
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger_fastapi.exception("Unexpected training error")
        raise HTTPException(status_code=500, detail=str(exc))


# 파일 업로드로 추론 시작 요청 처리
@app.post("/inference-files")
async def start_inference_files(
    target_audio: UploadFile = File(..., description="변환할 입력 오디오 파일 경로"),
    model_path: str = Form(..., description=".pth 모델 가중치 경로"),
    index_path: Optional[str] = Form(None, description="선택적 .index 파일 경로"),
    output_dir: str = Form("outputs", description="출력 디렉토리 (기본값: outputs)"),
    volume_envelope: Optional[float] = Form(
        INFERENCE_DEFAULTS.volume_envelope,
        description="선택적 원곡의 다이내믹 강약 정도",
    ),
    protect: Optional[float] = Form(
        INFERENCE_DEFAULTS.protect, description="선택적 원곡 포맷 보호 정도"
    ),
    f0_autotune: Optional[bool] = Form(
        INFERENCE_DEFAULTS.f0_autotune,
        description="선택적 원곡 피치를 스케일에 맞처 부드럽게 보정",
    ),
    f0_autotune_strength: Optional[float] = Form(
        INFERENCE_DEFAULTS.f0_autotune_strength, description="선택적"
    ),
    embedder_model: Optional[str] = Form(
        INFERENCE_DEFAULTS.embedder_model,
        description="선택적 모델 종류(contentvec or spin-v2)",
    ),
):
    logger_fastapi.info(
        f"파일 업로드 추론 요청: {target_audio.filename}, model: {model_path}"
    )

    try:
        # 모델명 기준 폴더 생성 (datasets 하위)
        AUDIO_ROOT = DATASET_ROOT / "target_audio"
        os.makedirs(AUDIO_ROOT, exist_ok=True)
        logger_app.info(f"타깃 오디오 저장 경로 생성: {AUDIO_ROOT}")

        temp_audio_path = (
            AUDIO_ROOT
            / f"temp_inference_{uuid4().hex}.{target_audio.filename.split('.')[-1]}"
        )
        with open(temp_audio_path, "wb") as f:
            content = await target_audio.read()
            f.write(content)

        logger_app.info(f"임시 오디오 파일 저장 완료: {temp_audio_path}")
    except Exception as exc:
        logger_app.exception(f"타깃 오디오 저장 중 오류 발생: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

    try:
        result = await inference_queue.enqueue(
            run_inference,
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
        return {"status": "completed", **result}
    except FileNotFoundError as exc:
        logger_app.error("Inference request failed: %s", exc)
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger_app.exception("Unexpected inference error")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/download")
async def download_file(path: str = Query(..., description="오디오 파일 이름")):
    requested_path = (ALLOWED_ROOT / path).resolve()
    logger_fastapi.info(f"다운로드 요청: {requested_path}")
    if not requested_path.is_file() or not requested_path.is_relative_to(ALLOWED_ROOT):
        raise HTTPException(status_code=404, detail="File not found")

    filename = os.path.basename(path)
    return FileResponse(
        path,
        filename=filename,
        media_type="audio/wav",
    )
