from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .services import run_inference, train_model
from .settings import INFERENCE_DEFAULTS, TRAINING_DEFAULTS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE_PATH = LOGS_DIR / "app.log"

# 롤링 파일 핸들러 (최대 100MB, 백업 10개)
handler = RotatingFileHandler(
    filename=str(LOG_FILE_PATH),
    maxBytes=100 * 1024 * 1024,  # 100MB
    backupCount=10,
    encoding="utf-8"
)

logging.basicConfig(
    level=logging.DEBUG, format="[%(asctime)s] %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger("applio.api")

# 기존 핸들러 제거 후 추가 (중복 방지용)
if logger.hasHandlers():
    logger.handlers.clear()

logger.addHandler(handler)

APP_DIR = Path(__file__).resolve().parent
PUBLIC_DIR = APP_DIR / "public"

app = FastAPI(
    title="Applio FastAPI Server",
    version="1.0.0",
    description="Minimal training & inference backend without GUI.",
)

if PUBLIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=PUBLIC_DIR), name="static")


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


class InferenceRequest(BaseModel):
    input_audio_path: str = Field(..., description="변환할 입력 오디오 경로")
    model_path: str = Field(..., description=".pth 모델 가중치 경로")
    index_path: Optional[str] = Field(
        None, description="선택적 .index 파일 경로 (없으면 자동으로 비활성화)"
    )
    output_dir: str = Field(
        "outputs", description="추론 결과를 저장할 디렉토리 (자동 생성)"
    )


class HealthResponse(BaseModel):
    status: str
    training_defaults: dict
    inference_defaults: dict


def _serialize_defaults(defaults) -> dict:
    return {k: getattr(defaults, k) for k in defaults.__dataclass_fields__.keys()}


@app.get("/", response_model=HealthResponse)
async def health_check():
    return {
        "status": "ok",
        "training_defaults": _serialize_defaults(TRAINING_DEFAULTS),
        "inference_defaults": _serialize_defaults(INFERENCE_DEFAULTS),
    }


@app.post("/train")
async def start_training(payload: TrainRequest):
    try:
        result = await train_model(
            model_name=payload.model_name,
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


@app.post("/inference")
async def start_inference(payload: InferenceRequest):
    try:
        result = await run_inference(
            input_audio_path=payload.input_audio_path,
            model_path=payload.model_path,
            index_path=payload.index_path,
            output_dir=payload.output_dir,
        )
        return {"status": "completed", **result}
    except FileNotFoundError as exc:
        logger.error("Inference request failed: %s", exc)
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception("Unexpected inference error")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/ui", include_in_schema=False)
async def serve_ui():
    index_path = PUBLIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="정적 UI 파일을 찾을 수 없습니다.")
    return FileResponse(index_path)


###############################################################################################################

from fastapi import UploadFile, Form, Depends, File
from typing import List
import os

# 프로젝트 루트 디렉토리를 기준으로 RVC 관련 경로 설정
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RVC_ROOT = PROJECT_ROOT / "applio"

@app.post("/train-files")
async def start_training2(
    model_name: str = Form(...),
    sample_rate: int = Form(TRAINING_DEFAULTS.sample_rate),
    total_epoch: int = Form(TRAINING_DEFAULTS.total_epoch),
    batch_size: int = Form(TRAINING_DEFAULTS.batch_size),
    files: List[UploadFile] = File(...)
):
    try:
        dataset_path = RVC_ROOT / f"/datasets/{model_name}"
        os.makedirs(dataset_path, exist_ok=True)
        logger.debug(f"데이터셋 저장 경로 생성: {dataset_path}")
        
        for idx, file in enumerate(files):
            ext = file.filename.split(".")[-1]
            file_path = os.path.join(dataset_path, f"audio_{idx+1:03d}.{ext}")
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            logger.debug(f"개별 업로드 파일 저장: {idx}: {file_path}-{file.filename}")
    except Exception as exc:
        logger.exception(f"데이터셋 저장 중 오류 발생: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
        
    try:
        result = await train_model(
            model_name=model_name,
            dataset_path=dataset_path,
            sample_rate=sample_rate,
            total_epoch=total_epoch,
            batch_size=batch_size,
        )
        return {"status": "running", **result}
    except FileNotFoundError as exc:
        logger.error("Train request failed: %s", exc)
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception("Unexpected training error")
        raise HTTPException(status_code=500, detail=str(exc))