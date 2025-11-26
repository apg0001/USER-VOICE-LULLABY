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

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger("applio.api")

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

