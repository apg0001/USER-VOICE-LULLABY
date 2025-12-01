from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .services import run_inference, train_model
from .settings import INFERENCE_DEFAULTS, TRAINING_DEFAULTS

# 프로젝트 루트 경로 및 로그 저장 디렉토리 설정
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)  # 없으면 생성
LOG_FILE_PATH = LOGS_DIR / "app.log"
ALLOWED_ROOT = PROJECT_ROOT / "applio/output"

# 로그 롤링 핸들러: 최대 100MB, 최대 10개 파일 유지
handler = RotatingFileHandler(
    filename=str(LOG_FILE_PATH),
    maxBytes=100 * 1024 * 1024,
    backupCount=10,
    encoding="utf-8"
)

# 로그 포맷 설정
formatter = logging.Formatter(
    fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
handler.setFormatter(formatter)

logger1 = logging.getLogger("applio.api")
logger2 = logging.getLogger("fastapi")

logger1.setLevel(logging.INFO)
logger2.setLevel(logging.INFO)

# # 기존 로그 핸들러 제거 후 새 롤링 핸들러 추가 (중복 방지)
# if logger1.hasHandlers():
#     logger1.handlers.clear()
# logger1.addHandler(handler)

# # 기존 로그 핸들러 제거 후 새 롤링 핸들러 추가 (중복 방지)
# if logger2.hasHandlers():
#     logger2.handlers.clear()
# logger2.addHandler(handler)

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

# 헬스체크 응답 모델
class HealthResponse(BaseModel):
    status: str
    training_defaults: dict
    inference_defaults: dict

# 설정 객체를 dict로 변환하는 헬퍼 함수
def _serialize_defaults(defaults) -> dict:
    return {k: getattr(defaults, k) for k in defaults.__dataclass_fields__.keys()}

# 헬스 체크 엔드포인트
@app.get("/", response_model=HealthResponse)
async def health_check():
    return {
        "status": "ok",
        "training_defaults": _serialize_defaults(TRAINING_DEFAULTS),
        "inference_defaults": _serialize_defaults(INFERENCE_DEFAULTS),
    }

# 학습 시작 요청 처리
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
        logger2.error("Train request failed: %s", exc)
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger2.exception("Unexpected training error")
        raise HTTPException(status_code=500, detail=str(exc))

# 추론 시작 요청 처리
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
        logger2.error("Inference request failed: %s", exc)
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger2.exception("Unexpected inference error")
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
import os

# 프로젝트 루트 기준 RVC 데이터셋 저장 폴더
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RVC_ROOT = PROJECT_ROOT / "applio"
DATASET_ROOT = RVC_ROOT / "datasets"

# 파일 업로드로 학습 요청 처리
@app.post("/train-files")
async def start_training2(
    model_name: str = Form(...),
    sample_rate: int = Form(TRAINING_DEFAULTS.sample_rate),
    total_epoch: int = Form(TRAINING_DEFAULTS.total_epoch),
    batch_size: int = Form(TRAINING_DEFAULTS.batch_size),
    files: List[UploadFile] = File(...)
):
    logger2.info(
        f"파일 업로드 학습 요청 mn: {model_name}, sr: {sample_rate}, e: {total_epoch}, bs: {batch_size}, f: {len(files)}"
    )
    try:
        # 모델명 기준 폴더 생성 (datasets 하위)
        dataset_path = DATASET_ROOT / model_name
        os.makedirs(dataset_path, exist_ok=True)
        logger2.info(f"데이터셋 저장 경로 생성: {dataset_path}")

        # 업로드한 파일들 순차적으로 저장
        for idx, file in enumerate(files):
            ext = file.filename.split(".")[-1]
            file_path = os.path.join(dataset_path, f"audio_{idx+1:03d}.{ext}")
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            logger2.info(f"개별 업로드 파일 저장: {idx}: {file_path}-{file.filename}")
    except Exception as exc:
        logger2.exception(f"데이터셋 저장 중 오류 발생: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

    try:
        # 저장된 데이터셋 폴더를 이용해 학습 시작
        result = await train_model(
            model_name=model_name,
            dataset_path=dataset_path,
            sample_rate=sample_rate,
            total_epoch=total_epoch,
            batch_size=batch_size,
        )
        return {"status": "running", **result}
    except FileNotFoundError as exc:
        logger2.error("Train request failed: %s", exc)
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger2.exception("Unexpected training error")
        raise HTTPException(status_code=500, detail=str(exc))


# 파일 업로드로 추론 시작 요청 처리
@app.post("/inference-files")
async def start_inference_files(
    target_audio: UploadFile = File(..., description="변환할 입력 오디오 파일"),
    model_path: str = Form(..., description=".pth 모델 가중치 경로"),
    index_path: Optional[str] = Form(None, description="선택적 .index 파일 경로"),
    output_dir: str = Form("outputs", description="출력 디렉토리 (기본값: outputs)")
):
    logger2.info(f"파일 업로드 추론 요청: {target_audio.filename}, model: {model_path}")

    try:
        # 모델명 기준 폴더 생성 (datasets 하위)
        AUDIO_ROOT = DATASET_ROOT / "target_audio"
        os.makedirs(AUDIO_ROOT, exist_ok=True)
        logger1.info(f"타깃 오디오 저장 경로 생성: {AUDIO_ROOT}")
        
        temp_audio_path = AUDIO_ROOT / f"temp_inference_{uuid4().hex}.{target_audio.filename.split('.')[-1]}"
        with open(temp_audio_path, "wb") as f:
            content = await target_audio.read()
            f.write(content)
            
        logger1.info(f"임시 오디오 파일 저장 완료: {temp_audio_path}")
    except Exception as exc:
        logger1.exception(f"타깃 오디오 저장 중 오류 발생: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

    try:
        result = await run_inference(
            input_audio_path=str(temp_audio_path),
            model_path=model_path,
            index_path=index_path,
            output_dir=output_dir,
        )
        return {"status": "completed", **result}
    except FileNotFoundError as exc:
        logger1.error("Inference request failed: %s", exc)
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger1.exception("Unexpected inference error")
        raise HTTPException(status_code=500, detail=str(exc))
    
@app.get("/download")
async def download_file(path: str = Query(..., description="오디오 파일 이름")):
    requested_path = (ALLOWED_ROOT / path).resolve()
    logger2.info(f"다운로드 요청: {requested_path}")
    if not requested_path.is_file() or not requested_path.is_relative_to(ALLOWED_ROOT):
        raise HTTPException(status_code=404, detail="File not found")
    
    filename = os.path.basename(path)
    return FileResponse(
        path,
        filename=filename,
        media_type="audio/wav",
    )