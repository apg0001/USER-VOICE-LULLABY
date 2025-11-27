from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Optional
from uuid import uuid4

# 프로젝트 루트 디렉토리를 기준으로 RVC 관련 경로 설정
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RVC_ROOT = PROJECT_ROOT / "applio"
INNER_RVC = RVC_ROOT / "rvc"

# RVC 루트가 존재하지 않으면 오류 발생
if not RVC_ROOT.exists():
    raise RuntimeError(f"rvc 디렉터리를 찾을 수 없습니다: {RVC_ROOT}")

# 현재 작업 디렉토리 저장 후 RVC 루트로 변경하여
# RVC 관련 모듈 임포트 전 환경 설정
_ORIGINAL_CWD = Path.cwd()
try:
    if _ORIGINAL_CWD != RVC_ROOT:
        os.chdir(RVC_ROOT)  # 작업디렉토리 이동
    # RVC 내부 모듈 경로를 sys.path에 추가하여 임포트 가능하게 설정
    for path in (INNER_RVC, RVC_ROOT):
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)
    # RVC 핵심 스크립트 임포트
    from core import (
        run_extract_script,
        run_infer_script,
        run_preprocess_script,
        run_train_script,
        run_prerequisites_script
    )
finally:
    os.chdir(_ORIGINAL_CWD)  # 작업 디렉토리 원복

# 설정값 임포트
from .settings import INFERENCE_DEFAULTS, TRAINING_DEFAULTS

logger = logging.getLogger(__name__)
RVC_LOGS_DIR = RVC_ROOT / "logs"  # 로그 저장 폴더
DEFAULT_OUTPUT_DIR = RVC_ROOT / "outputs"  # 출력 파일 기본 경로

# 디렉토리가 없으면 생성해주는 헬퍼함수
def _ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

# 절대경로 혹은 base 경로 기준 절대경로 변환
def _resolve_path(input_path: str, base: Path) -> Path:
    path_obj = Path(input_path)
    if not path_obj.is_absolute():
        path_obj = base / path_obj
    return path_obj.resolve()

# 모델별 로그 디렉토리 생성 및 반환
def _logs_dir(model_name: str) -> Path:
    return _ensure_directory(RVC_LOGS_DIR / model_name)

# 차단(blocking) 함수 비동기 실행 도와주는 헬퍼
async def _run_blocking(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

# 모델 학습 함수, 비동기로 학습 스크립트 호출
async def train_model(
    model_name: str,
    dataset_path: str,
    sample_rate: Optional[int] = None,
    total_epoch: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> dict:
    defaults = TRAINING_DEFAULTS
    sample_rate = sample_rate or defaults.sample_rate
    total_epoch = total_epoch or defaults.total_epoch
    batch_size = batch_size or defaults.batch_size

    dataset = _resolve_path(dataset_path, RVC_ROOT)
    if not dataset.exists():
        raise FileNotFoundError(f"Dataset 경로를 찾을 수 없습니다: {dataset}")

    model_dir = _logs_dir(model_name)
    logger.info("Training start | model=%s dataset=%s", model_name, dataset)

    # prerequisites, preprocess, extract, train 스크립트를 순차 실행
    await _run_blocking(
        run_prerequisites_script,
        True,
        True,
        True
    )

    await _run_blocking(
        run_preprocess_script,
        model_name,
        str(dataset),
        sample_rate,
        defaults.cpu_cores,
        defaults.cut_preprocess,
        defaults.process_effects,
        defaults.noise_reduction,
        defaults.clean_strength,
        defaults.chunk_len,
        defaults.overlap_len,
        defaults.normalization_mode,
    )

    await _run_blocking(
        run_extract_script,
        model_name,
        defaults.f0_method,
        defaults.cpu_cores,
        defaults.gpu,
        sample_rate,
        defaults.embedder_model,
        None,
        defaults.include_mutes,
    )

    await _run_blocking(
        run_train_script,
        model_name,
        defaults.save_every_epoch,
        defaults.save_only_latest,
        defaults.save_every_weights,
        total_epoch,
        sample_rate,
        batch_size,
        defaults.gpu,
        defaults.overtraining_detector,
        defaults.overtraining_threshold,
        defaults.pretrained,
        defaults.cleanup,
        defaults.index_algorithm,
        defaults.cache_data_in_gpu,
        False,
        None,
        None,
        defaults.vocoder,
        defaults.checkpointing,
    )

    logger.info("Training finished | model=%s dir=%s", model_name, model_dir)
    return {
        "model_name": model_name,
        "logs_dir": str(model_dir.resolve()),
        "sample_rate": sample_rate,
        "epochs": total_epoch,
    }

# 추론 실행 함수, 비동기로 infer 스크립트 호출
async def run_inference(
    input_audio_path: str,
    model_path: str,
    index_path: Optional[str] = None,
    output_dir: str = "outputs",
) -> dict:
    defaults = INFERENCE_DEFAULTS
    input_path = _resolve_path(input_audio_path, RVC_ROOT)
    model_file = _resolve_path(model_path, RVC_ROOT)
    idx_path = _resolve_path(index_path, RVC_ROOT) if index_path else None

    if not input_path.exists():
        raise FileNotFoundError(f"입력 오디오 경로를 찾을 수 없습니다: {input_path}")
    if not model_file.exists():
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_file}")
    if idx_path and not idx_path.exists():
        raise FileNotFoundError(f"인덱스 파일을 찾을 수 없습니다: {idx_path}")

    resolved_output_dir = (
        _resolve_path(output_dir, RVC_ROOT)
        if output_dir
        else DEFAULT_OUTPUT_DIR
    )
    output_folder = _ensure_directory(resolved_output_dir)
    temp_output = output_folder / f"{uuid4().hex}.wav"

    message, exported = await _run_blocking(
        run_infer_script,
        defaults.pitch,
        defaults.index_rate,
        defaults.volume_envelope,
        defaults.protect,
        defaults.f0_method,
        str(input_path),
        str(temp_output),
        str(model_file),
        str(idx_path) if idx_path else "",
        defaults.split_audio,
        defaults.f0_autotune,
        defaults.f0_autotune_strength,
        defaults.proposed_pitch,
        defaults.proposed_pitch_threshold,
        defaults.clean_audio,
        defaults.clean_strength,
        defaults.export_format,
        defaults.embedder_model,
        None,
        defaults.formant_shifting,
        defaults.formant_qfrency,
        defaults.formant_timbre,
        defaults.post_process,
    )

    logger.info(
        "Inference finished | input=%s model=%s output=%s",
        input_path,
        model_file,
        exported,
    )

    return {
        "message": message,
        "output_path": exported,
        "input_audio": str(input_path.resolve()),
        "model_path": str(model_file.resolve()),
        "index_path": str(idx_path.resolve()) if idx_path else None,
    }
