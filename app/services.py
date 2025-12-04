from __future__ import annotations

import asyncio
import os
import shutil
import sys
from pathlib import Path
from typing import Optional
from uuid import uuid4

import librosa
import numpy as np
import soundfile as sf
from spleeter.separator import Separator

from .logging_config import PROJECT_ROOT, get_logger

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
        run_prerequisites_script,
    )
finally:
    os.chdir(_ORIGINAL_CWD)  # 작업 디렉토리 원복

# 설정값 임포트
from .settings import INFERENCE_DEFAULTS, TRAINING_DEFAULTS

logger = get_logger(__name__)

RVC_LOGS_DIR = RVC_ROOT / "logs"  # 모델 저장 폴더
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


# 학습 완료 후 학습용 데이터셋 삭제
async def _remove_dataset(dataset_path):
    path = Path(dataset_path)
    if not path.exists():
        logger.error(f"삭제할 경로를 찾을 수 없습니다: {dataset_path}")
        raise FileNotFoundError(f"삭제할 경로를 찾을 수 없습니다: {dataset_path}")

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, lambda: shutil.rmtree(path))


# model_dir 내 모든 파일과 폴더를 삭제하되, 확장자가 .pth 인 파일만 유지
async def _remove_preprocess(model_dir):
    path = Path(model_dir)
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(
            f"디렉토리를 찾을 수 없거나 디렉토리가 아닙니다: {model_dir}"
        )

    def _clean_dir():
        for item in path.iterdir():
            if item.is_file():
                if item.suffix not in [".pth", ".index"]:
                    try:
                        item.unlink()
                    except Exception as e:
                        logger.warning(f"파일 삭제 실패: {item} - {e}")
            elif item.is_dir():
                try:
                    # 폴더 내 모든 내용 삭제 후 폴더 삭제
                    shutil.rmtree(item)
                except Exception as e:
                    logger.warning(f"폴더 삭제 실패: {item} - {e}")

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _clean_dir)


async def _remove_file(file_path: str):
    path = Path(file_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"삭제할 파일을 찾을 수 없습니다: {file_path}")

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, path.unlink)


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
    await _run_blocking(run_prerequisites_script, True, True, True)

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

    # 학습이 실제로 완료되었는지 확인 (모델 파일이 생성되었는지 확인)
    model_files = list(model_dir.glob("*.index"))
    if not model_files:
        raise RuntimeError(
            f"학습이 완료되지 않았습니다. 모델 파일(.index)이 생성되지 않았습니다: {model_dir}"
        )
    logger.info(f"생성된 모델 파일 수: {len(model_files)}")

    # 학습용 데이터셋 및 중간 산출물 정리
    try:
        await _remove_dataset(dataset_path)
        logger.info("학습용 데이터셋 삭제 완료: %s", dataset_path)
    except FileNotFoundError:
        logger.warning("삭제 대상 데이터셋 경로 없음: %s", dataset_path)
    except Exception as e:
        logger.warning(f"데이터셋 삭제 중 오류 발생 (무시): {dataset_path} - {e}")

    try:
        await _remove_preprocess(model_dir)
        logger.info("모델 .pth 파일을 제외한 전처리 산출물 삭제 완료: %s", model_dir)
    except FileNotFoundError:
        logger.warning("전처리 디렉토리 삭제 대상 없음: %s", model_dir)
    except Exception as e:
        logger.warning(f"전처리 산출물 삭제 중 오류 발생 (무시): {model_dir} - {e}")

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

    # 파일 존재 확인
    if not input_path.exists():
        raise FileNotFoundError(f"입력 오디오 경로를 찾을 수 없습니다: {input_path}")
    if not model_file.exists():
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_file}")
    if idx_path and not idx_path.exists():
        raise FileNotFoundError(f"인덱스 파일을 찾을 수 없습니다: {idx_path}")

    resolved_output_dir = _resolve_path(output_dir, RVC_ROOT)
    output_folder = _ensure_directory(resolved_output_dir)

    # 고유 ID 생성
    unique_id = uuid4().hex

    # 정리용 변수 초기화
    vocals_path = None
    instrumental_path = None
    temp_vocal_output = None
    separation_folder = None

    try:
        # 1단계: 보컬/인스트루멘탈 분리
        logger.info(f"보컬 분리 시작: {input_path}")
        separation_result = await separate_vocal_instrumental(
            str(input_path), str(output_folder)
        )
        vocals_path = Path(separation_result["vocals"])
        instrumental_path = Path(separation_result["instrumental"])
        separation_folder = vocals_path.parent  # spleeter가 생성한 폴더
        logger.info(f"분리 완료 - 보컬: {vocals_path}, 인스트: {instrumental_path}")

        # 2단계: 보컬만 inference 실행
        logger.info(f"보컬 inference 시작: {vocals_path}")
        temp_vocal_output = output_folder / f"{unique_id}_vocal_infer.wav"
        vocal_message, vocal_exported = await _run_blocking(
            run_infer_script,
            defaults.pitch,
            defaults.index_rate,
            defaults.volume_envelope,
            defaults.protect,
            defaults.f0_method,
            str(vocals_path),
            str(temp_vocal_output),
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
        logger.info(f"보컬 inference 완료: {vocal_exported}")

        # 3단계: 변환된 보컬 + 원본 인스트루멘탈 합성
        logger.info("오디오 합성 시작")
        final_output = output_folder / f"{unique_id}_final.wav"
        final_output_path = await merge_vocal_instrumental(
            str(vocal_exported), str(instrumental_path), str(final_output)
        )
        logger.info(f"최종 합성 완료: {final_output_path}")

        # 최종 출력 파일이 실제로 생성되었는지 확인
        final_path = Path(final_output_path)
        if not final_path.exists() or not final_path.is_file():
            raise RuntimeError(
                f"최종 출력 파일이 생성되지 않았습니다: {final_output_path}"
            )

        return {
            "message": f"보컬 분리 → 변환 → 합성 완료 | {vocal_message}",
            "output_path": str(final_output_path),
            # "input_audio": str(input_path.resolve()),
            "model_path": str(model_file.resolve()),
            "index_path": str(idx_path.resolve()) if idx_path else None,
            # "vocal_separated": str(vocals_path),
            # "instrumental": str(instrumental_path),
            # "vocal_inferred": str(vocal_exported),
        }

    except Exception as e:
        # 추론 실패 시 예외를 다시 발생시켜 엔드포인트에서 처리하도록 함
        logger.error(f"추론 처리 중 오류 발생: {e}")
        raise
    finally:
        # 예외 발생 여부와 관계없이 항상 임시 파일 정리
        cleanup_paths = [
            vocals_path,
            instrumental_path,
            temp_vocal_output,
        ]
        for path in cleanup_paths:
            if path and path.exists():
                try:
                    await _remove_file(str(path))
                    logger.info(f"임시 파일 삭제: {path}")
                except Exception as e:
                    logger.warning(f"임시 파일 삭제 실패: {path} - {e}")

        # spleeter가 생성한 폴더 전체 삭제 (output_dir/temp_inference_xxx/)
        if (
            separation_folder
            and separation_folder.exists()
            and separation_folder.is_dir()
        ):
            try:
                await _run_blocking(shutil.rmtree, separation_folder)
                logger.info(f"임시 추론 폴더 삭제 완료: {separation_folder}")
            except Exception as e:
                logger.warning(f"임시 추론 폴더 삭제 실패: {separation_folder} - {e}")


async def separate_vocal_instrumental(input_audio_path: str, output_dir: str) -> dict:
    """오디오를 보컬/인스트루멘탈로 분리 (문자열 경로 사용)"""
    input_path = Path(input_audio_path)
    if not input_path.exists():
        raise FileNotFoundError(f"입력 파일 없음: {input_audio_path}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    separator = Separator("spleeter:2stems")
    separator.separate_to_file(input_audio_path, output_dir)

    base_name = input_path.stem
    vocals_path = output_path / base_name / f"vocals.wav"
    instrumental_path = output_path / base_name / f"accompaniment.wav"

    return {"vocals": str(vocals_path), "instrumental": str(instrumental_path)}


async def merge_vocal_instrumental(
    vocals_path: str, instrumental_path: str, output_path: str
) -> str:
    """변환된 보컬과 원본 인스트루멘탈 합성"""
    loop = asyncio.get_running_loop()

    def _merge_audio():
        vocals, sr_v = librosa.load(vocals_path, sr=None, mono=True)
        instrumental, sr_i = librosa.load(instrumental_path, sr=None, mono=True)

        if sr_v != sr_i:
            raise ValueError("샘플레이트 불일치")

        # 길이 맞추기
        max_len = max(len(vocals), len(instrumental))
        vocals = np.pad(vocals, (0, max_len - len(vocals)), "constant")
        instrumental = np.pad(
            instrumental, (0, max_len - len(instrumental)), "constant"
        )

        # 단순 덧셈 합성
        mixed = vocals + instrumental

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, mixed, sr_v)
        return output_path

    return await loop.run_in_executor(None, _merge_audio)
