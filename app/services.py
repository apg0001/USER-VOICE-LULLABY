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

from app.constants import RVC_LOGS_DIR, RVC_ROOT, DEFAULT_OUTPUT_DIR
from app.logging_config import PROJECT_ROOT, get_logger

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
from app.settings import INFERENCE_DEFAULTS, TRAINING_DEFAULTS

logger = get_logger(__name__)

__all__ = ["run_inference", "train_model"]


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
    """학습용 데이터셋 디렉토리 삭제 (존재하지 않으면 무시)"""
    path = Path(dataset_path)
    if not path.exists():
        logger.debug(f"삭제할 데이터셋 경로가 존재하지 않음 (이미 삭제되었거나 생성되지 않음): {dataset_path}")
        return
    
    if not path.is_dir():
        logger.warning(f"데이터셋 경로가 디렉토리가 아님: {dataset_path}")
        return

    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: shutil.rmtree(path))
        logger.info(f"데이터셋 디렉토리 삭제 완료: {dataset_path}")
    except Exception as e:
        logger.error(f"데이터셋 삭제 중 오류 발생: {dataset_path} - {e}")
        raise


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
    embedder_model: Optional[str] = None,
    vocoder: Optional[str] = None,
    overtraining_detector: Optional[bool] = None,
    custom_pretrained: bool = False,
    g_pretrained_path: str = None,
    d_pretrained_path: str = None,
) -> dict:
    defaults = TRAINING_DEFAULTS
    sample_rate = sample_rate or defaults.sample_rate
    total_epoch = total_epoch or defaults.total_epoch
    batch_size = batch_size or defaults.batch_size
    embedder_model = embedder_model or defaults.embedder_model
    vocoder = vocoder or defaults.vocoder
    overtraining_detector = overtraining_detector or defaults.overtraining_detector
    custom_pretrained = custom_pretrained or defaults.custom_pretrained
    g_pretrained_path = g_pretrained_path or defaults.g_pretrained_path
    d_pretrained_path = d_pretrained_path or defaults.d_pretrained_path

    dataset = _resolve_path(dataset_path, RVC_ROOT)
    if not dataset.exists():
        raise FileNotFoundError(f"Dataset 경로를 찾을 수 없습니다: {dataset}")

    model_dir = _logs_dir(model_name)
    logger.info("Training start | model=%s dataset=%s", model_name, dataset)

    # 보컬 분리 수행 (추론과 동일한 방식)
    logger.info("학습용 오디오 파일 보컬 분리 시작")
    audio_extensions = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
    audio_files = [f for f in dataset.iterdir() if f.is_file() and f.suffix.lower() in audio_extensions]
    
    if not audio_files:
        raise ValueError(f"학습용 오디오 파일을 찾을 수 없습니다: {dataset}")
    
    logger.info(f"보컬 분리 대상 파일 수: {len(audio_files)}")
    
    # 임시 분리 폴더 생성
    temp_separation_dir = dataset / "temp_separation"
    temp_separation_dir.mkdir(exist_ok=True)
    
    separation_folders = []  # 정리용
    
    try:
        for audio_file in audio_files:
            logger.info(f"보컬 분리 중: {audio_file.name}")
            try:
                # 보컬 분리 수행
                separation_result = await separate_vocal_instrumental(
                    str(audio_file), str(temp_separation_dir)
                )
                vocals_path = Path(separation_result["vocals"])
                
                if not vocals_path.exists():
                    logger.warning(f"보컬 분리 실패: {vocals_path} - 원본 파일 사용")
                    continue
                
                # 원본 파일을 보컬 파일로 교체
                # 원본 파일 확장자 유지
                original_ext = audio_file.suffix
                backup_path = audio_file.with_suffix(f".original{original_ext}")
                
                # 원본 파일 백업 (안전을 위해)
                if not backup_path.exists():
                    shutil.copy2(audio_file, backup_path)
                
                # 보컬 파일을 원본 파일 위치로 복사
                shutil.copy2(vocals_path, audio_file)
                logger.info(f"보컬 파일로 교체 완료: {audio_file.name}")
                
                # 분리 폴더 기록 (나중에 정리)
                separation_folder = vocals_path.parent
                if separation_folder not in separation_folders:
                    separation_folders.append(separation_folder)
                
            except Exception as e:
                logger.error(f"보컬 분리 실패 (원본 파일 사용): {audio_file.name} - {e}")
                # 보컬 분리 실패 시 원본 파일 그대로 사용
        
        logger.info("학습용 오디오 파일 보컬 분리 완료")
        
    finally:
        # 임시 분리 폴더 정리
        if temp_separation_dir.exists():
            try:
                await _run_blocking(shutil.rmtree, temp_separation_dir)
                logger.info(f"임시 분리 폴더 삭제 완료: {temp_separation_dir}")
            except Exception as e:
                logger.warning(f"임시 분리 폴더 삭제 실패: {temp_separation_dir} - {e}")
        
        # 각 파일의 분리 폴더 정리
        for sep_folder in separation_folders:
            if sep_folder.exists() and sep_folder.is_dir():
                try:
                    await _run_blocking(shutil.rmtree, sep_folder)
                    logger.debug(f"분리 폴더 삭제 완료: {sep_folder}")
                except Exception as e:
                    logger.warning(f"분리 폴더 삭제 실패: {sep_folder} - {e}")

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
        embedder_model,
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
        overtraining_detector,
        defaults.overtraining_threshold,
        defaults.pretrained,
        defaults.cleanup,
        defaults.index_algorithm,
        defaults.cache_data_in_gpu,
        custom_pretrained,
        g_pretrained_path,
        d_pretrained_path,
        vocoder,
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

    # 모델 파일과 인덱스 파일의 절대 경로 수집
    pth_files = list(model_dir.glob("*.pth"))
    index_files = list(model_dir.glob("*.index"))
    
    pth_files_absolute = [str(f.resolve()) for f in pth_files]
    index_files_absolute = [str(f.resolve()) for f in index_files]

    # 최종 모델 저장 경로 로그 출력
    if pth_files_absolute:
        logger.info(f"최종 모델 저장 경로 (.pth): {', '.join(pth_files_absolute)}")
    if index_files_absolute:
        logger.info(f"최종 인덱스 파일 저장 경로 (.index): {', '.join(index_files_absolute)}")
    logger.info(f"모델 디렉토리 절대 경로: {model_dir.resolve()}")

    # 모델 정보를 JSON 파일로 저장
    model_info_json = {
        "model_name": model_name,
        "embedder_model": embedder_model,
        "sample_rate": sample_rate,
        "total_epoch": total_epoch,
        "vocoder": vocoder,
        "model_files_absolute": pth_files_absolute,
        "index_files_absolute": index_files_absolute,
    }
    
    import json
    model_info_path = model_dir / "model_info.json"
    with open(model_info_path, "w", encoding="utf-8") as f:
        json.dump(model_info_json, f, indent=2, ensure_ascii=False)
    logger.info(f"모델 정보 저장 완료: {model_info_path}")

    # 학습용 데이터셋 및 중간 산출물 정리
    try:
        await _remove_dataset(dataset_path)
    except Exception as e:
        logger.error(f"데이터셋 삭제 실패: {dataset_path} - {e}")
        # 데이터셋 삭제 실패 시에도 경고만 하고 계속 진행 (학습은 완료되었으므로)
        logger.warning("데이터셋 삭제 실패했지만 학습은 완료되었습니다. 수동으로 삭제해주세요.")

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
    volume_envelope: Optional[float] = None,
    protect: Optional[float] = None,
    f0_autotune: Optional[bool] = None,
    f0_autotune_strength: Optional[float] = None,
    embedder_model: Optional[str] = None,
    index_rate: Optional[float] = None,
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

    volume_envelope = volume_envelope or defaults.volume_envelope
    protect = protect or defaults.protect
    f0_autotune = f0_autotune or defaults.f0_autotune
    f0_autotune_strength = defaults.f0_autotune_strength
    embedder_model = defaults.embedder_model

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
            index_rate,
            volume_envelope,
            protect,
            defaults.f0_method,
            str(vocals_path),
            str(temp_vocal_output),
            str(model_file),
            str(idx_path) if idx_path else "",
            defaults.split_audio,
            f0_autotune,
            f0_autotune_strength,
            defaults.proposed_pitch,
            defaults.proposed_pitch_threshold,
            defaults.clean_audio,
            defaults.clean_strength,
            defaults.export_format,
            embedder_model,
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

        # 입력 오디오 파일이 임시 파일인 경우 삭제 (target_audio 디렉토리의 temp_inference_* 파일)
        input_path_obj = Path(input_audio_path)
        if (
            input_path_obj.exists()
            and "target_audio" in str(input_path_obj)
            and input_path_obj.name.startswith("temp_inference_")
        ):
            try:
                await _remove_file(input_audio_path)
                logger.info(f"입력 임시 오디오 파일 삭제 완료: {input_audio_path}")
            except Exception as e:
                logger.warning(
                    f"입력 임시 오디오 파일 삭제 실패: {input_audio_path} - {e}"
                )


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
            instrumental = librosa.resample(instrumental, orig_sr=sr_i, target_sr=sr_v)
            sr_i = sr_v

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
