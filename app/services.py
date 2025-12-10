from __future__ import annotations

import asyncio
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
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

if not RVC_ROOT.exists():
    raise RuntimeError(f"rvc 디렉터리를 찾을 수 없습니다: {RVC_ROOT}")

# RVC 모듈 임포트 전 환경 설정
# RVC 스크립트는 특정 작업 디렉토리에서 실행되어야 하며,
# sys.path에 RVC 내부 경로가 포함되어야 정상적으로 임포트됩니다.
_ORIGINAL_CWD = Path.cwd()
try:
    if _ORIGINAL_CWD != RVC_ROOT:
        os.chdir(RVC_ROOT)
    # RVC 내부 모듈 경로를 sys.path에 추가하여 core 모듈 임포트 가능하게 설정
    for path in (INNER_RVC, RVC_ROOT):
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)
    from core import (
        run_extract_script,
        run_infer_script,
        run_preprocess_script,
        run_train_script,
        run_prerequisites_script,
    )
finally:
    os.chdir(_ORIGINAL_CWD)

from app.settings import INFERENCE_DEFAULTS, TRAINING_DEFAULTS

logger = get_logger(__name__)

__all__ = ["run_inference", "train_model"]


def _ensure_directory(path: Path) -> Path:
    """디렉토리가 없으면 생성하고 반환"""
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_path(input_path: str, base: Path) -> Path:
    """상대 경로를 base 기준으로 절대 경로로 변환"""
    path_obj = Path(input_path)
    if not path_obj.is_absolute():
        path_obj = base / path_obj
    return path_obj.resolve()


def _logs_dir(model_name: str) -> Path:
    """모델별 로그 디렉토리 생성 및 반환"""
    return _ensure_directory(RVC_LOGS_DIR / model_name)


async def _run_blocking(func, *args, **kwargs):
    """블로킹 함수를 별도 스레드에서 비동기로 실행"""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))


async def _remove_dataset(dataset_path):
    """학습용 데이터셋 디렉토리 삭제 (존재하지 않으면 무시)"""
    path = Path(dataset_path)
    if not path.exists():
        logger.debug(
            f"삭제할 데이터셋 경로가 존재하지 않음 (이미 삭제되었거나 생성되지 않음): {dataset_path}"
        )
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


def _update_model_info_files(model_dir: Path) -> None:
    """학습 완료 후 model_info.json의 모델 파일(.pth)과 인덱스 파일(.index) 경로를 업데이트

    학습 중 생성된 모든 .pth와 .index 파일의 절대 경로를 수집하여
    model_info.json에 저장합니다. UI에서 모델 리스트 조회 시 사용됩니다.
    """
    import json

    model_info_path = model_dir / "model_info.json"
    if not model_info_path.exists():
        logger.warning(f"model_info.json 파일이 없습니다: {model_info_path}")
        return

    try:
        with open(model_info_path, "r", encoding="utf-8") as f:
            model_info_json = json.load(f)

        pth_files = list(model_dir.glob("*.pth"))
        index_files = list(model_dir.glob("*.index"))

        pth_files_absolute = sorted([str(f.resolve()) for f in pth_files])
        index_files_absolute = sorted([str(f.resolve()) for f in index_files])

        existing_pth = set(model_info_json.get("model_files_absolute", []))
        existing_index = set(model_info_json.get("index_files_absolute", []))

        new_pth = set(pth_files_absolute)
        new_index = set(index_files_absolute)

        added_pth = new_pth - existing_pth
        added_index = new_index - existing_index

        if added_pth:
            logger.info(
                f"새 모델 파일 감지 및 model_info.json 업데이트: {', '.join(sorted(added_pth))}"
            )
        if added_index:
            logger.info(
                f"새 인덱스 파일 감지 및 model_info.json 업데이트: {', '.join(sorted(added_index))}"
            )

        model_info_json["model_files_absolute"] = sorted(list(new_pth))
        model_info_json["index_files_absolute"] = sorted(list(new_index))

        with open(model_info_path, "w", encoding="utf-8") as f:
            json.dump(model_info_json, f, indent=2, ensure_ascii=False)

        if added_pth or added_index:
            logger.debug(f"model_info.json 업데이트 완료: {model_info_path}")

    except Exception as e:
        logger.error(
            f"model_info.json 업데이트 실패: {model_info_path} - {e}", exc_info=True
        )


async def _remove_preprocess(model_dir):
    """학습 완료 후 전처리 산출물 정리

    학습 중 생성된 전처리 산출물(특징 추출 파일, 임시 파일 등)을 삭제합니다.
    모델 파일(.pth), 인덱스 파일(.index), model_info.json은 유지합니다.
    """
    path = Path(model_dir)
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(
            f"디렉토리를 찾을 수 없거나 디렉토리가 아닙니다: {model_dir}"
        )

    def _clean_dir():
        for item in path.iterdir():
            if item.is_file():
                # 모델 파일, 인덱스 파일, 모델 정보 파일은 유지
                if (
                    item.suffix not in [".pth", ".index"]
                    and item.name != "model_info.json"
                ):
                    try:
                        item.unlink()
                    except Exception as e:
                        logger.warning(f"파일 삭제 실패: {item} - {e}")
            elif item.is_dir():
                try:
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
    model_description: Optional[str] = None,
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

    # 학습 시작 전에 model_info.json 생성
    # 학습 중간에 실패하더라도 모델 정보는 남아있어야 하므로,
    # 학습 시작 전에 기본 정보를 저장합니다.
    import json

    model_info_path = model_dir / "model_info.json"
    model_info_json = {
        "model_name": model_name,
        "embedder_model": embedder_model,
        "sample_rate": sample_rate,
        "total_epoch": total_epoch,
        "vocoder": vocoder,
        "model_files_absolute": [],  # 학습 완료 후 _update_model_info_files에서 업데이트
        "index_files_absolute": [],  # 학습 완료 후 _update_model_info_files에서 업데이트
        "model_description": model_description,
    }

    model_dir.mkdir(parents=True, exist_ok=True)

    with open(model_info_path, "w", encoding="utf-8") as f:
        json.dump(model_info_json, f, indent=2, ensure_ascii=False)
    logger.info(f"모델 정보 파일 생성 완료 (학습 시작 전): {model_info_path}")

    # 학습용 오디오 파일 보컬 분리
    # 추론과 동일하게 보컬만 추출하여 학습 품질을 향상시킵니다.
    logger.info("학습용 오디오 파일 보컬 분리 시작")
    audio_extensions = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
    audio_files = [
        f
        for f in dataset.iterdir()
        if f.is_file() and f.suffix.lower() in audio_extensions
    ]

    if not audio_files:
        raise ValueError(f"학습용 오디오 파일을 찾을 수 없습니다: {dataset}")

    logger.info(f"보컬 분리 대상 파일 수: {len(audio_files)}")

    # spleeter가 생성하는 임시 분리 폴더를 위한 디렉토리
    temp_separation_dir = dataset / "temp_separation"
    temp_separation_dir.mkdir(exist_ok=True)

    separation_folders = []

    try:
        for audio_file in audio_files:
            logger.info(f"보컬 분리 중: {audio_file.name}")
            try:
                separation_result = await separate_vocal_instrumental(
                    str(audio_file), str(temp_separation_dir)
                )
                vocals_path = Path(separation_result["vocals"])

                if not vocals_path.exists():
                    logger.warning(f"보컬 분리 실패: {vocals_path} - 원본 파일 사용")
                    continue

                # 빈 파일 체크: TensorFlow 오류로 인해 파일이 생성되었지만 비어있을 수 있음
                if vocals_path.stat().st_size == 0:
                    logger.warning(
                        f"보컬 분리 결과 파일이 비어있음: {vocals_path} - 원본 파일 사용"
                    )
                    continue

                # 원본 파일을 보컬 파일로 교체
                # 원본 파일 확장자는 유지하며, 보컬 파일로 덮어쓰기합니다.
                # 백업은 생성하지 않습니다 (사용자가 원본을 보존하려면 미리 백업해야 함).
                shutil.copy2(vocals_path, audio_file)
                logger.info(f"보컬 파일로 교체 완료: {audio_file.name}")

                # 나중에 정리하기 위해 분리 폴더 경로 저장
                separation_folder = vocals_path.parent
                if separation_folder not in separation_folders:
                    separation_folders.append(separation_folder)

            except RuntimeError as e:
                logger.error(
                    f"보컬 분리 실패 (원본 파일 사용): {audio_file.name} - {e}",
                    exc_info=True,
                )
            except Exception as e:
                logger.error(
                    f"보컬 분리 중 예상치 못한 오류 (원본 파일 사용): {audio_file.name} - {e}",
                    exc_info=True,
                )

        logger.info("학습용 오디오 파일 보컬 분리 완료")

    finally:
        # 보컬 분리 성공/실패 여부와 관계없이 임시 파일 정리
        if temp_separation_dir.exists():
            try:
                await _run_blocking(shutil.rmtree, temp_separation_dir)
                logger.info(f"임시 분리 폴더 삭제 완료: {temp_separation_dir}")
            except Exception as e:
                logger.warning(f"임시 분리 폴더 삭제 실패: {temp_separation_dir} - {e}")

        # spleeter가 각 파일마다 생성한 분리 폴더 정리
        for sep_folder in separation_folders:
            if sep_folder.exists() and sep_folder.is_dir():
                try:
                    await _run_blocking(shutil.rmtree, sep_folder)
                    logger.debug(f"분리 폴더 삭제 완료: {sep_folder}")
                except Exception as e:
                    logger.warning(f"분리 폴더 삭제 실패: {sep_folder} - {e}")

    # RVC 학습 파이프라인 순차 실행
    # prerequisites: 의존성 확인 및 초기화
    # preprocess: 오디오 전처리 (샘플링, 노이즈 제거 등)
    # extract: 특징 추출 (F0, 임베딩 등)
    # train: 모델 학습 및 인덱스 생성
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

    # 학습 완료 검증: 인덱스 파일 생성 여부 확인
    # 인덱스 파일이 없으면 학습이 제대로 완료되지 않은 것으로 간주
    model_files = list(model_dir.glob("*.index"))
    if not model_files:
        raise RuntimeError(
            f"학습이 완료되지 않았습니다. 모델 파일(.index)이 생성되지 않았습니다: {model_dir}"
        )
    logger.info(f"생성된 모델 파일 수: {len(model_files)}")

    # 학습 완료 후 생성된 모든 .pth와 .index 파일 경로를 model_info.json에 저장
    _update_model_info_files(model_dir)

    # 학습용 데이터셋 및 중간 산출물 정리
    # 데이터셋 삭제 실패해도 학습은 완료되었으므로 경고만 출력하고 계속 진행
    try:
        await _remove_dataset(dataset_path)
    except Exception as e:
        logger.error(f"데이터셋 삭제 실패: {dataset_path} - {e}")
        logger.warning(
            "데이터셋 삭제 실패했지만 학습은 완료되었습니다. 수동으로 삭제해주세요."
        )

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

    unique_id = uuid4().hex

    # finally 블록에서 정리하기 위한 변수
    vocals_path = None
    instrumental_path = None
    temp_vocal_output = None
    separation_folder = None

    try:
        # 1단계: 보컬/인스트루멘탈 분리
        # spleeter를 사용하여 입력 오디오를 보컬과 인스트루멘탈로 분리합니다.
        # 보컬만 변환하여 원본 인스트루멘탈과 합성하면 더 자연스러운 결과를 얻을 수 있습니다.
        logger.info(f"보컬 분리 시작: {input_path}")
        separation_result = await separate_vocal_instrumental(
            str(input_path), str(output_folder)
        )
        vocals_path = Path(separation_result["vocals"])
        instrumental_path = Path(separation_result["instrumental"])
        separation_folder = vocals_path.parent
        logger.info(f"분리 완료 - 보컬: {vocals_path}, 인스트: {instrumental_path}")

        # 2단계: 보컬만 inference 실행
        # 분리된 보컬에만 음성 변환을 적용합니다.
        # 인스트루멘탈은 원본 그대로 유지하여 음질 손실을 최소화합니다.
        logger.info(f"보컬 inference 시작: {vocals_path}")
        temp_vocal_output = output_folder / f"{unique_id}_vocal_infer.wav"

        try:
            # prerequisites: 의존성 확인 및 초기화
            await _run_blocking(run_prerequisites_script, True, True, True)

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
        except Exception as e:
            logger.error(f"보컬 inference 실행 중 오류 발생: {e}", exc_info=True)
            raise RuntimeError(f"보컬 inference 실패: {str(e)}")

        # 출력 파일 검증: TensorFlow 오류로 인해 파일이 생성되지 않았거나 비어있을 수 있음
        vocal_exported_path = Path(vocal_exported)
        if not vocal_exported_path.exists() or not vocal_exported_path.is_file():
            error_msg = (
                f"보컬 inference 출력 파일이 생성되지 않았습니다: {vocal_exported}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        if vocal_exported_path.stat().st_size == 0:
            error_msg = f"보컬 inference 출력 파일이 비어있습니다: {vocal_exported}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        logger.info(
            f"보컬 inference 완료: {vocal_exported} (크기: {vocal_exported_path.stat().st_size} bytes)"
        )

        # 3단계: 변환된 보컬과 원본 인스트루멘탈 합성
        # 변환된 보컬과 원본 인스트루멘탈을 믹싱하여 최종 출력을 생성합니다.
        logger.info("오디오 합성 시작")
        final_output = output_folder / f"{unique_id}_final.wav"

        try:
            final_output_path = await merge_vocal_instrumental(
                str(vocal_exported), str(instrumental_path), str(final_output)
            )
        except Exception as e:
            logger.error(f"오디오 합성 중 오류 발생: {e}", exc_info=True)
            raise RuntimeError(f"오디오 합성 실패: {str(e)}")

        logger.info(f"최종 합성 완료: {final_output_path}")

        # 최종 출력 파일 검증
        final_path = Path(final_output_path)
        if not final_path.exists() or not final_path.is_file():
            error_msg = f"최종 출력 파일이 생성되지 않았습니다: {final_output_path}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        if final_path.stat().st_size == 0:
            error_msg = f"최종 출력 파일이 비어있습니다: {final_output_path}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        logger.info(
            f"최종 출력 파일 확인 완료: {final_output_path} (크기: {final_path.stat().st_size} bytes)"
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
        logger.error(f"추론 처리 중 오류 발생: {e}")
        raise
    finally:
        # 성공/실패 여부와 관계없이 임시 파일 정리
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

        # spleeter가 생성한 분리 폴더 삭제
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

        # 업로드된 임시 입력 파일 삭제
        # target_audio 디렉토리에 저장된 temp_inference_* 파일은 추론 완료 후 삭제합니다.
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


# 보컬 분리 작업 동시 실행 제한
# spleeter는 TensorFlow를 사용하는데, TensorFlow는 스레드 안전하지 않습니다.
# 전용 스레드 풀(max_workers=1)을 사용하여 한 번에 하나의 작업만 실행되도록 제한합니다.
_separation_lock = asyncio.Lock()
_separation_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="spleeter")


async def separate_vocal_instrumental(input_audio_path: str, output_dir: str) -> dict:
    """오디오를 보컬/인스트루멘탈로 분리

    spleeter를 사용하여 입력 오디오를 보컬과 인스트루멘탈로 분리합니다.
    TensorFlow Graph 충돌 방지를 위해 락을 사용하여 동시 실행을 제한합니다.
    """
    input_path = Path(input_audio_path)
    if not input_path.exists():
        raise FileNotFoundError(f"입력 파일 없음: {input_audio_path}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    base_name = input_path.stem
    vocals_path = output_path / base_name / f"vocals.wav"
    instrumental_path = output_path / base_name / f"accompaniment.wav"

    # 락을 사용하여 동시 실행 제한
    # TensorFlow Graph 중첩 오류를 방지하기 위해 한 번에 하나의 보컬 분리 작업만 실행합니다.
    async with _separation_lock:
        separation_success = False
        separation_error = None

        try:
            # 별도 스레드에서 실행하여 메인 이벤트 루프와 TensorFlow 컨텍스트 격리
            def _separate_audio():
                # spleeter가 자체적으로 TensorFlow Graph를 관리하므로
                # 우리가 명시적으로 Graph 컨텍스트를 만들지 않습니다.
                # 락으로 동시 실행을 제한하여 충돌을 방지합니다.
                try:
                    # 각 작업마다 새로운 Separator 인스턴스를 생성하여
                    # 이전 작업의 상태가 영향을 주지 않도록 합니다.
                    separator = Separator("spleeter:2stems")
                    separator.separate_to_file(input_audio_path, output_dir)
                except Exception as e:
                    logger.error(
                        f"보컬 분리 실행 중 오류 | input={input_audio_path} | error={e}",
                        exc_info=True,
                    )
                    raise

            # 전용 스레드 풀 사용: max_workers=1로 설정하여 한 번에 하나의 작업만 실행
            # TensorFlow Graph 충돌을 완전히 방지하기 위해 스레드 레벨에서도 제한합니다.
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(_separation_executor, _separate_audio)
            separation_success = True

        except Exception as e:
            separation_error = e
            logger.warning(
                f"보컬 분리 실행 중 예외 발생 (출력 파일 확인 예정) | input={input_audio_path} | error={e}"
            )
            # TensorFlow 오류가 발생해도 실제로 파일이 생성되었는지 확인합니다.

    # 출력 파일 검증: 실제 성공 여부는 파일 생성 여부로 판단
    # TensorFlow 오류 메시지가 나와도 파일이 정상적으로 생성되었으면 성공으로 처리합니다.
    if vocals_path.exists() and instrumental_path.exists():
        # 빈 파일 체크: 파일이 생성되었지만 비어있을 수 있음
        if vocals_path.stat().st_size > 0 and instrumental_path.stat().st_size > 0:
            logger.info(
                f"보컬 분리 성공 | input={input_audio_path} | "
                f"vocals={vocals_path} ({vocals_path.stat().st_size} bytes) | "
                f"instrumental={instrumental_path} ({instrumental_path.stat().st_size} bytes)"
            )
            return {"vocals": str(vocals_path), "instrumental": str(instrumental_path)}
        else:
            error_msg = (
                f"보컬 분리 출력 파일이 비어있습니다: "
                f"vocals={vocals_path.stat().st_size} bytes, "
                f"instrumental={instrumental_path.stat().st_size} bytes"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    else:
        # 출력 파일이 생성되지 않았으면 실패
        error_msg = (
            f"보컬 분리 출력 파일이 생성되지 않았습니다: "
            f"vocals={vocals_path.exists()}, instrumental={instrumental_path.exists()}"
        )
        if separation_error:
            error_msg += f" | 원인: {str(separation_error)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)


async def merge_vocal_instrumental(
    vocals_path: str, instrumental_path: str, output_path: str
) -> str:
    """변환된 보컬과 원본 인스트루멘탈 합성

    변환된 보컬과 원본 인스트루멘탈을 믹싱하여 최종 오디오를 생성합니다.
    샘플링 레이트가 다르면 보컬 기준으로 리샘플링하고, 길이가 다르면 패딩하여 맞춥니다.
    """
    loop = asyncio.get_running_loop()

    def _merge_audio():
        vocals, sr_v = librosa.load(vocals_path, sr=None, mono=True)
        instrumental, sr_i = librosa.load(instrumental_path, sr=None, mono=True)

        # 샘플링 레이트가 다르면 보컬 기준으로 리샘플링
        if sr_v != sr_i:
            instrumental = librosa.resample(instrumental, orig_sr=sr_i, target_sr=sr_v)
            sr_i = sr_v

        # 길이가 다르면 더 긴 쪽에 맞춰 패딩
        max_len = max(len(vocals), len(instrumental))
        vocals = np.pad(vocals, (0, max_len - len(vocals)), "constant")
        instrumental = np.pad(
            instrumental, (0, max_len - len(instrumental)), "constant"
        )

        # 단순 덧셈으로 믹싱
        mixed = vocals + instrumental

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, mixed, sr_v)
        return output_path

    return await loop.run_in_executor(None, _merge_audio)
