from __future__ import annotations

import os
from dataclasses import dataclass


def _cpu_cores(default: int = 2) -> int:
    """
    Return a conservative estimate of usable CPU cores.
    """
    cores = os.cpu_count() or default
    return max(1, cores - 1)


@dataclass(frozen=True)
class TrainingDefaults:
    sample_rate: int = 40000
    batch_size: int = 4
    total_epoch: int = 200
    save_every_epoch: int = 10
    save_only_latest: bool = True
    save_every_weights: bool = (
        False  # 위 설정한 epoch 수 진행할 때마다 모델 저장 / 배포시 False
    )
    cpu_cores: int = _cpu_cores()
    cut_preprocess: str = "Automatic"
    process_effects: bool = False
    noise_reduction: bool = False
    clean_strength: float = 0.6
    chunk_len: float = 20.0
    overlap_len: float = 2.0
    normalization_mode: str = "pre"
    f0_method: str = "rmvpe"
    embedder_model: str = "contentvec"
    gpu: int = 0
    include_mutes: int = 2
    overtraining_detector: bool = (
        True  # 과적합 방지(일정 epoch 동안 학습이 되지 않으면 학습 종료)
    )
    overtraining_threshold: int = 25  # epoch 수
    pretrained: bool = True
    cleanup: bool = False
    cache_data_in_gpu: bool = False
    index_algorithm: str = "Auto"
    custom_pretrained: bool = True  # 기본값을 True로 변경
    g_pretrained_path: str = None  # 기본값은 None, UI에서 자동 설정
    d_pretrained_path: str = None  # 기본값은 None, UI에서 자동 설정
    vocoder: str = "HiFi-GAN"
    checkpointing: bool = False
    enable_augmentation: bool = False
    speed_perturbation: list = None
    volume_augmentation: list = None


@dataclass(frozen=True)
class InferenceDefaults:
    # pitch: int = 0
    # index_rate: float = 0.3
    # volume_envelope: float = 1.0
    # protect: float = 0.33
    # f0_method: str = "rmvpe"
    # split_audio: bool = False
    # f0_autotune: bool = False
    # f0_autotune_strength: float = 0.5
    # proposed_pitch: bool = False
    # proposed_pitch_threshold: float = 0.5
    # clean_audio: bool = False
    # clean_strength: float = 0.7
    # export_format: str = "wav"
    # embedder_model: str = "contentvec"
    # formant_shifting: bool = False
    # formant_qfrency: float = 1.0
    # formant_timbre: float = 1.0
    # post_process: bool = False
    pitch: int = 0
    index_rate: float = 0.75
    volume_envelope: float = 1.0
    protect: float = 0.5
    f0_method: str = "rmvpe"
    split_audio: bool = False
    f0_autotune: bool = False
    f0_autotune_strength: float = 1
    proposed_pitch: bool = False
    proposed_pitch_threshold: float = 155.0
    clean_audio: bool = False
    clean_strength: float = 0.5
    export_format: str = "wav"
    embedder_model: str = "contentvec"
    formant_shifting: bool = False
    formant_qfrency: float = 1.0
    formant_timbre: float = 1.0
    post_process: bool = False
    reverb: bool = False
    reverb_room_size: float = 0.5
    reverb_damping: float = 0.5
    reverb_wet_gain: float = 0.5
    reverb_dry_gain: float = 0.5
    reverb_width: float = 0.5
    reverb_freeze_mode: float = 0.5


TRAINING_DEFAULTS = TrainingDefaults()
INFERENCE_DEFAULTS = InferenceDefaults()

# 큐 작업 내역 보관 기간 (일)
JOB_RETENTION_DAYS = int(os.getenv("JOB_RETENTION_DAYS", "7"))  # 기본값: 7일
