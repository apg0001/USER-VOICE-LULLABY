import os
import sys
import time
from scipy import signal
from scipy.io import wavfile
import numpy as np
import concurrent.futures
from tqdm import tqdm
import json
from distutils.util import strtobool
import librosa
import multiprocessing
import noisereduce as nr
import soxr

now_directory = os.getcwd()
sys.path.append(now_directory)

from applio.rvc.lib.utils import load_audio
from applio.rvc.train.preprocess.slicer import Slicer

import logging

logging.getLogger("numba.core.byteflow").setLevel(logging.WARNING)
logging.getLogger("numba.core.ssa").setLevel(logging.WARNING)
logging.getLogger("numba.core.interpreter").setLevel(logging.WARNING)

OVERLAP = 0.3
PERCENTAGE = 3.0
MAX_AMPLITUDE = 0.9
ALPHA = 0.75
HIGH_PASS_CUTOFF = 48
SAMPLE_RATE_16K = 16000
RES_TYPE = "soxr_vhq"


class PreProcess:
    def __init__(self, sr: int, exp_dir: str):
        self.slicer = Slicer(
            sr=sr,
            threshold=-42,
            min_length=1500,
            min_interval=400,
            hop_size=15,
            max_sil_kept=500,
        )
        self.sr = sr
        self.b_high, self.a_high = signal.butter(
            N=5, Wn=HIGH_PASS_CUTOFF, btype="high", fs=self.sr
        )
        self.exp_dir = exp_dir
        self.device = "cpu"
        self.gt_wavs_dir = os.path.join(exp_dir, "sliced_audios")
        self.wavs16k_dir = os.path.join(exp_dir, "sliced_audios_16k")
        os.makedirs(self.gt_wavs_dir, exist_ok=True)
        os.makedirs(self.wavs16k_dir, exist_ok=True)

    def _normalize_audio(self, audio: np.ndarray):
        tmp_max = np.abs(audio).max()
        if tmp_max > 2.5:
            return None
        return (audio / tmp_max * (MAX_AMPLITUDE * ALPHA)) + (1 - ALPHA) * audio

    def process_audio_segment(
        self,
        normalized_audio: np.ndarray,
        sid: int,
        idx0: int,
        idx1: int,
        normalization_mode: str,
        enable_augmentation: bool = False,
        speed_perturbation: list = None,
        volume_augmentation: list = None,
    ):
        if normalized_audio is None:
            print(f"[DEBUG] {sid}-{idx0}-{idx1}-filtered")
            return 0  # 생성된 증강 파일 수 반환
        
        if normalization_mode == "post":
            normalized_audio = self._normalize_audio(normalized_audio)
        
        # 원본 오디오 저장
        original_filename = f"{sid}_{idx0}_{idx1}.wav"
        print(f"[DEBUG] 원본 파일 저장: {original_filename}")
        wavfile.write(
            os.path.join(self.gt_wavs_dir, original_filename),
            self.sr,
            normalized_audio.astype(np.float32),
        )
        audio_16k = librosa.resample(
            normalized_audio,
            orig_sr=self.sr,
            target_sr=SAMPLE_RATE_16K,
            res_type=RES_TYPE,
        )
        wavfile.write(
            os.path.join(self.wavs16k_dir, original_filename),
            SAMPLE_RATE_16K,
            audio_16k.astype(np.float32),
        )
        
        # 데이터 증강이 활성화된 경우에만 증강된 버전 생성
        aug_count = 0  # 생성된 증강 파일 수
        if enable_augmentation:
            aug_idx = idx1 + 1  # idx1 다음부터 시작하여 겹침 방지
            print(f"[DEBUG] 데이터 증강 시작: 원본 idx1={idx1}, 증강 시작 aug_idx={aug_idx}")
            
            # Speed perturbation (0.9, 1.1)
            if speed_perturbation:
                print(f"[DEBUG] Speed perturbation 적용: {speed_perturbation}")
                for speed_factor in speed_perturbation:
                    # librosa의 time_stretch는 시간을 늘리거나 줄이지만, 
                    # speed perturbation은 pitch를 유지하면서 속도만 변경
                    # librosa.effects.time_stretch는 실제로는 pitch도 변경하므로
                    # librosa의 resample을 사용하여 속도만 변경
                    # 하지만 더 정확한 방법은 librosa.effects.time_stretch를 사용
                    augmented_audio = librosa.effects.time_stretch(normalized_audio, rate=speed_factor)
                    aug_filename = f"{sid}_{idx0}_{aug_idx}.wav"
                    print(f"[DEBUG] Speed perturbation 파일 저장: {aug_filename} (factor={speed_factor})")
                    
                    wavfile.write(
                        os.path.join(self.gt_wavs_dir, aug_filename),
                        self.sr,
                        augmented_audio.astype(np.float32),
                    )
                    augmented_audio_16k = librosa.resample(
                        augmented_audio,
                        orig_sr=self.sr,
                        target_sr=SAMPLE_RATE_16K,
                        res_type=RES_TYPE,
                    )
                    wavfile.write(
                        os.path.join(self.wavs16k_dir, aug_filename),
                        SAMPLE_RATE_16K,
                        augmented_audio_16k.astype(np.float32),
                    )
                    aug_idx += 1
                    aug_count += 1
            
            # Volume augmentation (0.9, 1.1)
            if volume_augmentation:
                print(f"[DEBUG] Volume augmentation 적용: {volume_augmentation}")
                for volume_factor in volume_augmentation:
                    augmented_audio = normalized_audio * volume_factor
                    # 클리핑 방지
                    augmented_audio = np.clip(augmented_audio, -1.0, 1.0)
                    aug_filename = f"{sid}_{idx0}_{aug_idx}.wav"
                    print(f"[DEBUG] Volume augmentation 파일 저장: {aug_filename} (factor={volume_factor})")
                    
                    wavfile.write(
                        os.path.join(self.gt_wavs_dir, aug_filename),
                        self.sr,
                        augmented_audio.astype(np.float32),
                    )
                    augmented_audio_16k = librosa.resample(
                        augmented_audio,
                        orig_sr=self.sr,
                        target_sr=SAMPLE_RATE_16K,
                        res_type=RES_TYPE,
                    )
                    wavfile.write(
                        os.path.join(self.wavs16k_dir, aug_filename),
                        SAMPLE_RATE_16K,
                        augmented_audio_16k.astype(np.float32),
                    )
                    aug_idx += 1
                    aug_count += 1
            
            print(f"[DEBUG] 증강 완료: 총 {aug_count}개 파일 생성 (원본 idx1={idx1}, 마지막 aug_idx={aug_idx-1})")
        else:
            print(f"[DEBUG] 데이터 증강 비활성화: aug_count=0")
        
        return aug_count  # 생성된 증강 파일 수 반환

    def simple_cut(
        self,
        audio: np.ndarray,
        sid: int,
        idx0: int,
        chunk_len: float,
        overlap_len: float,
        normalization_mode: str,
        enable_augmentation: bool = False,
        speed_perturbation: list = None,
        volume_augmentation: list = None,
    ):
        chunk_length = int(self.sr * chunk_len)
        overlap_length = int(self.sr * overlap_len)
        i = 0
        chunk_idx = 0
        print(f"[DEBUG] Simple cut 시작: sid={sid}, idx0={idx0}, chunk_len={chunk_len}, overlap_len={overlap_len}")
        while i < len(audio):
            chunk = audio[i : i + chunk_length]
            if normalization_mode == "post":
                chunk = self._normalize_audio(chunk)
            if len(chunk) == chunk_length:
                # 원본 청크 저장
                print(f"[DEBUG] Simple cut 처리: chunk_idx={chunk_idx}")
                aug_count = self.process_audio_segment(
                    chunk,
                    sid,
                    idx0,
                    chunk_idx,
                    normalization_mode,
                    enable_augmentation,
                    speed_perturbation,
                    volume_augmentation,
                )
                # 원본 1개 + 증강 파일 수만큼 증가
                old_chunk_idx = chunk_idx
                chunk_idx += 1 + aug_count
                print(f"[DEBUG] Simple cut idx 업데이트: {old_chunk_idx} -> {chunk_idx} (증강 {aug_count}개 포함)")
            i += chunk_length - overlap_length

    def process_audio(
        self,
        path: str,
        idx0: int,
        sid: int,
        cut_preprocess: str,
        process_effects: bool,
        noise_reduction: bool,
        reduction_strength: float,
        chunk_len: float,
        overlap_len: float,
        normalization_mode: str,
        enable_augmentation: bool = False,
        speed_perturbation: list = None,
        volume_augmentation: list = None,
    ):
        audio_length = 0
        try:
            audio = load_audio(path, self.sr)
            audio_length = librosa.get_duration(y=audio, sr=self.sr)

            if process_effects:
                audio = signal.lfilter(self.b_high, self.a_high, audio)
            if normalization_mode == "pre":
                audio = self._normalize_audio(audio)
            if noise_reduction:
                audio = nr.reduce_noise(
                    y=audio, sr=self.sr, prop_decrease=reduction_strength
                )
            if cut_preprocess == "Skip":
                # no cutting
                aug_count = self.process_audio_segment(
                    audio,
                    sid,
                    idx0,
                    0,
                    normalization_mode,
                    enable_augmentation,
                    speed_perturbation,
                    volume_augmentation,
                )
                # Skip 모드에서는 단일 파일이므로 idx1 증가 불필요
            elif cut_preprocess == "Simple":
                # simple
                self.simple_cut(
                    audio,
                    sid,
                    idx0,
                    chunk_len,
                    overlap_len,
                    normalization_mode,
                    enable_augmentation,
                    speed_perturbation,
                    volume_augmentation,
                )
            elif cut_preprocess == "Automatic":
                idx1 = 0
                print(f"[DEBUG] Automatic cut 시작: sid={sid}, idx0={idx0}, 초기 idx1={idx1}")
                # legacy
                segment_count = 0
                for audio_segment in self.slicer.slice(audio):
                    segment_count += 1
                    print(f"[DEBUG] Automatic cut 세그먼트 {segment_count} 처리 시작: 현재 idx1={idx1}")
                    i = 0
                    while True:
                        start = int(self.sr * (PERCENTAGE - OVERLAP) * i)
                        i += 1
                        if (
                            len(audio_segment[start:])
                            > (PERCENTAGE + OVERLAP) * self.sr
                        ):
                            tmp_audio = audio_segment[
                                start : start + int(PERCENTAGE * self.sr)
                            ]
                            print(f"[DEBUG] Automatic cut 청크 처리: idx1={idx1}")
                            aug_count = self.process_audio_segment(
                                tmp_audio,
                                sid,
                                idx0,
                                idx1,
                                normalization_mode,
                                enable_augmentation,
                                speed_perturbation,
                                volume_augmentation,
                            )
                            # 원본 1개 + 증강 파일 수만큼 증가
                            old_idx1 = idx1
                            idx1 += 1 + aug_count
                            print(f"[DEBUG] Automatic cut idx 업데이트: {old_idx1} -> {idx1} (증강 {aug_count}개 포함)")
                        else:
                            tmp_audio = audio_segment[start:]
                            print(f"[DEBUG] Automatic cut 마지막 청크 처리: idx1={idx1}")
                            aug_count = self.process_audio_segment(
                                tmp_audio,
                                sid,
                                idx0,
                                idx1,
                                normalization_mode,
                                enable_augmentation,
                                speed_perturbation,
                                volume_augmentation,
                            )
                            # 원본 1개 + 증강 파일 수만큼 증가
                            old_idx1 = idx1
                            idx1 += 1 + aug_count
                            print(f"[DEBUG] Automatic cut idx 업데이트: {old_idx1} -> {idx1} (증강 {aug_count}개 포함)")
                            break
                print(f"[DEBUG] Automatic cut 완료: 총 {segment_count}개 세그먼트, 최종 idx1={idx1}")

        except Exception as error:
            print(f"Error processing audio: {error}")
        return audio_length


def format_duration(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def save_dataset_duration(file_path, dataset_duration):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {}

    formatted_duration = format_duration(dataset_duration)
    new_data = {
        "total_dataset_duration": formatted_duration,
        "total_seconds": dataset_duration,
    }
    data.update(new_data)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def process_audio_wrapper(args):
    (
        pp,
        file,
        cut_preprocess,
        process_effects,
        noise_reduction,
        reduction_strength,
        chunk_len,
        overlap_len,
        normalization_mode,
        enable_augmentation,
        speed_perturbation,
        volume_augmentation,
    ) = args
    file_path, idx0, sid = file
    return pp.process_audio(
        file_path,
        idx0,
        sid,
        cut_preprocess,
        process_effects,
        noise_reduction,
        reduction_strength,
        chunk_len,
        overlap_len,
        normalization_mode,
        enable_augmentation,
        speed_perturbation,
        volume_augmentation,
    )


def preprocess_training_set(
    input_root: str,
    sr: int,
    num_processes: int,
    exp_dir: str,
    cut_preprocess: str,
    process_effects: bool,
    noise_reduction: bool,
    reduction_strength: float,
    chunk_len: float,
    overlap_len: float,
    normalization_mode: str,
    enable_augmentation: bool = False,
    speed_perturbation: list = None,
    volume_augmentation: list = None,
):
    start_time = time.time()
    pp = PreProcess(sr, exp_dir)
    print(f"Starting preprocess with {num_processes} processes...")
    print(f"{exp_dir}")

    files = []
    idx = 0

    for root, _, filenames in os.walk(input_root):
        try:
            sid = 0 if root == input_root else int(os.path.basename(root))
            for f in filenames:
                if f.lower().endswith((".wav", ".mp3", ".flac", ".ogg")):
                    files.append((os.path.join(root, f), idx, sid))
                    idx += 1
        except ValueError:
            print(
                f'Speaker ID folder is expected to be integer, got "{os.path.basename(root)}" instead.'
            )

    # print(f"Number of files: {len(files)}")
    audio_length = []
    with tqdm(total=len(files)) as pbar:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_processes
        ) as executor:
            futures = [
                executor.submit(
                    process_audio_wrapper,
                    (
                        pp,
                        file,
                        cut_preprocess,
                        process_effects,
                        noise_reduction,
                        reduction_strength,
                        chunk_len,
                        overlap_len,
                        normalization_mode,
                        enable_augmentation,
                        speed_perturbation,
                        volume_augmentation,
                    ),
                )
                for file in files
            ]
            for future in concurrent.futures.as_completed(futures):
                audio_length.append(future.result())
                pbar.update(1)

    audio_length = sum(audio_length)
    save_dataset_duration(
        os.path.join(exp_dir, "model_info.json"), dataset_duration=audio_length
    )
    elapsed_time = time.time() - start_time
    print(
        f"Preprocess completed in {elapsed_time:.2f} seconds on {format_duration(audio_length)} seconds of audio."
    )


if __name__ == "__main__":
    experiment_directory = str(sys.argv[1])
    input_root = str(sys.argv[2])
    sample_rate = int(sys.argv[3])
    num_processes = sys.argv[4]
    if num_processes.lower() == "none":
        num_processes = multiprocessing.cpu_count()
    else:
        num_processes = int(num_processes)
    cut_preprocess = str(sys.argv[5])
    process_effects = strtobool(sys.argv[6])
    noise_reduction = strtobool(sys.argv[7])
    reduction_strength = float(sys.argv[8])
    chunk_len = float(sys.argv[9])
    overlap_len = float(sys.argv[10])
    normalization_mode = str(sys.argv[11])
    enable_augmentation = strtobool(sys.argv[12]) if len(sys.argv) > 12 else False
    speed_perturbation = json.loads(sys.argv[13]) if len(sys.argv) > 13 and sys.argv[13] else None
    volume_augmentation = json.loads(sys.argv[14]) if len(sys.argv) > 14 and sys.argv[14] else None
    preprocess_training_set(
        input_root,
        sample_rate,
        num_processes,
        experiment_directory,
        cut_preprocess,
        process_effects,
        noise_reduction,
        reduction_strength,
        chunk_len,
        overlap_len,
        normalization_mode,
        enable_augmentation,
        speed_perturbation,
        volume_augmentation,
    )
