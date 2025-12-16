import os
import sys
import soxr
import time
import torch
import librosa
import logging
import traceback
import numpy as np
import soundfile as sf
import noisereduce as nr
from pedalboard import (
    Pedalboard,
    Chorus,
    Distortion,
    Reverb,
    PitchShift,
    Limiter,
    Gain,
    Bitcrush,
    Clipping,
    Compressor,
    Delay,
)

now_dir = os.getcwd()
sys.path.append(now_dir)

from applio.rvc.infer.pipeline import Pipeline as VC
from applio.rvc.lib.utils import load_audio_infer, load_embedding
from applio.rvc.lib.tools.split_audio import process_audio, merge_audio
from applio.rvc.lib.algorithm.synthesizers import Synthesizer
from applio.rvc.configs.config import Config

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("faiss").setLevel(logging.WARNING)
logging.getLogger("faiss.loader").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


class VoiceConverter:
    """
    A class for performing voice conversion using the Retrieval-Based Voice Conversion (RVC) method.
    """

    def __init__(self):
        """
        Initializes the VoiceConverter with default configuration, and sets up models and parameters.
        """
        self.config = Config()  # Load configuration
        self.hubert_model = (
            None  # Initialize the Hubert model (for embedding extraction)
        )
        self.last_embedder_model = None  # Last used embedder model
        self.tgt_sr = None  # Target sampling rate for the output audio
        self.net_g = None  # Generator network for voice conversion
        self.vc = None  # Voice conversion pipeline instance
        self.cpt = None  # Checkpoint for loading model weights
        self.version = None  # Model version
        self.n_spk = None  # Number of speakers in the model
        self.use_f0 = None  # Whether the model uses F0
        self.loaded_model = None

    def load_hubert(self, embedder_model: str, embedder_model_custom: str = None):
        """
        Loads the HuBERT model for speaker embedding extraction.

        Args:
            embedder_model (str): Path to the pre-trained HuBERT model.
            embedder_model_custom (str): Path to the custom HuBERT model.
        """
        self.hubert_model = load_embedding(embedder_model, embedder_model_custom)
        self.hubert_model = self.hubert_model.to(self.config.device).float()
        self.hubert_model.eval()

    @staticmethod
    def remove_audio_noise(data, sr, reduction_strength=0.7):
        """
        Removes noise from an audio file using the NoiseReduce library.

        Args:
            data (numpy.ndarray): The audio data as a NumPy array.
            sr (int): The sample rate of the audio data.
            reduction_strength (float): Strength of the noise reduction. Default is 0.7.
        """
        try:
            reduced_noise = nr.reduce_noise(
                y=data, sr=sr, prop_decrease=reduction_strength
            )
            return reduced_noise
        except Exception as error:
            logger.error(f"오디오 노이즈 제거 중 오류 발생: {error}", exc_info=True)
            return None

    @staticmethod
    def convert_audio_format(input_path, output_path, output_format):
        """
        Converts an audio file to a specified output format.

        Args:
            input_path (str): Path to the input audio file.
            output_path (str): Path to the output audio file.
            output_format (str): Desired audio format (e.g., "WAV", "MP3").
        """
        try:
            if output_format != "WAV":
                audio, sample_rate = librosa.load(input_path, sr=None)
                common_sample_rates = [
                    8000,
                    11025,
                    12000,
                    16000,
                    22050,
                    24000,
                    32000,
                    44100,
                    48000,
                ]
                target_sr = min(common_sample_rates, key=lambda x: abs(x - sample_rate))
                audio = librosa.resample(
                    audio, orig_sr=sample_rate, target_sr=target_sr, res_type="soxr_vhq"
                )
                sf.write(output_path, audio, target_sr, format=output_format.lower())
            return output_path
        except Exception as error:
            logger.error(f"오디오 형식 변환 중 오류 발생: {error}", exc_info=True)
            return None

    @staticmethod
    def post_process_audio(
        audio_input,
        sample_rate,
        **kwargs,
    ):
        board = Pedalboard()
        if kwargs.get("reverb", False):
            reverb = Reverb(
                room_size=kwargs.get("reverb_room_size", 0.5),
                damping=kwargs.get("reverb_damping", 0.5),
                wet_level=kwargs.get("reverb_wet_level", 0.33),
                dry_level=kwargs.get("reverb_dry_level", 0.4),
                width=kwargs.get("reverb_width", 1.0),
                freeze_mode=kwargs.get("reverb_freeze_mode", 0),
            )
            board.append(reverb)
        if kwargs.get("pitch_shift", False):
            pitch_shift = PitchShift(semitones=kwargs.get("pitch_shift_semitones", 0))
            board.append(pitch_shift)
        if kwargs.get("limiter", False):
            limiter = Limiter(
                threshold_db=kwargs.get("limiter_threshold", -6),
                release_ms=kwargs.get("limiter_release", 0.05),
            )
            board.append(limiter)
        if kwargs.get("gain", False):
            gain = Gain(gain_db=kwargs.get("gain_db", 0))
            board.append(gain)
        if kwargs.get("distortion", False):
            distortion = Distortion(drive_db=kwargs.get("distortion_gain", 25))
            board.append(distortion)
        if kwargs.get("chorus", False):
            chorus = Chorus(
                rate_hz=kwargs.get("chorus_rate", 1.0),
                depth=kwargs.get("chorus_depth", 0.25),
                centre_delay_ms=kwargs.get("chorus_delay", 7),
                feedback=kwargs.get("chorus_feedback", 0.0),
                mix=kwargs.get("chorus_mix", 0.5),
            )
            board.append(chorus)
        if kwargs.get("bitcrush", False):
            bitcrush = Bitcrush(bit_depth=kwargs.get("bitcrush_bit_depth", 8))
            board.append(bitcrush)
        if kwargs.get("clipping", False):
            clipping = Clipping(threshold_db=kwargs.get("clipping_threshold", 0))
            board.append(clipping)
        if kwargs.get("compressor", False):
            compressor = Compressor(
                threshold_db=kwargs.get("compressor_threshold", 0),
                ratio=kwargs.get("compressor_ratio", 1),
                attack_ms=kwargs.get("compressor_attack", 1.0),
                release_ms=kwargs.get("compressor_release", 100),
            )
            board.append(compressor)
        if kwargs.get("delay", False):
            delay = Delay(
                delay_seconds=kwargs.get("delay_seconds", 0.5),
                feedback=kwargs.get("delay_feedback", 0.0),
                mix=kwargs.get("delay_mix", 0.5),
            )
            board.append(delay)
        return board(audio_input, sample_rate)

    def convert_audio(
        self,
        audio_input_path: str,
        audio_output_path: str,
        model_path: str,
        index_path: str,
        pitch: int = 0,
        f0_method: str = "rmvpe",
        index_rate: float = 0.75,
        volume_envelope: float = 1.0,
        protect: float = 0.5,
        hop_length: int = 128,
        split_audio: bool = False,
        f0_autotune: bool = False,
        f0_autotune_strength: float = 1,
        embedder_model: str = "contentvec",
        embedder_model_custom: str = None,
        clean_audio: bool = False,
        clean_strength: float = 0.5,
        export_format: str = "WAV",
        post_process: bool = False,
        resample_sr: int = 0,
        sid: int = 0,
        proposed_pitch: bool = False,
        proposed_pitch_threshold: float = 155.0,
        **kwargs,
    ):
        """
        Performs voice conversion on the input audio.

        Args:
            pitch (int): Key for F0 up-sampling.
            index_rate (float): Rate for index matching.
            volume_envelope (int): RMS mix rate.
            protect (float): Protection rate for certain audio segments.
            hop_length (int): Hop length for audio processing.
            f0_method (str): Method for F0 extraction.
            audio_input_path (str): Path to the input audio file.
            audio_output_path (str): Path to the output audio file.
            model_path (str): Path to the voice conversion model.
            index_path (str): Path to the index file.
            split_audio (bool): Whether to split the audio for processing.
            f0_autotune (bool): Whether to use F0 autotune.
            clean_audio (bool): Whether to clean the audio.
            clean_strength (float): Strength of the audio cleaning.
            export_format (str): Format for exporting the audio.
            f0_file (str): Path to the F0 file.
            embedder_model (str): Path to the embedder model.
            embedder_model_custom (str): Path to the custom embedder model.
            resample_sr (int, optional): Resample sampling rate. Default is 0.
            sid (int, optional): Speaker ID. Default is 0.
            **kwargs: Additional keyword arguments.
        """
        if not model_path:
            logger.error("모델 경로가 제공되지 않았습니다. 변환을 중단합니다.")
            return

        self.get_vc(model_path, sid)

        try:
            start_time = time.time()

            audio = load_audio_infer(
                audio_input_path,
                16000,
                **kwargs,
            )
            audio_max = np.abs(audio).max() / 0.95

            if audio_max > 1:
                audio /= audio_max

            if not self.hubert_model or embedder_model != self.last_embedder_model:
                self.load_hubert(embedder_model, embedder_model_custom)
                self.last_embedder_model = embedder_model

            file_index = (
                index_path.strip()
                .strip('"')
                .strip("\n")
                .strip('"')
                .strip()
                .replace("trained", "added")
            )

            if self.tgt_sr != resample_sr >= 16000:
                self.tgt_sr = resample_sr

            if split_audio:
                chunks, intervals = process_audio(audio, 16000)
            else:
                chunks = []
                chunks.append(audio)

            converted_chunks = []
            for c in chunks:
                audio_opt = self.vc.pipeline(
                    model=self.hubert_model,
                    net_g=self.net_g,
                    sid=sid,
                    audio=c,
                    pitch=pitch,
                    f0_method=f0_method,
                    file_index=file_index,
                    index_rate=index_rate,
                    pitch_guidance=self.use_f0,
                    volume_envelope=volume_envelope,
                    version=self.version,
                    protect=protect,
                    f0_autotune=f0_autotune,
                    f0_autotune_strength=f0_autotune_strength,
                    proposed_pitch=proposed_pitch,
                    proposed_pitch_threshold=proposed_pitch_threshold,
                )
                converted_chunks.append(audio_opt)

            if split_audio:
                audio_opt = merge_audio(
                    chunks, converted_chunks, intervals, 16000, self.tgt_sr
                )
            else:
                audio_opt = converted_chunks[0]

            if clean_audio:
                cleaned_audio = self.remove_audio_noise(
                    audio_opt, self.tgt_sr, clean_strength
                )
                if cleaned_audio is not None:
                    audio_opt = cleaned_audio

            if post_process:
                audio_opt = self.post_process_audio(
                    audio_input=audio_opt,
                    sample_rate=self.tgt_sr,
                    **kwargs,
                )

            sf.write(audio_output_path, audio_opt, self.tgt_sr, format="WAV")
            output_path_format = audio_output_path.replace(
                ".wav", f".{export_format.lower()}"
            )
            audio_output_path = self.convert_audio_format(
                audio_output_path, output_path_format, export_format
            )
        except Exception as error:
            logger.error(f"오디오 변환 중 오류 발생: {error}", exc_info=True)
            raise
        finally:
            # 작업 완료 후 메모리 정리
            # 오디오 데이터 해제 (CPU 메모리 해제)
            try:
                if 'audio' in locals():
                    if isinstance(audio, np.ndarray):
                        audio = None
                    del audio
                if 'audio_opt' in locals():
                    if isinstance(audio_opt, np.ndarray):
                        audio_opt = None
                    del audio_opt
                if 'chunks' in locals():
                    if isinstance(chunks, list):
                        for chunk in chunks:
                            if chunk is not None and isinstance(chunk, np.ndarray):
                                chunk = None
                            del chunk
                    del chunks
                if 'converted_chunks' in locals():
                    if isinstance(converted_chunks, list):
                        for chunk in converted_chunks:
                            if chunk is not None and isinstance(chunk, np.ndarray):
                                chunk = None
                            del chunk
                    del converted_chunks
                if 'cleaned_audio' in locals():
                    if isinstance(cleaned_audio, np.ndarray):
                        cleaned_audio = None
                    del cleaned_audio
            except Exception:
                pass
            
            # GPU 메모리 정리
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except Exception:
                pass
            
            # 가비지 컬렉션
            try:
                import gc
                gc.collect()
            except Exception:
                pass

    def convert_audio_batch(
        self,
        audio_input_paths: str,
        audio_output_path: str,
        **kwargs,
    ):
        """
        Performs voice conversion on a batch of input audio files.

        Args:
            audio_input_paths (str): List of paths to the input audio files.
            audio_output_path (str): Path to the output audio file.
            resample_sr (int, optional): Resample sampling rate. Default is 0.
            sid (int, optional): Speaker ID. Default is 0.
            **kwargs: Additional keyword arguments.
        """
        pid = os.getpid()
        try:
            with open(
                os.path.join(now_dir, "assets", "infer_pid.txt"), "w"
            ) as pid_file:
                pid_file.write(str(pid))
            start_time = time.time()
            audio_files = [
                f
                for f in os.listdir(audio_input_paths)
                if f.lower().endswith(
                    (
                        "wav",
                        "mp3",
                        "flac",
                        "ogg",
                        "opus",
                        "m4a",
                        "mp4",
                        "aac",
                        "alac",
                        "wma",
                        "aiff",
                        "webm",
                        "ac3",
                    )
                )
            ]
            for a in audio_files:
                new_input = os.path.join(audio_input_paths, a)
                new_output = os.path.splitext(a)[0] + "_output.wav"
                new_output = os.path.join(audio_output_path, new_output)
                if os.path.exists(new_output):
                    continue
                self.convert_audio(
                    audio_input_path=new_input,
                    audio_output_path=new_output,
                    **kwargs,
                )
        except Exception as error:
            logger.error(f"배치 오디오 변환 중 오류 발생: {error}", exc_info=True)
            raise
        finally:
            os.remove(os.path.join(now_dir, "assets", "infer_pid.txt"))

    def get_vc(self, weight_root, sid):
        """
        Loads the voice conversion model and sets up the pipeline.

        Args:
            weight_root (str): Path to the model weights.
            sid (int): Speaker ID.
        """
        if sid == "" or sid == []:
            self.cleanup_model()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 다른 모델로 전환할 때 이전 모델 정리
        if self.loaded_model and self.loaded_model != weight_root:
            # 이전 모델이 다른 경우 정리
            self.cleanup_model()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()

        if not self.loaded_model or self.loaded_model != weight_root:
            self.load_model(weight_root)
            if self.cpt is not None:
                self.setup_network()
                self.setup_vc_instance()
                self.loaded_model = weight_root
            else:
                self.vc = None
                self.loaded_model = None

    def cleanup_model(self):
        """
        Cleans up the model and releases resources.
        메모리 조회는 외부에서 하도록 하여 cleanup 과정에서 메모리 증가를 방지합니다.
        """
        try:
            # 1. GPU 메모리 즉시 정리 (모델 삭제 전)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # 2. 모델 인스턴스 직접 삭제 (CPU 이동 없이 - 메모리 복사 방지)
            # state_dict() 호출을 피하여 메모리 복사를 방지합니다.
            if self.net_g is not None:
                try:
                    # 모델의 모든 파라미터와 버퍼를 직접 삭제
                    if hasattr(self.net_g, 'parameters'):
                        for param in self.net_g.parameters():
                            if param is not None:
                                param.data = None
                                if hasattr(param, 'grad') and param.grad is not None:
                                    param.grad = None
                    if hasattr(self.net_g, 'buffers'):
                        for buffer in self.net_g.buffers():
                            if buffer is not None:
                                buffer.data = None
                    # 모델 자체 삭제
                    del self.net_g
                    self.net_g = None
                except Exception as e:
                    logger.error(f"net_g 모델 삭제 중 오류 발생: {e}", exc_info=True)
                    self.net_g = None
            
            if self.vc is not None:
                try:
                    # Pipeline의 모든 속성 정리
                    if hasattr(self.vc, 'autotune') and self.vc.autotune is not None:
                        try:
                            del self.vc.autotune
                        except Exception:
                            pass
                    # Pipeline 내부의 다른 리소스 정리
                    for attr_name in ['f0_model', 'f0_predictor']:
                        if hasattr(self.vc, attr_name):
                            try:
                                attr = getattr(self.vc, attr_name)
                                if attr is not None:
                                    del attr
                                    setattr(self.vc, attr_name, None)
                            except Exception:
                                pass
                    del self.vc
                    self.vc = None
                except Exception as e:
                    logger.error(f"vc 파이프라인 삭제 중 오류 발생: {e}", exc_info=True)
                    self.vc = None
            
            if self.hubert_model is not None:
                try:
                    # Hubert 모델의 모든 파라미터와 버퍼를 직접 삭제
                    if hasattr(self.hubert_model, 'parameters'):
                        for param in self.hubert_model.parameters():
                            if param is not None:
                                param.data = None
                                if hasattr(param, 'grad') and param.grad is not None:
                                    param.grad = None
                    if hasattr(self.hubert_model, 'buffers'):
                        for buffer in self.hubert_model.buffers():
                            if buffer is not None:
                                buffer.data = None
                    # 모델 자체 삭제
                    del self.hubert_model
                    self.hubert_model = None
                except Exception as e:
                    logger.error(f"hubert_model 삭제 중 오류 발생: {e}", exc_info=True)
                    self.hubert_model = None
            
            # 3. 체크포인트 삭제
            if self.cpt is not None:
                try:
                    # 체크포인트 내부의 모든 텐서 삭제
                    if isinstance(self.cpt, dict):
                        for key in list(self.cpt.keys()):
                            if isinstance(self.cpt[key], dict):
                                for sub_key in list(self.cpt[key].keys()):
                                    if isinstance(self.cpt[key][sub_key], torch.Tensor):
                                        del self.cpt[key][sub_key]
                            elif isinstance(self.cpt[key], torch.Tensor):
                                del self.cpt[key]
                    del self.cpt
                    self.cpt = None
                except Exception:
                    self.cpt = None
            
            # 4. 기타 변수 정리
            self.n_spk = None
            self.tgt_sr = None
            self.version = None
            self.use_f0 = None
            self.loaded_model = None
            self.last_embedder_model = None
            
            # 5. GPU 메모리 최종 정리
            if torch.cuda.is_available():
                for _ in range(3):
                    torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # 6. 가비지 컬렉션 (여러 번 실행하여 확실히 정리)
            import gc
            for _ in range(3):
                gc.collect()
            
        except Exception as e:
            logger.error(f"모델 정리 중 오류 발생: {e}", exc_info=True)
            # 예외가 발생해도 최대한 정리 시도
            try:
                if self.net_g is not None:
                    del self.net_g
                    self.net_g = None
                if self.vc is not None:
                    del self.vc
                    self.vc = None
                if self.hubert_model is not None:
                    del self.hubert_model
                    self.hubert_model = None
                if self.cpt is not None:
                    del self.cpt
                    self.cpt = None
                import gc
                for _ in range(3):
                    gc.collect()
                if torch.cuda.is_available():
                    for _ in range(3):
                        torch.cuda.empty_cache()
            except Exception:
                pass

    def load_model(self, weight_root):
        """
        Loads the model weights from the specified path.

        Args:
            weight_root (str): Path to the model weights.
        """
        self.cpt = (
            torch.load(weight_root, map_location="cpu", weights_only=True)
            if os.path.isfile(weight_root)
            else None
        )

    def setup_network(self):
        """
        Sets up the network configuration based on the loaded checkpoint.
        """
        if self.cpt is not None:
            self.tgt_sr = self.cpt["config"][-1]
            self.cpt["config"][-3] = self.cpt["weight"]["emb_g.weight"].shape[0]
            self.use_f0 = self.cpt.get("f0", 1)

            self.version = self.cpt.get("version", "v1")
            self.text_enc_hidden_dim = 768 if self.version == "v2" else 256
            self.vocoder = self.cpt.get("vocoder", "HiFi-GAN")
            self.net_g = Synthesizer(
                *self.cpt["config"],
                use_f0=self.use_f0,
                text_enc_hidden_dim=self.text_enc_hidden_dim,
                vocoder=self.vocoder,
            )
            del self.net_g.enc_q
            self.net_g.load_state_dict(self.cpt["weight"], strict=False)
            self.net_g = self.net_g.to(self.config.device).float()
            self.net_g.eval()

    def setup_vc_instance(self):
        """
        Sets up the voice conversion pipeline instance based on the target sampling rate and configuration.
        """
        if self.cpt is not None:
            self.vc = VC(self.tgt_sr, self.config)
            self.n_spk = self.cpt["config"][-3]
