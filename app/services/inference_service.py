"""추론 서비스"""

from __future__ import annotations

from typing import Optional

from ..logging_config import get_logger
from ..models.requests import InferenceFilesRequest

# services.py의 함수를 직접 import (순환 참조 방지를 위해 동적 import)
import importlib.util
from pathlib import Path

_services_module_path = Path(__file__).parent.parent / "services.py"
spec = importlib.util.spec_from_file_location(
    "app_services_module", _services_module_path
)
app_services_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app_services_module)
_run_inference = app_services_module.run_inference

logger = get_logger(__name__)


class InferenceService:
    """음성 변환 추론 서비스"""

    def __init__(self):
        self._logger = logger

    async def infer(
        self,
        input_audio_path: str,
        model_path: str,
        index_path: Optional[str] = None,
        output_dir: str = "outputs",
        pitch: Optional[int] = None,
        volume_envelope: Optional[float] = None,
        protect: Optional[float] = None,
        f0_autotune: Optional[bool] = None,
        f0_autotune_strength: Optional[float] = None,
        embedder_model: Optional[str] = None,
        index_rate: Optional[float] = None,
        clean_audio: Optional[bool] = None,
        clean_strength: Optional[float] = None,
        reverb: Optional[bool] = None,
        reverb_room_size: Optional[float] = None,
        reverb_damping: Optional[float] = None,
        reverb_wet_gain: Optional[float] = None,
        reverb_dry_gain: Optional[float] = None,
        reverb_width: Optional[float] = None,
        reverb_freeze_mode: Optional[float] = None,
    ) -> dict:
        """추론 실행"""
        return await _run_inference(
            input_audio_path=input_audio_path,
            model_path=model_path,
            index_path=index_path,
            output_dir=output_dir,
            pitch=pitch,
            volume_envelope=volume_envelope,
            protect=protect,
            f0_autotune=f0_autotune,
            f0_autotune_strength=f0_autotune_strength,
            embedder_model=embedder_model,
            index_rate=index_rate,
            clean_audio=clean_audio,
            clean_strength=clean_strength,
            reverb=reverb,
            reverb_room_size=reverb_room_size,
            reverb_damping=reverb_damping,
            reverb_wet_gain=reverb_wet_gain,
            reverb_dry_gain=reverb_dry_gain,
            reverb_width=reverb_width,
            reverb_freeze_mode=reverb_freeze_mode,
        )

    async def infer_from_request(self, request: InferenceFilesRequest) -> dict:
        """요청 객체로부터 추론 실행"""
        return await self.infer(
            input_audio_path=request.input_audio_path,
            model_path=request.model_path,
            index_path=request.index_path,
            output_dir=request.output_dir,
            pitch=request.pitch,
            volume_envelope=request.volume_envelope,
            protect=request.protect,
            f0_autotune=request.f0_autotune,
            f0_autotune_strength=request.f0_autotune_strength,
            embedder_model=request.embedder_model,
            index_rate=request.index_rate,
            clean_audio=request.clean_audio,
            clean_strength=request.clean_strength,
            reverb=request.reverb,
            reverb_room_size=request.reverb_room_size,
            reverb_damping=request.reverb_damping,
            reverb_wet_gain=request.reverb_wet_gain,
            reverb_dry_gain=request.reverb_dry_gain,
            reverb_width=request.reverb_width,
            reverb_freeze_mode=request.reverb_freeze_mode,
        )
