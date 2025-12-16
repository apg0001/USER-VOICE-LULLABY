"""RVC를 별도 프로세스에서 실행하기 위한 워커 스크립트

메모리 누수 방지를 위해 별도 프로세스에서 실행되며,
프로세스 종료 시 모든 메모리가 자동으로 해제됩니다.
"""
import sys
import json
import os
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# RVC 모듈 경로 추가
rvc_root = project_root / "applio"
if str(rvc_root) not in sys.path:
    sys.path.insert(0, str(rvc_root))

# RVC 내부 모듈 경로 추가
inner_rvc = rvc_root / "rvc"
if str(inner_rvc) not in sys.path:
    sys.path.insert(0, str(inner_rvc))

# 작업 디렉토리를 RVC 루트로 변경
original_cwd = os.getcwd()
try:
    if original_cwd != str(rvc_root):
        os.chdir(str(rvc_root))
    
    from core import run_infer_script
except Exception as e:
    print(f"Error importing RVC modules: {e}", file=sys.stderr)
    sys.exit(1)
finally:
    os.chdir(original_cwd)


def run_inference_from_config(config_path: str):
    """JSON 설정 파일을 읽어서 RVC inference 실행"""
    try:
        # 설정 파일 읽기
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # run_infer_script 호출
        message, output_path = run_infer_script(
            pitch=config.get('pitch', 0),
            index_rate=config.get('index_rate', 0.75),
            volume_envelope=config.get('volume_envelope', 1.0),
            protect=config.get('protect', 0.5),
            f0_method=config.get('f0_method', 'rmvpe'),
            input_path=config['input_path'],
            output_path=config['output_path'],
            pth_path=config['pth_path'],
            index_path=config.get('index_path', ''),
            split_audio=config.get('split_audio', False),
            f0_autotune=config.get('f0_autotune', False),
            f0_autotune_strength=config.get('f0_autotune_strength', 1.0),
            proposed_pitch=config.get('proposed_pitch', False),
            proposed_pitch_threshold=config.get('proposed_pitch_threshold', 155.0),
            clean_audio=config.get('clean_audio', False),
            clean_strength=config.get('clean_strength', 0.5),
            export_format=config.get('export_format', 'WAV'),
            embedder_model=config.get('embedder_model', 'contentvec'),
            embedder_model_custom=config.get('embedder_model_custom', None),
            formant_shifting=config.get('formant_shifting', False),
            formant_qfrency=config.get('formant_qfrency', 1.0),
            formant_timbre=config.get('formant_timbre', 1.0),
            post_process=config.get('post_process', False),
            reverb=config.get('reverb', False),
            pitch_shift=config.get('pitch_shift', False),
            limiter=config.get('limiter', False),
            gain=config.get('gain', False),
            distortion=config.get('distortion', False),
            chorus=config.get('chorus', False),
            bitcrush=config.get('bitcrush', False),
            clipping=config.get('clipping', False),
            compressor=config.get('compressor', False),
            delay=config.get('delay', False),
            reverb_room_size=config.get('reverb_room_size', 0.5),
            reverb_damping=config.get('reverb_damping', 0.5),
            reverb_wet_gain=config.get('reverb_wet_gain', 0.5),
            reverb_dry_gain=config.get('reverb_dry_gain', 0.5),
            reverb_width=config.get('reverb_width', 0.5),
            reverb_freeze_mode=config.get('reverb_freeze_mode', 0.5),
            pitch_shift_semitones=config.get('pitch_shift_semitones', 0.0),
            limiter_threshold=config.get('limiter_threshold', -6),
            limiter_release_time=config.get('limiter_release_time', 0.01),
            gain_db=config.get('gain_db', 0.0),
            distortion_gain=config.get('distortion_gain', 25),
            chorus_rate=config.get('chorus_rate', 1.0),
            chorus_depth=config.get('chorus_depth', 0.25),
            chorus_center_delay=config.get('chorus_center_delay', 7),
            chorus_feedback=config.get('chorus_feedback', 0.0),
            chorus_mix=config.get('chorus_mix', 0.5),
            bitcrush_bit_depth=config.get('bitcrush_bit_depth', 8),
            clipping_threshold=config.get('clipping_threshold', -6),
            compressor_threshold=config.get('compressor_threshold', 0),
            compressor_ratio=config.get('compressor_ratio', 1),
            compressor_attack=config.get('compressor_attack', 1.0),
            compressor_release=config.get('compressor_release', 100),
            delay_seconds=config.get('delay_seconds', 0.5),
            delay_feedback=config.get('delay_feedback', 0.0),
            delay_mix=config.get('delay_mix', 0.5),
            sid=config.get('sid', 0),
        )
        
        # 성공 결과를 JSON으로 출력
        result = {
            'success': True,
            'message': message,
            'output_path': output_path
        }
        print(json.dumps(result, ensure_ascii=False))
        return 0
        
    except Exception as e:
        # 에러 결과를 JSON으로 출력
        result = {
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__
        }
        print(json.dumps(result, ensure_ascii=False), file=sys.stderr)
        print(f"Error in RVC worker: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return 1
    finally:
        # 프로세스 종료 시 모든 메모리 자동 해제
        pass


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python rvc_worker.py <config_json_path>", file=sys.stderr)
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    
    exit_code = run_inference_from_config(config_path)
    sys.exit(exit_code)

