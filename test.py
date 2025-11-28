import os
from pathlib import Path
import spleeter
from spleeter.separator import Separator

def separate_vocals_inst(audio_path: str) -> dict:
    """Spleeter로 보컬/인스트 분리. 원본과 같은 경로에 저장"""
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"파일 없음: {audio_path}")
    
    # 원본 파일과 같은 디렉토리
    output_dir = audio_path.parent
    
    # 출력 파일명 (원본명 기반)
    base_name = audio_path.stem
    vocals_path = output_dir / f"{base_name}_vocals.wav"
    inst_path = output_dir / f"{base_name}_instrumental.wav"
    
    # 2-stem 모델 (vocals + accompaniment)
    separator = Separator('spleeter:2stems')
    
    try:
        # 분리 실행
        separator.separate_to_file(
            str(audio_path), 
            str(output_dir),
            filename_format="{filename}/vocals.wav"  # _vocals.wav 형태로 저장
        )
        
        # 실제 파일명 재조정 (spleeter가 폴더 생성함)
        generated_vocals = output_dir / f"{base_name}/vocals.wav"
        generated_inst = output_dir / f"{base_name}/accompaniment.wav"
        
        generated_vocals.rename(vocals_path)
        generated_inst.rename(inst_path)
        
        # 임시 폴더 삭제
        (output_dir / base_name).rmdir()
        
        return {
            "vocals": str(vocals_path),
            "instrumental": str(inst_path),
            "success": True
        }
    
    except Exception as e:
        return {"error": str(e), "success": False}

# 사용법
result = separate_vocals_inst("C:/Users/user/Downloads/보고싶다.mp3")
print(f"보컬: {result['vocals']}")
print(f"인스트: {result['instrumental']}")
