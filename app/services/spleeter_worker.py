"""Spleeter를 별도 프로세스에서 실행하기 위한 워커 스크립트"""
import sys
import os
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from spleeter.separator import Separator


def separate_audio(input_path: str, output_dir: str):
    """별도 프로세스에서 오디오 분리 실행"""
    try:
        separator = Separator("spleeter:2stems")
        separator.separate_to_file(input_path, output_dir)
        return 0
    except Exception as e:
        print(f"Error in spleeter worker: {e}", file=sys.stderr)
        return 1
    finally:
        # 프로세스 종료 시 모든 메모리 자동 해제
        pass


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python spleeter_worker.py <input_path> <output_dir>", file=sys.stderr)
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_dir = sys.argv[2]
    
    exit_code = separate_audio(input_path, output_dir)
    sys.exit(exit_code)

