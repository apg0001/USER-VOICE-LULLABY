from __future__ import annotations

import argparse
import logging

import uvicorn

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Applio FastAPI server.")
    parser.add_argument("--host", default=DEFAULT_HOST, help="서버 호스트 (기본값 0.0.0.0)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="서버 포트")
    parser.add_argument(
        "--reload",
        action="store_true",
        help="파일 변경 시 자동 재시작 (개발용 옵션)",
    )
    return parser.parse_args()

def check_cuda_torch():
    print("=== PyTorch CUDA 정보 ===")
    print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
    print(f"CUDA 장치 개수: {torch.cuda.device_count()}")
    
    if torch.cuda.is_available():
        print(f"현재 CUDA 장치: {torch.cuda.get_device_name(0)}")
        print(f"CUDA 버전: {torch.version.cuda}")
        print(f"PyTorch 버전: {torch.__version__}")
        print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")
    else:
        print("CUDA를 사용할 수 없습니다.")
    
    print()

def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    check_cuda_torch()
    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()

