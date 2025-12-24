"""모델 및 출력 관리 라우터"""

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse

from ..dependencies import (
    get_file_repository,
    get_model_service,
    get_output_service,
)
from ..models.responses import ModelInfo, OutputInfo
from ..repositories.file_repository import FileRepository
from ..services.model_service import ModelService
from ..services.output_service import OutputService
from ..logging_config import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.get("/models", response_model=list[ModelInfo])
async def list_models(
    model_service: ModelService = Depends(get_model_service),
) -> list[ModelInfo]:
    """학습된 모델 리스트를 조회합니다."""
    return model_service.list_models()


def _extract_sample_rate(filename: str) -> str:
    """파일명에서 샘플레이트 추출 (예: f0G32k.pth -> 32k)"""
    import re

    match = re.search(r"(\d+)k", filename, re.IGNORECASE)
    if match:
        return f"{match.group(1)}k"
    return "unknown"


@router.get("/pretrained-models")
async def list_pretrained_models():
    """사전 학습 모델 리스트를 조회합니다."""
    from pathlib import Path
    from ..constants import RVC_ROOT, RVC_LOGS_DIR

    pretrained_models = {
        "hifi-gan": [],
        "custom": [],
        "logs": [],  # logs 폴더의 사전 학습 모델
    }

    # 3. logs 폴더의 사전 학습 모델 (KLM 등)
    if RVC_LOGS_DIR.exists():
        for model_dir in RVC_LOGS_DIR.iterdir():
            if not model_dir.is_dir():
                continue

            # 보호된 디렉토리 제외
            from ..repositories.model_repository import PROTECTED_DIRS

            if model_dir.name in PROTECTED_DIRS:
                continue

            # logs 폴더 내의 모든 .pth 파일 스캔
            for pth_file in model_dir.glob("*.pth"):
                file_name = pth_file.name
                file_path = str(pth_file.resolve())
                # G_ 또는 D_로 시작하는 파일만 사전 학습 모델로 간주
                if file_name.startswith("G_") or file_name.startswith("G-"):
                    pretrained_models["logs"].append(
                        {
                            "name": file_name,
                            "path": file_path,
                            "type": "G",
                            "sample_rate": _extract_sample_rate(file_name),
                            "model_id": model_dir.name,
                        }
                    )
                elif file_name.startswith("D_") or file_name.startswith("D-"):
                    pretrained_models["logs"].append(
                        {
                            "name": file_name,
                            "path": file_path,
                            "type": "D",
                            "sample_rate": _extract_sample_rate(file_name),
                            "model_id": model_dir.name,
                        }
                    )

    return pretrained_models


def _extract_sample_rate(filename: str) -> str:
    """파일명에서 샘플레이트 추출 (예: f0G32k.pth -> 32k)"""
    import re

    match = re.search(r"(\d+)k", filename, re.IGNORECASE)
    if match:
        return f"{match.group(1)}k"
    return "unknown"


@router.delete("/models/{model_id}")
async def delete_model(
    model_id: str,
    model_service: ModelService = Depends(get_model_service),
):
    """모델 ID로 모델을 삭제합니다."""
    try:
        model_service.delete_model(model_id)
        return {"status": "deleted", "model_id": model_id}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception(f"모델 삭제 실패: {model_id}")
        raise HTTPException(
            status_code=500, detail=f"모델 삭제 중 오류 발생: {str(exc)}"
        )


@router.get("/outputs", response_model=list[OutputInfo])
async def list_outputs(
    output_service: OutputService = Depends(get_output_service),
) -> list[OutputInfo]:
    """추론 결과 리스트를 조회합니다."""
    return output_service.list_outputs()


@router.delete("/outputs/{output_id}")
async def delete_output(
    output_id: str,
    output_service: OutputService = Depends(get_output_service),
):
    """출력 파일 ID로 추론 결과를 삭제합니다."""
    try:
        output_service.delete_output(output_id)
        return {"status": "deleted", "output_id": output_id}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception(f"출력 파일 삭제 실패: {output_id}")
        raise HTTPException(
            status_code=500, detail=f"출력 파일 삭제 중 오류 발생: {str(exc)}"
        )


@router.get("/outputs/{output_id}/download")
async def download_output_file(
    output_id: str,
    output_service: OutputService = Depends(get_output_service),
):
    """출력 파일 다운로드"""
    logger.info(f"다운로드 요청 수신 | output_id={output_id}")
    try:
        from ..repositories.output_repository import OutputRepository

        repo = OutputRepository()
        file_path = repo.get_output_path(output_id)
        filename = file_path.name

        if not file_path.exists():
            logger.error(
                f"파일이 존재하지 않음 | output_id={output_id} | path={file_path}"
            )
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.debug(
            f"파일 다운로드 시작 | output_id={output_id} | filename={filename} | path={file_path}"
        )

        # 파일 확장자에 따른 미디어 타입 결정
        media_type_map = {
            ".wav": "audio/wav",
            ".mp3": "audio/mpeg",
            ".flac": "audio/flac",
            ".m4a": "audio/mp4",
        }
        media_type = media_type_map.get(file_path.suffix.lower(), "audio/wav")

        response = FileResponse(
            str(file_path),
            filename=filename,
            media_type=media_type,
        )
        logger.debug(
            f"파일 다운로드 응답 생성 완료 | output_id={output_id} | filename={filename}"
        )
        return response
    except FileNotFoundError as e:
        logger.error(f"파일을 찾을 수 없음 | output_id={output_id} | error={str(e)}")
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as exc:
        logger.exception(
            f"출력 파일 다운로드 실패 | output_id={output_id} | error={str(exc)}"
        )
        raise HTTPException(
            status_code=500, detail=f"파일 다운로드 중 오류 발생: {str(exc)}"
        )


@router.get("/download")
async def download_file(
    path: str = Query(..., description="오디오 파일 이름"),
    file_repo: FileRepository = Depends(get_file_repository),
):
    """파일 다운로드"""
    logger.info(f"다운로드 요청 수신 | path={path}")
    try:
        file_path = file_repo.get_file_path(path)
        filename = file_path.name

        if not file_path.exists():
            logger.error(
                f"파일이 존재하지 않음 | path={path} | resolved_path={file_path}"
            )
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.debug(
            f"파일 다운로드 시작 | path={path} | filename={filename} | resolved_path={file_path}"
        )

        response = FileResponse(
            str(file_path),
            filename=filename,
            media_type="audio/wav",
        )
        logger.info(f"파일 다운로드 응답 생성 완료 | path={path} | filename={filename}")
        return response
    except FileNotFoundError as e:
        logger.error(f"파일을 찾을 수 없음 | path={path} | error={str(e)}")
        raise HTTPException(status_code=404, detail="File not found")
    except ValueError as exc:
        logger.error(f"잘못된 경로 | path={path} | error={str(exc)}")
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception(f"파일 다운로드 실패 | path={path} | error={str(exc)}")
        raise HTTPException(
            status_code=500, detail=f"파일 다운로드 중 오류 발생: {str(exc)}"
        )
