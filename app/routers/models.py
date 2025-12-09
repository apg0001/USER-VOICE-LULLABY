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
        raise HTTPException(status_code=500, detail=f"모델 삭제 중 오류 발생: {str(exc)}")


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
        raise HTTPException(status_code=500, detail=f"출력 파일 삭제 중 오류 발생: {str(exc)}")


@router.get("/outputs/{output_id}/download")
async def download_output_file(
    output_id: str,
    output_service: OutputService = Depends(get_output_service),
):
    """출력 파일 다운로드"""
    try:
        from ..repositories.output_repository import OutputRepository
        repo = OutputRepository()
        file_path = repo.get_output_path(output_id)
        filename = file_path.name
        
        # 파일 확장자에 따른 미디어 타입 결정
        media_type_map = {
            ".wav": "audio/wav",
            ".mp3": "audio/mpeg",
            ".flac": "audio/flac",
            ".m4a": "audio/mp4",
        }
        media_type = media_type_map.get(file_path.suffix.lower(), "audio/wav")
        
        return FileResponse(
            str(file_path),
            filename=filename,
            media_type=media_type,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as exc:
        logger.exception(f"출력 파일 다운로드 실패: {output_id}")
        raise HTTPException(status_code=500, detail=f"파일 다운로드 중 오류 발생: {str(exc)}")


@router.get("/download")
async def download_file(
    path: str = Query(..., description="오디오 파일 이름"),
    file_repo: FileRepository = Depends(get_file_repository),
):
    """파일 다운로드"""
    try:
        file_path = file_repo.get_file_path(path)
        filename = file_path.name
        return FileResponse(
            str(file_path),
            filename=filename,
            media_type="audio/wav",
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

