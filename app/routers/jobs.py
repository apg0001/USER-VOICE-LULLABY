"""작업 상태 조회 라우터"""
from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import get_inference_queue, get_train_queue
from ..models.responses import JobStatusResponse
from ..task_queue import AsyncJobQueue

router = APIRouter()


@router.get("/jobs/{queue_name}/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    queue_name: str,
    job_id: str,
    train_queue: AsyncJobQueue = Depends(get_train_queue),
    inference_queue: AsyncJobQueue = Depends(get_inference_queue),
) -> JobStatusResponse:
    """작업 상태를 조회합니다."""
    queue = None
    if queue_name == "train":
        queue = train_queue
    elif queue_name == "inference":
        queue = inference_queue
    else:
        raise HTTPException(status_code=400, detail=f"Unknown queue: {queue_name}")
    
    result = queue.get_job_result(job_id)
    if "error" in result and result["error"] == "Job not found":
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JobStatusResponse(**result)

