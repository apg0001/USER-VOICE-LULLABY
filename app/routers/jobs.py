"""작업 상태 조회 라우터"""
import re
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from ..constants import RVC_LOGS_DIR
from ..dependencies import get_inference_queue, get_train_queue
from ..models.responses import JobStatusResponse
from ..task_queue import AsyncJobQueue

router = APIRouter()


def _get_training_progress(job, queue_name: str) -> dict | None:
    """학습 작업의 진행률을 계산합니다."""
    if queue_name != "train" or job.status.value != "running":
        return None
    
    # job.metadata에서 model_name과 total_epoch 확인
    model_name = job.metadata.get("model_name")
    total_epoch = job.metadata.get("total_epoch")
    
    if not model_name or not total_epoch:
        return None
    
    try:
        model_dir = RVC_LOGS_DIR / model_name
        if not model_dir.exists():
            return None
        
        # 최신 체크포인트 파일 찾기 (G_*.pth)
        checkpoint_files = sorted(
            model_dir.glob("G_*.pth"),
            key=lambda f: f.stat().st_mtime,
            reverse=True
        )
        
        if not checkpoint_files:
            return {"current_epoch": 0, "total_epoch": total_epoch, "progress_percent": 0.0}
        
        latest_checkpoint = checkpoint_files[0]
        
        # 체크포인트 파일명에서 epoch 추출 시도
        # G_2333333.pth (save_only_latest인 경우) 또는 G_12345.pth (global_step)
        # 또는 model_name_50e_1000s.pth 형식
        current_epoch = 0
        
        # 체크포인트 파일을 로드하여 epoch 확인
        try:
            import torch
            checkpoint = torch.load(str(latest_checkpoint), map_location="cpu", weights_only=True)
            # checkpoint의 iteration이 epoch일 수 있음 (save_checkpoint에서 epoch를 iteration으로 저장)
            # 하지만 정확하지 않을 수 있으므로 파일명에서도 시도
            if "iteration" in checkpoint:
                # iteration이 epoch일 가능성이 높음 (train.py의 save_checkpoint 참고)
                current_epoch = checkpoint.get("iteration", 0)
        except Exception:
            # 체크포인트 로드 실패 시 파일명에서 추출 시도
            filename = latest_checkpoint.name
            epoch_match = re.search(r'(\d+)e', filename)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
            else:
                current_epoch = 0
        
        # total_epoch와 비교하여 진행률 계산
        progress_percent = min((current_epoch / total_epoch * 100) if total_epoch > 0 else 0, 100.0)
        
        return {
            "current_epoch": current_epoch,
            "total_epoch": total_epoch,
            "progress_percent": round(progress_percent, 2),
        }
    except Exception:
        return None


@router.get("/jobs/{queue_name}", response_model=list[JobStatusResponse])
async def list_jobs(
    queue_name: str,
    train_queue: AsyncJobQueue = Depends(get_train_queue),
    inference_queue: AsyncJobQueue = Depends(get_inference_queue),
) -> list[JobStatusResponse]:
    """큐에 있는 모든 작업 리스트를 조회합니다."""
    queue = None
    if queue_name == "train":
        queue = train_queue
    elif queue_name == "inference":
        queue = inference_queue
    else:
        raise HTTPException(status_code=400, detail=f"Unknown queue: {queue_name}")
    
    all_jobs = queue.list_all_jobs()
    results = []
    
    for job_data in all_jobs:
        # 학습 작업의 경우 진행률 계산
        job = queue.get_job_status(job_data["job_id"])
        progress = None
        if job:
            progress = _get_training_progress(job, queue_name)
        
        job_data["progress"] = progress
        results.append(JobStatusResponse(**job_data))
    
    return results


@router.get("/jobs/{queue_name}/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    queue_name: str,
    job_id: str,
    train_queue: AsyncJobQueue = Depends(get_train_queue),
    inference_queue: AsyncJobQueue = Depends(get_inference_queue),
) -> JobStatusResponse:
    """특정 작업 상태를 조회합니다."""
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
    
    # 학습 작업의 경우 진행률 계산
    job = queue.get_job_status(job_id)
    progress = None
    if job:
        progress = _get_training_progress(job, queue_name)
    
    result["progress"] = progress
    
    return JobStatusResponse(**result)


@router.delete("/jobs/{queue_name}/{job_id}")
async def cancel_job(
    queue_name: str,
    job_id: str,
    train_queue: AsyncJobQueue = Depends(get_train_queue),
    inference_queue: AsyncJobQueue = Depends(get_inference_queue),
):
    """작업을 취소합니다."""
    queue = None
    if queue_name == "train":
        queue = train_queue
    elif queue_name == "inference":
        queue = inference_queue
    else:
        raise HTTPException(status_code=400, detail=f"Unknown queue: {queue_name}")
    
    success = queue.cancel_job(job_id)
    if not success:
        raise HTTPException(
            status_code=404, 
            detail="Job not found or cannot be cancelled (job may be already completed, failed, or cancelled)"
        )
    
    return {"status": "cancelled", "job_id": job_id}

