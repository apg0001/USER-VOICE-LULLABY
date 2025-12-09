"""헬스 체크 및 리소스 모니터링 라우터"""
from fastapi import APIRouter, Depends

from ..dependencies import get_resource_monitor, get_train_queue, get_inference_queue
from ..models.responses import HealthResponse, ResourceInfo, QueueStats
from ..monitors.resource_monitor import ResourceMonitor
from ..task_queue import AsyncJobQueue

router = APIRouter()


@router.get("/resources", response_model=ResourceInfo)
async def get_resources(
    monitor: ResourceMonitor = Depends(get_resource_monitor),
) -> ResourceInfo:
    """현재 시스템 리소스 상태를 조회합니다."""
    status = monitor.get_resource_status()
    
    return ResourceInfo(
        cpu_percent=status.cpu_percent,
        memory_percent=status.memory_percent,
        disk_percent=status.disk_percent,
        gpu_utilization=status.gpu_utilization,
        gpu_memory_used=status.gpu_memory_used,
        can_accept_job=status.can_accept_job,
    )


@router.get("/", response_model=HealthResponse)
async def health_check(
    monitor: ResourceMonitor = Depends(get_resource_monitor),
    train_queue: AsyncJobQueue = Depends(get_train_queue),
    inference_queue: AsyncJobQueue = Depends(get_inference_queue),
) -> HealthResponse:
    """헬스 체크 엔드포인트"""
    status = monitor.get_resource_status(interval=0.0)
    
    queue_stats = {
        "train": QueueStats(**train_queue.stats()),
        "inference": QueueStats(**inference_queue.stats()),
    }
    
    return HealthResponse(
        status="ok",
        cpu_percent=status.cpu_percent,
        memory_percent=status.memory_percent,
        disk_percent=status.disk_percent,
        queues=queue_stats,
        gpus=status.gpu_info,
        resource_available=status.can_accept_job,
    )

