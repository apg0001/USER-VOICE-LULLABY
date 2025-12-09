"""리소스 모니터링 서비스"""
from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import psutil

from ..logging_config import PROJECT_ROOT


@dataclass
class ResourceStatus:
    """리소스 상태"""
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    gpu_utilization: Optional[list[float]] = None
    gpu_memory_used: Optional[list[float]] = None
    gpu_info: Optional[list[dict]] = None

    @property
    def can_accept_job(self) -> bool:
        """새 작업을 받을 수 있는지 판단"""
        # CPU 사용률이 90% 이상이면 거부
        if self.cpu_percent > 90.0:
            return False
        
        # 메모리 사용률이 90% 이상이면 거부
        if self.memory_percent > 90.0:
            return False
        
        # 디스크 사용률이 95% 이상이면 거부
        if self.disk_percent > 95.0:
            return False
        
        # GPU가 있고 모든 GPU가 95% 이상 사용 중이면 거부
        if self.gpu_utilization:
            if all(util > 95.0 for util in self.gpu_utilization):
                return False
        
        return True


class ResourceMonitor:
    """시스템 리소스 모니터링 클래스"""
    
    def __init__(self, project_root: Path | None = None):
        self.project_root = project_root or PROJECT_ROOT
        self._logger = logging.getLogger(__name__)
    
    def get_cpu_percent(self, interval: float = 0.1) -> float:
        """CPU 사용률 조회"""
        return psutil.cpu_percent(interval=interval)
    
    def get_memory_percent(self) -> float:
        """메모리 사용률 조회"""
        return psutil.virtual_memory().percent
    
    def get_disk_percent(self) -> float:
        """디스크 사용률 조회"""
        disk_usage = shutil.disk_usage(self.project_root)
        if disk_usage.total == 0:
            return 0.0
        return (disk_usage.used / disk_usage.total) * 100
    
    def get_gpu_info(self) -> tuple[Optional[list[float]], Optional[list[float]], Optional[list[dict]]]:
        """GPU 정보 조회 (utilization, memory, full_info)"""
        try:
            import pynvml
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            
            gpu_utils = []
            gpu_memory = []
            gpu_info_list = []
            
            for i in range(count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                gpu_utils.append(float(util.gpu))
                gpu_memory.append(round(mem.used / (1024 * 1024), 2))
                gpu_info_list.append({
                    "index": i,
                    "name": name,
                    "utilization_percent": float(util.gpu),
                    "memory_used_mb": round(mem.used / (1024 * 1024), 2),
                    "memory_total_mb": round(mem.total / (1024 * 1024), 2),
                })
            
            pynvml.nvmlShutdown()
            return gpu_utils, gpu_memory, gpu_info_list
        except Exception as e:
            self._logger.debug(f"GPU info fetch failed: {e}")
            return None, None, None
    
    def get_resource_status(self, interval: float = 0.1) -> ResourceStatus:
        """전체 리소스 상태 조회"""
        cpu_percent = self.get_cpu_percent(interval)
        memory_percent = self.get_memory_percent()
        disk_percent = self.get_disk_percent()
        gpu_utils, gpu_memory, gpu_info = self.get_gpu_info()
        
        return ResourceStatus(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            disk_percent=disk_percent,
            gpu_utilization=gpu_utils,
            gpu_memory_used=gpu_memory,
            gpu_info=gpu_info,
        )

