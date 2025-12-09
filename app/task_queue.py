from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Awaitable, Callable
from uuid import uuid4


class JobStatus(str, Enum):
    """작업 상태"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Job:
    """작업 정보"""
    job_id: str
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result: Any = None
    error: str | None = None
    future: asyncio.Future | None = None
    metadata: dict[str, Any] = field(default_factory=dict)  # 작업 메타데이터 (예: model_name, total_epoch)


class AsyncJobQueue:
    """Job ID 기반 비동기 작업 큐."""

    def __init__(self, name: str):
        self.name = name
        self._queue: "asyncio.Queue[tuple[str, Callable[..., Awaitable[Any]], tuple[Any, ...], dict[str, Any], asyncio.Future]]" = asyncio.Queue()
        self._worker: asyncio.Task | None = None
        self._active = False
        self._jobs: dict[str, Job] = {}  # job_id -> Job

    @property
    def pending(self) -> int:
        return self._queue.qsize()

    @property
    def is_running(self) -> bool:
        return self._worker is not None and not self._worker.done()

    async def start(self) -> None:
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._worker_loop())

    async def stop(self) -> None:
        if self._worker:
            self._worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker
            self._worker = None

    def enqueue_async(
        self, coroutine_func: Callable[..., Awaitable[Any]], *args, **kwargs
    ) -> str:
        """작업을 큐에 추가하고 job_id를 즉시 반환 (비동기)"""
        job_id = str(uuid4())
        loop = asyncio.get_running_loop()
        future: "asyncio.Future[Any]" = loop.create_future()
        
        job = Job(job_id=job_id, future=future)
        self._jobs[job_id] = job
        
        # 큐에 추가 (비동기로 실행하되 즉시 반환)
        async def _put_to_queue():
            await self._queue.put((job_id, coroutine_func, args, kwargs, future))
        
        asyncio.create_task(_put_to_queue())
        
        return job_id

    async def enqueue(
        self, coroutine_func: Callable[..., Awaitable[Any]], *args, **kwargs
    ) -> Any:
        """작업을 큐에 추가하고 완료될 때까지 대기 (동기)"""
        loop = asyncio.get_running_loop()
        future: "asyncio.Future[Any]" = loop.create_future()
        job_id = str(uuid4())
        
        job = Job(job_id=job_id, future=future)
        self._jobs[job_id] = job
        
        await self._queue.put((job_id, coroutine_func, args, kwargs, future))
        return await future

    def get_job_status(self, job_id: str) -> Job | None:
        """작업 상태 조회"""
        return self._jobs.get(job_id)

    def get_job_result(self, job_id: str) -> dict[str, Any]:
        """작업 결과 조회"""
        job = self._jobs.get(job_id)
        if job is None:
            return {"error": "Job not found"}
        
        return {
            "job_id": job.job_id,
            "status": job.status.value,
            "created_at": job.created_at.isoformat(),
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "result": job.result,
            "error": job.error,
        }

    def stats(self) -> dict[str, Any]:
        """현재 큐 상태를 딕셔너리로 반환한다."""
        return {
            "name": self.name,
            "pending": self.pending,  # 대기 중인 작업 수
            "running": 1 if self._active else 0,  # 실행 중인 작업 수 (단일 워커)
        }

    async def _worker_loop(self) -> None:
        while True:
            job_id, coroutine_func, args, kwargs, future = await self._queue.get()
            job = self._jobs.get(job_id)
            
            if job is None:
                continue
            
            try:
                self._active = True
                job.status = JobStatus.RUNNING
                job.started_at = datetime.now()
                
                result = await coroutine_func(*args, **kwargs)
                
                job.status = JobStatus.COMPLETED
                job.completed_at = datetime.now()
                job.result = result
                
                if not future.done():
                    future.set_result(result)
            except Exception as exc:  # pragma: no cover - 안전망
                job.status = JobStatus.FAILED
                job.completed_at = datetime.now()
                job.error = str(exc)
                
                if not future.done():
                    future.set_exception(exc)
            finally:
                self._active = False
                self._queue.task_done()
