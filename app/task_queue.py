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
    CANCELLED = "cancelled"


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
    cancelled: bool = False  # 취소 플래그


class AsyncJobQueue:
    """Job ID 기반 비동기 작업 큐."""

    def __init__(self, name: str, resource_monitor=None, max_workers: int = 4):
        self.name = name
        self._queue: "asyncio.Queue[tuple[str, Callable[..., Awaitable[Any]], tuple[Any, ...], dict[str, Any], asyncio.Future]]" = asyncio.Queue()
        self._workers: list[asyncio.Task] = []
        self._max_workers = max_workers
        self._resource_monitor = resource_monitor
        self._active_jobs: set[str] = set()  # 현재 실행 중인 job_id 집합
        self._active_tasks: dict[str, asyncio.Task] = {}  # job_id -> 실행 중인 Task
        self._jobs: dict[str, Job] = {}  # job_id -> Job

    @property
    def pending(self) -> int:
        return self._queue.qsize()

    @property
    def is_running(self) -> bool:
        return len(self._workers) > 0 and any(not w.done() for w in self._workers)
    
    @property
    def running_count(self) -> int:
        """현재 실행 중인 작업 수"""
        return len(self._active_jobs)

    async def start(self) -> None:
        """워커 시작 (여러 워커 생성)"""
        if len(self._workers) == 0 or all(w.done() for w in self._workers):
            # 리소스 모니터가 있으면 동적으로 워커 수 결정
            num_workers = self._max_workers
            if self._resource_monitor:
                try:
                    status = self._resource_monitor.get_resource_status()
                    num_workers = min(status.get_max_concurrent_jobs(), self._max_workers)
                except Exception:
                    pass  # 리소스 모니터 오류 시 기본값 사용
            
            self._workers = [
                asyncio.create_task(self._worker_loop()) 
                for _ in range(num_workers)
            ]

    async def stop(self) -> None:
        """모든 워커 중지"""
        for worker in self._workers:
            worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await worker
        self._workers = []

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
    
    def cancel_job(self, job_id: str) -> bool:
        """작업 취소
        
        Returns:
            bool: 취소 성공 여부
        """
        from .logging_config import get_logger
        logger = get_logger(f"{__name__}.{self.name}")
        
        job = self._jobs.get(job_id)
        if job is None:
            logger.warning(f"작업을 찾을 수 없음 | job_id={job_id}")
            return False
        
        # 이미 완료된 작업은 취소 불가
        if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
            logger.debug(
                f"작업 취소 불가 (이미 완료됨) | job_id={job_id} | status={job.status.value}"
            )
            return False
        
        # PENDING 상태인 작업: 큐에서 제거
        if job.status == JobStatus.PENDING:
            try:
                job.status = JobStatus.CANCELLED
                job.completed_at = datetime.now()
                job.error = "작업이 취소되었습니다."
                if job.future and not job.future.done():
                    job.future.cancel()
                logger.info(f"작업 취소 완료 (대기 중) | queue={self.name} | job_id={job_id}")
                return True
            except Exception as e:
                logger.error(f"작업 취소 중 오류 | queue={self.name} | job_id={job_id} | error={e}", exc_info=True)
                return False
        
        # RUNNING 상태인 작업: 취소 플래그 설정
        if job.status == JobStatus.RUNNING:
            try:
                job.cancelled = True
                logger.info(f"작업 취소 플래그 설정 (실행 중) | queue={self.name} | job_id={job_id}")
                return True
            except Exception as e:
                logger.error(f"작업 취소 플래그 설정 중 오류 | queue={self.name} | job_id={job_id} | error={e}", exc_info=True)
                return False
        
        logger.warning(f"알 수 없는 작업 상태 | queue={self.name} | job_id={job_id} | status={job.status.value}")
        return False
    
    def list_all_jobs(self) -> list[dict[str, Any]]:
        """모든 작업 리스트 반환"""
        jobs = []
        for job in self._jobs.values():
            jobs.append({
                "job_id": job.job_id,
                "status": job.status.value,
                "created_at": job.created_at.isoformat(),
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                "result": job.result,
                "error": job.error,
                "metadata": job.metadata,
            })
        # 생성 시간 역순으로 정렬 (최신 작업이 먼저)
        return sorted(jobs, key=lambda x: x["created_at"], reverse=True)

    async def _wait_for_cancel(self, job_id: str) -> None:
        """작업 취소를 기다리는 헬퍼 함수"""
        job = self._jobs.get(job_id)
        if not job:
            return
        
        # 취소 플래그가 설정될 때까지 대기
        while not job.cancelled:
            await asyncio.sleep(0.5)  # 0.5초마다 확인
    
    def stats(self) -> dict[str, Any]:
        """현재 큐 상태를 딕셔너리로 반환한다."""
        return {
            "name": self.name,
            "pending": self.pending,  # 대기 중인 작업 수
            "running": self.running_count,  # 실행 중인 작업 수
        }

    async def _worker_loop(self) -> None:
        from .logging_config import get_logger
        worker_logger = get_logger(f"{__name__}.{self.name}_worker")
        
        worker_logger.info(f"워커 루프 시작 | queue={self.name}")
        
        while True:
            try:
                job_id, coroutine_func, args, kwargs, future = await self._queue.get()
                job = self._jobs.get(job_id)
                
                if job is None:
                    worker_logger.warning(f"작업 정보를 찾을 수 없음 (큐에서 제거됨) | queue={self.name} | job_id={job_id}")
                    self._queue.task_done()
                    continue
            except Exception as e:
                worker_logger.error(f"큐에서 작업 가져오기 실패 | queue={self.name} | error={e}", exc_info=True)
                continue
            
            # 리소스 모니터가 있으면 리소스 상태 확인
            if self._resource_monitor:
                try:
                    status = self._resource_monitor.get_resource_status()
                    # 리소스가 부족하면 대기 (최대 30초)
                    wait_count = 0
                    max_wait = 30  # 최대 30초 대기
                    while not status.can_accept_job and wait_count < max_wait:
                        await asyncio.sleep(1)
                        wait_count += 1
                        status = self._resource_monitor.get_resource_status()
                    
                    if wait_count >= max_wait:
                        worker_logger.warning(
                            f"리소스 부족으로 작업 시작 지연 | queue={self.name} | job_id={job_id}"
                        )
                except Exception as e:
                    worker_logger.debug(f"리소스 모니터 확인 실패: {e}")
            
            # 재시도 로직
            max_retries = 3
            retry_count = 0
            
            while retry_count <= max_retries:
                try:
                    # 취소 확인 (큐에서 가져온 후 취소 상태 확인)
                    if job.status == JobStatus.CANCELLED or job.cancelled:
                        if job.status != JobStatus.CANCELLED:
                            job.status = JobStatus.CANCELLED
                            job.completed_at = datetime.now()
                            job.error = "작업이 취소되었습니다."
                        worker_logger.info(f"작업 취소됨 (대기 중) | queue={self.name} | job_id={job_id}")
                        if not future.done():
                            future.cancel()
                        break
                    
                    self._active_jobs.add(job_id)
                    if retry_count == 0:
                        job.status = JobStatus.RUNNING
                        job.started_at = datetime.now()
                        worker_logger.info(
                            f"작업 시작 | queue={self.name} | job_id={job_id} | "
                            f"function={coroutine_func.__name__ if hasattr(coroutine_func, '__name__') else 'unknown'}"
                        )
                    else:
                        worker_logger.warning(
                            f"작업 재시도 | queue={self.name} | job_id={job_id} | "
                            f"retry_count={retry_count}/{max_retries}"
                        )
                    
                    # 실행 중 취소 확인을 위한 태스크 래핑
                    task = asyncio.create_task(coroutine_func(*args, **kwargs))
                    self._active_tasks[job_id] = task  # 실행 중인 태스크 추적
                    
                    # 취소 확인 태스크 생성
                    cancel_check_task = asyncio.create_task(self._wait_for_cancel(job_id))
                    
                    try:
                        # 태스크와 취소 확인을 동시에 대기
                        done, pending = await asyncio.wait(
                            {task, cancel_check_task},
                            return_when=asyncio.FIRST_COMPLETED
                        )
                        
                        # 취소 확인 태스크가 먼저 완료되면 작업 취소
                        if cancel_check_task in done and job.cancelled:
                            task.cancel()
                            worker_logger.info(f"작업 취소 요청 감지, 태스크 취소 중 | queue={self.name} | job_id={job_id}")
                        
                        # 나머지 태스크 취소
                        for p in pending:
                            p.cancel()
                            try:
                                await p
                            except asyncio.CancelledError:
                                pass
                        
                        # 작업 결과 가져오기
                        if task.done() and not task.cancelled():
                            result = await task
                        elif job.cancelled:
                            raise asyncio.CancelledError()
                        else:
                            result = await task
                            
                    except asyncio.CancelledError:
                        # 작업이 취소된 경우
                        if job.cancelled:
                            job.status = JobStatus.CANCELLED
                            job.completed_at = datetime.now()
                            job.error = "작업이 취소되었습니다."
                            worker_logger.info(f"작업 취소됨 (실행 중) | queue={self.name} | job_id={job_id}")
                            if not future.done():
                                future.cancel()
                        raise
                    finally:
                        # 태스크 추적에서 제거
                        self._active_tasks.pop(job_id, None)
                    
                    # 실행 중 취소 플래그 확인 (작업 완료 후)
                    if job.cancelled:
                        job.status = JobStatus.CANCELLED
                        job.completed_at = datetime.now()
                        job.error = "작업이 취소되었습니다."
                        worker_logger.info(f"작업 취소됨 (완료 후 취소 처리) | queue={self.name} | job_id={job_id}")
                        if not future.done():
                            future.cancel()
                        break
                    
                    job.status = JobStatus.COMPLETED
                    job.completed_at = datetime.now()
                    job.result = result
                    
                    worker_logger.info(
                        f"작업 완료 | queue={self.name} | job_id={job_id} | "
                        f"duration={(job.completed_at - job.started_at).total_seconds():.2f}초 | "
                        f"retry_count={retry_count}"
                    )
                    
                    if not future.done():
                        future.set_result(result)
                    
                    # 성공 시 루프 종료
                    break
                    
                except Exception as exc:  # pragma: no cover - 안전망
                    last_exception = exc
                    retry_count += 1
                    
                    if retry_count <= max_retries:
                        # 재시도 전 대기 (지수 백오프)
                        wait_time = min(2 ** retry_count, 10)  # 최대 10초
                        worker_logger.warning(
                            f"작업 실패 (재시도 예정) | queue={self.name} | job_id={job_id} | "
                            f"retry_count={retry_count}/{max_retries} | wait_time={wait_time}초 | error={str(exc)}"
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        # 최대 재시도 횟수 초과
                        job.status = JobStatus.FAILED
                        job.completed_at = datetime.now()
                        job.error = str(exc)
                        
                        worker_logger.error(
                            f"작업 최종 실패 (재시도 한도 초과) | queue={self.name} | job_id={job_id} | "
                            f"retry_count={retry_count} | error={str(exc)}",
                            exc_info=True
                        )
                        
                        if not future.done():
                            future.set_exception(exc)
                finally:
                    self._active_jobs.discard(job_id)
            
            self._queue.task_done()
