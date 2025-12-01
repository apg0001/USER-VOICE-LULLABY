from __future__ import annotations

import asyncio
import contextlib
from typing import Any, Awaitable, Callable


class AsyncJobQueue:
    """간단한 단일 워커 FIFO 큐."""

    def __init__(self, name: str):
        self.name = name
        self._queue: "asyncio.Queue[tuple[Callable[..., Awaitable[Any]], tuple[Any, ...], dict[str, Any], asyncio.Future]]" = asyncio.Queue()
        self._worker: asyncio.Task | None = None
        self._active = False

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

    async def enqueue(self, coroutine_func: Callable[..., Awaitable[Any]], *args, **kwargs) -> Any:
        loop = asyncio.get_running_loop()
        future: "asyncio.Future[Any]" = loop.create_future()
        await self._queue.put((coroutine_func, args, kwargs, future))
        return await future

    def stats(self) -> dict[str, Any]:
        """현재 큐 상태를 딕셔너리로 반환한다."""
        return {
            "name": self.name,
            "pending": self.pending,              # 대기 중인 작업 수
            "running": 1 if self._active else 0,  # 실행 중인 작업 수 (단일 워커)
        }

    async def _worker_loop(self) -> None:
        while True:
            coroutine_func, args, kwargs, future = await self._queue.get()
            try:
                self._active = True
                result = await coroutine_func(*args, **kwargs)
                if not future.done():
                    future.set_result(result)
            except Exception as exc:  # pragma: no cover - 안전망
                if not future.done():
                    future.set_exception(exc)
            finally:
                self._active = False
                self._queue.task_done()

