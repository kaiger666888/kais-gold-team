"""In-memory task store with async queue — mock mode placeholder."""
from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from typing import Optional

from src.v6.models.task import (
    GenerationTask,
    TaskMetadata,
    TaskOutputs,
    TaskStatus,
    TaskType,
    Priority,
    EnginePool,
)


class TaskStore:
    """Simple in-memory task store. Will be replaced by Redis in production."""

    def __init__(self) -> None:
        self._tasks: OrderedDict[str, GenerationTask] = OrderedDict()
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._lock = asyncio.Lock()
        self._subscribers: list[asyncio.Queue] = []

    async def put(self, task: GenerationTask) -> None:
        async with self._lock:
            self._tasks[task.task_id] = task
        await self._queue.put(task.task_id)
        await self._publish("queued", task.task_id)

    async def get(self, task_id: str) -> Optional[GenerationTask]:
        return self._tasks.get(task_id)

    async def update(
        self,
        task_id: str,
        *,
        status: Optional[TaskStatus] = None,
        engine_used: Optional[EnginePool] = None,
        engine_id: Optional[str] = None,
        outputs: Optional[TaskOutputs] = None,
        metadata: Optional[TaskMetadata] = None,
        error: Optional[str] = None,
        progress: Optional[float] = None,
    ) -> Optional[GenerationTask]:
        async with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return None
            if status is not None:
                task.status = status
            if engine_used is not None:
                task.engine_used = engine_used
            if engine_id is not None:
                task.engine_id = engine_id
            if outputs is not None:
                task.outputs = outputs
            if metadata is not None:
                task.metadata = metadata
            if error is not None:
                task.error = error
            if progress is not None:
                task.progress = progress

            from datetime import datetime
            if status == TaskStatus.RUNNING and task.started_at is None:
                task.started_at = datetime.utcnow()
            if status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
                task.completed_at = datetime.utcnow()

            self._tasks[task_id] = task

        event_name = status.value if status else "progress"
        await self._publish(event_name, task_id)
        return task

    async def list_tasks(
        self,
        *,
        status: Optional[TaskStatus] = None,
        type: Optional[TaskType] = None,
        priority: Optional[Priority] = None,
        engine_used: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
        created_after: Optional[str] = None,
        created_before: Optional[str] = None,
    ) -> tuple[list[GenerationTask], int]:
        tasks = list(self._tasks.values())

        if status:
            tasks = [t for t in tasks if t.status == status]
        if type:
            tasks = [t for t in tasks if t.type == type]
        if priority:
            tasks = [t for t in tasks if t.priority == priority]
        if engine_used:
            tasks = [t for t in tasks if t.engine_used and t.engine_used.value == engine_used]

        total = len(tasks)
        tasks = tasks[offset : offset + limit]
        return tasks, total

    async def queue_size(self) -> int:
        return self._queue.qsize()

    # ─── SSE subscriber support ───

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        if q in self._subscribers:
            self._subscribers.remove(q)

    async def _publish(self, event: str, task_id: str) -> None:
        for q in self._subscribers:
            await q.put((event, task_id))


# Singleton
_store: Optional[TaskStore] = None


def get_task_store() -> TaskStore:
    global _store
    if _store is None:
        _store = TaskStore()
    return _store
