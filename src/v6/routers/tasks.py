"""Task CRUD API — POST/GET/cancel."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query

from src.v6.callbacks import send_callback
from src.v6.engine.cloud_pool import get_cloud_pool
from src.v6.engine.router import get_engine_router, EnginePool
from src.v6.models.task import (
    BatchCreateRequest,
    BatchCreateResponse,
    BatchTaskResult,
    ErrorResponse,
    GenerationTask,
    Priority,
    TaskAcceptedResponse,
    TaskCancelResponse,
    TaskCreateRequest,
    TaskDetailResponse,
    TaskListResponse,
    TaskStatus,
    TaskType,
)
from src.v6.store import get_task_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/tasks", tags=["Task"])


@router.post("", response_model=TaskAcceptedResponse, status_code=202)
async def create_task(req: TaskCreateRequest):
    store = get_task_store()
    engine_router = get_engine_router()

    # Check duplicate
    existing = await store.get(req.task_id)
    if existing:
        raise HTTPException(status_code=400, detail={
            "error": "duplicate_task_id",
            "message": f"Task '{req.task_id}' already exists",
        })

    # Build task
    task = GenerationTask(
        task_id=req.task_id,
        type=req.type,
        priority=req.priority,
        model_preference=req.model_preference,
        params=req.params,
        callback_url=req.callback_url,
        callback_secret=req.callback_secret,
    )

    # Route to engine
    pool, engine_id = engine_router.route(task)
    task.engine_used = pool
    task.engine_id = engine_id

    await store.put(task)

    queue_pos = await store.queue_size()

    return TaskAcceptedResponse(
        task_id=task.task_id,
        status="queued",
        engine_target=pool.value,
        queue_position=queue_pos,
        estimated_start_sec=queue_pos * 5.0 if queue_pos > 0 else 0.0,
        created_at=task.created_at,
    )


@router.get("", response_model=TaskListResponse)
async def list_tasks(
    status: Optional[TaskStatus] = None,
    type: Optional[str] = None,
    priority: Optional[Priority] = None,
    engine_used: Optional[str] = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    created_after: Optional[str] = None,
    created_before: Optional[str] = None,
):
    store = get_task_store()

    type_enum = None
    if type:
        try:
            type_enum = TaskType(type)
        except ValueError:
            pass

    tasks, total = await store.list_tasks(
        status=status,
        type=type_enum,
        priority=priority,
        engine_used=engine_used,
        limit=limit,
        offset=offset,
        created_after=created_after,
        created_before=created_before,
    )

    return TaskListResponse(
        tasks=[t.to_detail() for t in tasks],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{task_id}", response_model=TaskDetailResponse)
async def get_task(task_id: str):
    store = get_task_store()
    task = await store.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail={
            "error": "task_not_found",
            "message": f"Task '{task_id}' not found",
        })
    return task.to_detail()


@router.post("/{task_id}/cancel", response_model=TaskCancelResponse)
async def cancel_task(task_id: str):
    store = get_task_store()
    task = await store.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail={
            "error": "task_not_found",
            "message": f"Task '{task_id}' not found",
        })
    if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
        raise HTTPException(status_code=409, detail={
            "error": "cannot_cancel",
            "message": f"Task '{task_id}' is already {task.status.value}",
        })

    await store.update(task_id, status=TaskStatus.CANCELLED)
    return TaskCancelResponse(task_id=task_id)


@router.post("/batch", response_model=BatchCreateResponse, status_code=202)
async def batch_create(req: BatchCreateRequest):
    store = get_task_store()
    engine_router = get_engine_router()
    batch_id = str(uuid4())
    results: list[BatchTaskResult] = []
    accepted = 0
    rejected = 0

    for task_req in req.tasks:
        existing = await store.get(task_req.task_id)
        if existing:
            rejected += 1
            results.append(BatchTaskResult(
                task_id=task_req.task_id,
                status="rejected",
                error=f"Task '{task_req.task_id}' already exists",
            ))
            if req.fail_fast:
                break
            continue

        task = GenerationTask(
            task_id=task_req.task_id,
            type=task_req.type,
            priority=task_req.priority,
            model_preference=task_req.model_preference,
            params=task_req.params,
            callback_url=task_req.callback_url,
            callback_secret=task_req.callback_secret,
        )
        pool, engine_id = engine_router.route(task)
        task.engine_used = pool
        task.engine_id = engine_id
        await store.put(task)
        accepted += 1
        results.append(BatchTaskResult(
            task_id=task_req.task_id,
            status="queued",
            queue_position=await store.queue_size(),
        ))

    return BatchCreateResponse(
        batch_id=batch_id,
        accepted=accepted,
        rejected=rejected,
        results=results,
    )
