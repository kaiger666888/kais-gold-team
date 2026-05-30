"""SSE event stream for task lifecycle events."""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException, Request
from sse_starlette.sse import EventSourceResponse

from src.v6.models.task import TaskStatus
from src.v6.store import get_task_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/tasks", tags=["Task"])


@router.get("/{task_id}/events")
async def task_event_stream(task_id: str, request: Request):
    """SSE endpoint for real-time task events."""
    store = get_task_store()
    task = await store.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail={
            "error": "task_not_found",
            "message": f"Task '{task_id}' not found",
        })

    async def event_generator():
        q = store.subscribe()
        try:
            # Send current state as first event
            current = await store.get(task_id)
            if current:
                yield {
                    "event": current.status.value,
                    "data": json.dumps({
                        "task_id": current.task_id,
                        "status": current.status.value,
                        "progress": current.progress,
                        "message": f"Current state: {current.status.value}",
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                    }),
                }

            # If already terminal, close immediately
            if current and current.status in (
                TaskStatus.COMPLETED,
                TaskStatus.FAILED,
                TaskStatus.CANCELLED,
            ):
                return

            # Listen for events
            while True:
                if await request.is_disconnected():
                    break

                try:
                    event_name, evt_task_id = await asyncio.wait_for(q.get(), timeout=30.0)
                except asyncio.TimeoutError:
                    # Send keepalive
                    yield {"event": "ping", "data": ""}
                    continue

                if evt_task_id != task_id:
                    continue

                t = await store.get(task_id)
                if not t:
                    continue

                yield {
                    "event": event_name,
                    "data": json.dumps({
                        "task_id": t.task_id,
                        "status": t.status.value,
                        "progress": t.progress,
                        "message": f"Task {event_name}",
                        "outputs": t.outputs.model_dump() if t.outputs else None,
                        "metadata": t.metadata.model_dump() if t.metadata else None,
                        "error": t.error,
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                    }),
                }

                # Terminal state → close stream
                if t.status in (
                    TaskStatus.COMPLETED,
                    TaskStatus.FAILED,
                    TaskStatus.CANCELLED,
                ):
                    break

        finally:
            store.unsubscribe(q)

    return EventSourceResponse(event_generator())
