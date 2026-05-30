"""Engine status and health query APIs.

Hardware: Single machine (192.168.71.166) with dual GPU:
    - GPU 0: RTX 3060 Ti 8G — display + NVENC/NVDEC + ffmpeg IO (host)
    - GPU 1: RTX 3090 24G — CUDA inference primary
"""
from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException

from src.v6.engine.cloud_pool import CLOUD_PROVIDERS, get_cloud_pool
from src.v6.engine.local_pool import get_local_pool
from src.v6.engine.router import LOCAL_VRAM_GB, VRAM_HARD_CAP_GB
from src.v6.executor import get_executor
from src.v6.models.task import TaskType, TaskStatus
from src.v6.store import get_task_store

router = APIRouter(prefix="/api/v1/engines", tags=["Engine"])


@router.get("")
async def list_engines():
    """List all registered engines (executor-managed + legacy)."""
    executor = get_executor()
    local_pool = get_local_pool()
    cloud_pool = get_cloud_pool()
    local_health = local_pool.health()

    engines = []

    # Executor-managed engines (new)
    for engine in executor.list_engines():
        cap = engine.capabilities
        engines.append({
            "id": engine.engine_id,
            "name": engine.name,
            "pool": "local",
            "type": "executor",
            "status": "online",
            "supported_types": cap.supported_types,
            "vram_total_mb": cap.vram_total_mb,
            "vram_used_mb": cap.vram_total_mb - cap.vram_available_mb,
            "queue_depth": 0,
            "models": cap.models,
        })

    # Legacy local ComfyUI
    engines.append({
        "id": "local-comfyui",
        "name": "ComfyUI Local (RTX 3090 - inference primary)",
        "pool": "local",
        "type": "comfyui",
        "status": "online" if local_health["available"] else "offline",
        "supported_types": [t.value for t in TaskType],
        "vram_total_mb": local_health["vram_total_mb"],
        "vram_used_mb": local_health.get("vram_used_mb", 0),
        "queue_depth": 0,
        "models": ["wan2.2-14b", "flux-dev", "ace-step", "cosyvoice", "real-esrgan"],
    })

    # Legacy cloud providers
    for pid, info in CLOUD_PROVIDERS.items():
        engines.append({
            "id": f"cloud-{pid}",
            "name": info["name"],
            "pool": "cloud",
            "type": pid,
            "status": "online" if info["available"] else "offline",
            "supported_types": info["supported_types"],
            "vram_total_mb": None,
            "vram_used_mb": None,
            "queue_depth": 0,
            "models": [],
        })

    return {"engines": engines}


@router.get("/capacity")
async def get_capacity():
    local_pool = get_local_pool()
    cloud_pool = get_cloud_pool()
    store = get_task_store()
    local_health = local_pool.health()
    cloud_health = cloud_pool.health()

    _, queue_total = await store.list_tasks(limit=1)
    running_tasks, _ = await store.list_tasks(status=TaskStatus.RUNNING, limit=1)

    return {
        "local": {
            "available": local_health["available"],
            "vram_total_mb": local_health["vram_total_mb"],
            "vram_available_mb": local_health["vram_available_mb"],
            "gpu_utilization_pct": local_health["gpu_utilization_pct"],
            "running_tasks": len(running_tasks),
            "queued_tasks": await store.queue_size(),
            "estimated_wait_sec": await store.queue_size() * 5.0,
        },
        "cloud": cloud_health,
        "total_queue_depth": await store.queue_size(),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


@router.get("/{engine_id}/health")
async def engine_health(engine_id: str):
    executor = get_executor()
    local_pool = get_local_pool()
    cloud_pool = get_cloud_pool()

    start = time.monotonic()

    # Check executor-managed engines first
    engine = executor.get_engine(engine_id)
    if engine:
        health_data = await engine.health()
        elapsed = (time.monotonic() - start) * 1000
        return {
            "id": engine_id,
            "status": "healthy" if health_data.get("available") else "unhealthy",
            "response_time_ms": elapsed,
            "details": health_data,
            "checked_at": datetime.utcnow().isoformat() + "Z",
        }

    # Legacy engines
    if engine_id.startswith("local"):
        health = local_pool.health()
        status = "healthy" if health["available"] else "unhealthy"
        details = health
    elif engine_id.startswith("cloud-"):
        provider = engine_id.replace("cloud-", "")
        info = CLOUD_PROVIDERS.get(provider, {})
        status = "healthy" if info.get("available") else "unhealthy"
        details = {"provider": provider, "available": info.get("available", False)}
    else:
        raise HTTPException(status_code=404, detail={
            "error": "engine_not_found",
            "message": f"Engine '{engine_id}' not found",
        })

    elapsed = (time.monotonic() - start) * 1000

    return {
        "id": engine_id,
        "status": status,
        "response_time_ms": elapsed,
        "details": details,
        "checked_at": datetime.utcnow().isoformat() + "Z",
    }


@router.get("/{engine_id}/status")
async def engine_status(engine_id: str):
    """Detailed status for a specific executor-managed engine."""
    executor = get_executor()
    engine = executor.get_engine(engine_id)
    if not engine:
        raise HTTPException(status_code=404, detail={
            "error": "engine_not_found",
            "message": f"Executor engine '{engine_id}' not found",
        })

    health_data = await engine.health()
    cap = engine.capabilities

    return {
        "id": engine.engine_id,
        "name": engine.name,
        "health": health_data,
        "capabilities": {
            "supported_types": cap.supported_types,
            "max_resolution": list(cap.max_resolution),
            "max_duration_sec": cap.max_duration_sec,
            "vram_total_mb": cap.vram_total_mb,
            "models": cap.models,
        },
    }
