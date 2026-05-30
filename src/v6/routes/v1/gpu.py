"""GPU VRAM management API endpoints."""
from __future__ import annotations

import logging

from fastapi import APIRouter

from src.v6.engine_pool import get_engine_pool
from src.v6.gpu_monitor import get_active_engines, get_gpu_vram_usage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/gpu", tags=["gpu"])


@router.get("/status")
async def gpu_status():
    """GPU VRAM usage overview."""
    return get_gpu_vram_usage()


@router.get("/engines")
async def engine_load_status():
    """Engine load state list (LRU-ordered)."""
    pool = get_engine_pool()
    return {
        "pool": pool.status(),
        "active": get_active_engines(),
    }


@router.post("/unload/{engine_name}")
async def unload_engine(engine_name: str):
    """Manually unload a specific engine."""
    pool = get_engine_pool()
    ok = await pool.unload(engine_name)
    return {"engine": engine_name, "unloaded": ok}


@router.post("/unload-all")
async def unload_all_engines():
    """Unload all engines from GPU."""
    pool = get_engine_pool()
    count = await pool.unload_all()
    return {"unloaded_count": count}
