"""Health check endpoint."""
from __future__ import annotations

import logging
import os
import subprocess
import time
from datetime import datetime

from fastapi import APIRouter

from src.v6.executor import get_executor
from src.v6 import __version__

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Health"])

_start_time = time.monotonic()


def _get_gpu_info() -> list[dict]:
    """Query all GPUs via nvidia-smi, return list."""
    gpus = []
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,gpu_name,memory.total,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return gpus
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                gpus.append({
                    "index": int(parts[0]),
                    "device": parts[1],
                    "vram_total_mb": int(float(parts[2])),
                    "vram_used_mb": int(float(parts[3])),
                    "utilization_pct": float(parts[4]),
                })
    except Exception as e:
        logger.debug("nvidia-smi failed: %s", e)
    return gpus


@router.get("/health")
async def health_check():
    executor = get_executor()
    engines = executor.list_engines()

    # Gather engine health info
    engine_states = []
    for e in engines:
        try:
            h = await e.health()
            engine_states.append({"id": e.engine_id, "name": e.name, **h})
        except Exception:
            engine_states.append({"id": e.engine_id, "name": e.name, "available": False})

    real_engines = [e for e in engines if e.engine_id != "mock"]
    online_engines = [s for s in engine_states if s.get("available")]
    has_real_online = any(s["id"] != "mock" and s.get("available") for s in engine_states)

    # GPU info from nvidia-smi
    gpus = _get_gpu_info()

    # Overall status
    if has_real_online or gpus:
        overall = "healthy"
    elif online_engines:
        overall = "degraded"
    else:
        overall = "unhealthy"

    return {
        "status": overall,
        "version": __version__,
        "uptime_sec": round(time.monotonic() - _start_time, 1),
        "gpus": gpus,
        "engines": {
            "total": len(engines),
            "online": len(online_engines),
            "real": len(real_engines),
        },
        "redis": os.environ.get("REDIS_URL", "not configured"),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
