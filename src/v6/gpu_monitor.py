"""GPU VRAM monitor and manager.

Provides real-time VRAM usage tracking, allocation checks, and per-engine
memory accounting.  Uses ``nvidia-smi`` for hardware-level readings plus
an in-process registry for per-engine estimates.
"""
from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass, field
from time import monotonic
from typing import Any

logger = logging.getLogger(__name__)

# ── defaults ────────────────────────────────────────────────────────────
DEFAULT_GPU_INDEX = 1          # RTX 3090 (CUDA inference primary)
VRAM_RESERVE_MB = 2048         # keep 2 GB for system / misc
VRAM_TOTAL_MB = 24576           # RTX 3090

# ── data classes ──────────────────────────────────────────────────────────

@dataclass
class EngineMemoryRecord:
    """Tracks a single engine's VRAM footprint."""
    engine_id: str
    vram_mb: int
    loaded_at: float = field(default_factory=monotonic)   # last acquire
    last_used_at: float = field(default_factory=monotonic)

    def touch(self) -> None:
        self.last_used_at = monotonic()


@dataclass
class GPUStatus:
    total_mb: int
    used_mb: int
    free_mb: int
    gpu_name: str = ""
    gpu_index: int = DEFAULT_GPU_INDEX


# ── singleton state ───────────────────────────────────────────────────────

_engine_records: dict[str, EngineMemoryRecord] = {}


def register_engine(engine_id: str, vram_mb: int) -> None:
    """Register an engine's expected VRAM footprint (called by EnginePool)."""
    if engine_id in _engine_records:
        _engine_records[engine_id].vram_mb = vram_mb
        return
    _engine_records[engine_id] = EngineMemoryRecord(engine_id=engine_id, vram_mb=vram_mb)


def unregister_engine(engine_id: str) -> None:
    _engine_records.pop(engine_id, None)


def touch_engine(engine_id: str) -> None:
    """Mark *engine_id* as recently used (LRU bookkeeping)."""
    rec = _engine_records.get(engine_id)
    if rec:
        rec.touch()


def mark_unloaded(engine_id: str) -> None:
    """Remove a loaded engine from accounting (after actual unload)."""
    _engine_records.pop(engine_id, None)


# ── hardware queries ────────────────────────────────────────────────────

def _query_nvidia_smi(gpu_index: int = DEFAULT_GPU_INDEX) -> GPUStatus:
    try:
        # memory.used, memory.total
        r = subprocess.run(
            ["nvidia-smi", "-i", str(gpu_index),
             "--query-gpu=memory.used,memory.total,name",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode != 0:
            return GPUStatus(total_mb=VRAM_TOTAL_MB, used_mb=0, free_mb=VRAM_TOTAL_MB)
        parts = [p.strip() for p in r.stdout.strip().split(",")]
        used_mb = int(parts[0]) if parts else 0
        total_mb = int(parts[1]) if len(parts) > 1 else VRAM_TOTAL_MB
        gpu_name = parts[2] if len(parts) > 2 else ""
        return GPUStatus(
            total_mb=total_mb,
            used_mb=used_mb,
            free_mb=max(0, total_mb - used_mb),
            gpu_name=gpu_name,
            gpu_index=gpu_index,
        )
    except Exception:
        return GPUStatus(total_mb=VRAM_TOTAL_MB, used_mb=0, free_mb=VRAM_TOTAL_MB)


def get_gpu_vram_usage(gpu_index: int = DEFAULT_GPU_INDEX) -> dict[str, Any]:
    """Return ``{"total_mb", "used_mb", "free_mb", "gpu_name"}``."""
    s = _query_nvidia_smi(gpu_index)
    return {
        "total_mb": s.total_mb,
        "used_mb": s.used_mb,
        "free_mb": s.free_mb,
        "gpu_name": s.gpu_name,
    }


def can_allocate(require_mb: int, gpu_index: int = DEFAULT_GPU_INDEX) -> bool:
    """True if *require_mb* can fit after reserving VRAM_RESERVE_MB."""
    s = _query_nvidia_smi(gpu_index)
    return s.free_mb >= require_mb + VRAM_RESERVE_MB


def get_active_engines() -> list[dict[str, Any]]:
    """Return currently-loaded engines sorted by last-used (oldest first = LRU head)."""
    now = monotonic()
    return [
        {
            "engine_id": r.engine_id,
            "vram_mb": r.vram_mb,
            "loaded_at": r.loaded_at,
            "last_used_at": r.last_used_at,
            "idle_seconds": round(now - r.last_used_at, 1),
        }
        for r in sorted(_engine_records.values(), key=lambda r: r.last_used_at)
    ]


def auto_evict(require_mb: int, gpu_index: int = DEFAULT_GPU_INDEX) -> list[str]:
    """Evict LRU engines until *require_mb* (+reserve) can be allocated.

    Returns list of evicted engine IDs (callers must perform actual unload).
    """
    evicted: list[str] = []
    # work on a snapshot sorted LRU → MRU
    snapshot = sorted(_engine_records.values(), key=lambda r: r.last_used_at)
    for rec in snapshot:
        if can_allocate(require_mb, gpu_index):
            break
        evicted.append(rec.engine_id)
        mark_unloaded(rec.engine_id)
        logger.info("GPU auto-evict: unloaded '%s' (%d MB)", rec.engine_id, rec.vram_mb)
    return evicted


def force_unload_engine(engine_name: str) -> bool:
    """Remove engine from accounting. Returns True if it was tracked."""
    if engine_name in _engine_records:
        mark_unloaded(engine_name)
        return True
    return False
