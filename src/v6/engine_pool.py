"""Engine lifecycle manager — lazy load, LRU evict, explicit unload.

Manages a pool of engine instances that can be loaded into GPU memory on
demand and automatically evicted when VRAM is insufficient.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from time import monotonic
from typing import Any, Awaitable, Callable, Optional

from src.v6.gpu_monitor import (
    VRAM_RESERVE_MB,
    can_allocate,
    get_active_engines,
    get_gpu_vram_usage,
    mark_unloaded,
    _query_nvidia_smi as query_nvidia_smi,
    register_engine,
    touch_engine,
)

logger = logging.getLogger(__name__)


class EngineState(str, Enum):
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    EVICTING = "evicting"
    FAILED = "failed"


@dataclass
class _PoolEntry:
    engine_id: str
    vram_mb: int
    loader: Callable[[], Awaitable[Any]]   # async factory → engine instance
    unloader: Optional[Callable[[Any], Awaitable[None]]] = None
    state: EngineState = EngineState.UNLOADED
    instance: Any = None
    loaded_at: float = 0.0
    last_used_at: float = 0.0
    _release_handle: Optional[asyncio.TimerHandle] = None


class EnginePool:
    """Lazy-loading engine pool with LRU eviction for GPU VRAM management.

    Usage::

        pool = EnginePool(idle_timeout_sec=300)
        pool.register("wan-video", loader=my_loader, vram_mb=16000)
        engine = await pool.acquire("wan-video")
        # ... use engine ...
        pool.release("wan-video")   # starts idle timer

    When VRAM is insufficient, ``acquire`` auto-evicts LRU engines.
    """

    def __init__(
        self,
        idle_timeout_sec: float = 300.0,   # 5 min keep-alive after release
        max_concurrent: int = 2,           # max engines in GPU simultaneously
    ) -> None:
        self._entries: dict[str, _PoolEntry] = {}
        self._idle_timeout = idle_timeout_sec
        self._max_concurrent = max_concurrent

    # ── registration ──────────────────────────────────────────────────────

    def register(
        self,
        engine_id: str,
        loader: Callable[[], Awaitable[Any]],
        vram_mb: int = 0,
        unloader: Callable[[Any], Awaitable[None]] | None = None,
    ) -> None:
        """Register an engine without loading it."""
        if engine_id in self._entries:
            logger.warning("EnginePool: overwriting existing entry '%s'", engine_id)
        self._entries[engine_id] = _PoolEntry(
            engine_id=engine_id,
            vram_mb=vram_mb,
            loader=loader,
            unloader=unloader,
        )
        register_engine(engine_id, vram_mb)
        logger.info("EnginePool: registered '%s' (vram=%d MB, idle_timeout=%.0fs)",
                     engine_id, vram_mb, self._idle_timeout)

    # ── acquire / release ────────────────────────────────────────────────

    async def acquire(self, engine_id: str) -> Any:
        """Load (if needed) and return the engine instance.

        If VRAM is insufficient, LRU engines are auto-evicted first.
        """
        entry = self._entries.get(engine_id)
        if entry is None:
            raise KeyError(f"Engine '{engine_id}' not registered in pool")

        # Already loaded — just touch
        if entry.state == EngineState.LOADED:
            entry.last_used_at = monotonic()
            touch_engine(engine_id)
            self._cancel_idle_timer(entry)
            return entry.instance

        if entry.state == EngineState.LOADING:
            logger.info("EnginePool: '%s' is loading, waiting...", engine_id)
            # simple spin — in production use an Event
            for _ in range(120):
                await asyncio.sleep(0.5)
                if entry.state == EngineState.LOADED:
                    entry.last_used_at = monotonic()
                    touch_engine(engine_id)
                    return entry.instance
                if entry.state in (EngineState.UNLOADED, EngineState.FAILED):
                    break
            raise RuntimeError(f"Engine '{engine_id}' stuck in LOADING state")

        # Need to load — evict if necessary
        await self._ensure_vram(entry.vram_mb)

        # Enforce max concurrent
        loaded_count = sum(1 for e in self._entries.values() if e.state == EngineState.LOADED)
        if loaded_count >= self._max_concurrent:
            await self._evict_oldest(loaded_count - self._max_concurrent + 1)

        # Load
        entry.state = EngineState.LOADING
        try:
            entry.instance = await entry.loader()
            entry.state = EngineState.LOADED
            now = monotonic()
            entry.loaded_at = now
            entry.last_used_at = now
            touch_engine(engine_id)
            logger.info("EnginePool: loaded '%s' (%d MB)", engine_id, entry.vram_mb)
            return entry.instance
        except Exception:
            entry.state = EngineState.FAILED
            mark_unloaded(engine_id)
            logger.exception("EnginePool: failed to load '%s'", engine_id)
            raise

    def release(self, engine_id: str) -> None:
        """Mark engine as releasable; starts idle-unload timer."""
        entry = self._entries.get(engine_id)
        if entry is None or entry.state != EngineState.LOADED:
            return
        entry.last_used_at = monotonic()
        touch_engine(engine_id)

        # Schedule idle unload
        loop = asyncio.get_running_loop()
        entry._release_handle = loop.call_later(
            self._idle_timeout,
            lambda: asyncio.ensure_future(self._idle_unload(engine_id)),
        )
        logger.info("EnginePool: released '%s', idle unload in %.0fs",
                     engine_id, self._idle_timeout)

    # ── explicit unload ──────────────────────────────────────────────────

    async def unload(self, engine_id: str) -> bool:
        """Immediately unload an engine from GPU."""
        entry = self._entries.get(engine_id)
        if entry is None or entry.state != EngineState.LOADED:
            return False
        self._cancel_idle_timer(entry)
        await self._do_unload(entry)
        return True

    async def unload_all(self) -> int:
        """Unload all loaded engines. Returns count unloaded."""
        tasks = [self.unload(eid) for eid, e in self._entries.items()
                 if e.state == EngineState.LOADED]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return sum(1 for r in results if r is True)

    # ── status ────────────────────────────────────────────────────────────

    def get_engine(self, engine_id: str) -> Any:
        """Return the loaded engine instance or None."""
        entry = self._entries.get(engine_id)
        if entry and entry.state == EngineState.LOADED:
            return entry.instance
        return None

    def status(self) -> dict[str, Any]:
        """Return full pool status: engine states + GPU info."""
        now = monotonic()
        engines = {}
        for eid, e in self._entries.items():
            engines[eid] = {
                "state": e.state.value,
                "vram_mb": e.vram_mb,
                "loaded_at": e.loaded_at,
                "last_used_at": e.last_used_at,
                "idle_seconds": round(now - e.last_used_at, 1) if e.state == EngineState.LOADED else None,
            }
        return {
            "engines": engines,
            "gpu": get_gpu_vram_usage(),
            "idle_timeout_sec": self._idle_timeout,
            "max_concurrent": self._max_concurrent,
        }

    # ── internals ────────────────────────────────────────────────────────

    async def _ensure_vram(self, require_mb: int) -> None:
        if can_allocate(require_mb):
            return
        logger.info("EnginePool: VRAM insufficient for %d MB, evicting...", require_mb)
        from src.v6.gpu_monitor import auto_evict
        evicted = auto_evict(require_mb)
        # Actually unload evicted engines
        for eid in evicted:
            entry = self._entries.get(eid)
            if entry and entry.state == EngineState.LOADED:
                await self._do_unload(entry)

    async def _evict_oldest(self, count: int) -> None:
        """Evict *count* LRU engines."""
        loaded = sorted(
            [e for e in self._entries.values() if e.state == EngineState.LOADED],
            key=lambda e: e.last_used_at,
        )
        for entry in loaded[:count]:
            logger.info("EnginePool: evicting '%s' (LRU, idle %.1fs)",
                         entry.engine_id, monotonic() - entry.last_used_at)
            await self._do_unload(entry)

    async def _idle_unload(self, engine_id: str) -> None:
        """Called by timer to unload an idle engine."""
        entry = self._entries.get(engine_id)
        if entry is None or entry.state != EngineState.LOADED:
            return
        idle_sec = monotonic() - entry.last_used_at
        if idle_sec < self._idle_timeout * 0.9:
            return  # timer fired but engine was re-acquired
        logger.info("EnginePool: idle-unloading '%s' (idle %.1fs)", engine_id, idle_sec)
        await self._do_unload(entry)

    async def _do_unload(self, entry: _PoolEntry) -> None:
        entry.state = EngineState.EVICTING
        self._cancel_idle_timer(entry)
        try:
            if entry.unloader and entry.instance:
                await entry.unloader(entry.instance)
            mark_unloaded(entry.engine_id)
            entry.instance = None
            entry.state = EngineState.UNLOADED
            logger.info("EnginePool: unloaded '%s'", entry.engine_id)

            # Verify VRAM was actually released
            await self._verify_vram_released(entry)
        except Exception:
            entry.state = EngineState.FAILED
            mark_unloaded(entry.engine_id)
            logger.exception("EnginePool: unload error for '%s'", entry.engine_id)

    async def _verify_vram_released(self, entry: _PoolEntry) -> None:
        """Wait briefly for GPU VRAM to drain after unload."""
        import time as _time
        gpu = query_nvidia_smi()
        # If there was significant VRAM claimed and it's still held, log a warning
        if entry.vram_mb > 1000:
            await asyncio.sleep(2)  # Brief pause for GPU driver cleanup
            gpu_after = query_nvidia_smi()
            if gpu_after.free_mb < gpu.free_mb + entry.vram_mb * 0.5:
                logger.warning(
                    "EnginePool: VRAM not fully released after unloading '%s' "
                    "(expected ~%d MB freed, free before=%d, after=%d)",
                    entry.engine_id, entry.vram_mb,
                    gpu.free_mb, gpu_after.free_mb,
                )

    @staticmethod
    def _cancel_idle_timer(entry: _PoolEntry) -> None:
        if entry._release_handle:
            entry._release_handle.cancel()
            entry._release_handle = None


# ── singleton ─────────────────────────────────────────────────────────────

_pool: Optional[EnginePool] = None


def get_engine_pool() -> EnginePool:
    global _pool
    if _pool is None:
        _pool = EnginePool()
    return _pool
