"""GPU guard middleware — auto-evicts engines on OOM before API requests.

Intercepts incoming requests and ensures there is enough free VRAM for the
target engine.  If not, LRU engines are evicted first.
"""
from __future__ import annotations

import logging

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)

# VRAM headroom to keep free even when an engine is about to load (MB)
_HEADROOM_MB = 2048


class GPUGuardMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware that checks GPU VRAM on engine-related requests."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Only intercept task submission / engine acquire endpoints
        path = request.url.path
        if path.startswith("/api/v1/tasks") and request.method in ("POST", "PUT"):
            await self._ensure_vram_headroom()
        return await call_next(request)

    async def _ensure_vram_headroom(self) -> None:
        """Evict LRU engines until at least _HEADROOM_MB is free."""
        try:
            from src.v6.gpu_monitor import get_gpu_vram_usage, can_allocate
            info = get_gpu_vram_usage()
            free_mb = info.get("free_mb", 0)

            if free_mb >= _HEADROOM_MB:
                return

            logger.warning(
                "GPUGuard: low VRAM (free=%d MB, headroom=%d MB), evicting...",
                free_mb, _HEADROOM_MB,
            )

            from src.v6.gpu_monitor import auto_evict
            from src.v6.engine_pool import get_engine_pool

            evicted = auto_evict(_HEADROOM_MB - free_mb + 1024)
            for eid in evicted:
                try:
                    await get_engine_pool().unload(eid)
                except Exception:
                    logger.warning("GPUGuard: failed to unload '%s'", eid, exc_info=True)

        except ImportError:
            pass  # gpu_monitor / engine_pool not available yet (startup race)
        except Exception:
            logger.warning("GPUGuard: VRAM check failed", exc_info=True)
