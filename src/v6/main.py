"""kais-gold-team V6.0 — FastAPI application entry point."""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI

from src.v6.engine.local_pool import get_local_pool
from src.v6.engine_pool import EnginePool, get_engine_pool
from src.v6.executor import get_executor
from src.v6.engines.mock import MockEngine
from src.v6.engines.tts import TTSEngine
from src.v6.gpu_monitor import get_gpu_vram_usage
from src.v6.middleware.gpu_guard import GPUGuardMiddleware
from src.v6.routers import tasks, engines, events, health

# GPU management routes
from src.v6.routes.v1.gpu import router as gpu_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ComfyUI integration — controlled via env vars
COMFYUI_ENABLED = os.environ.get("COMFYUI_ENABLED", "false").lower() in ("true", "1", "yes")
COMFYUI_HOST = os.environ.get("COMFYUI_HOST", "127.0.0.1")
COMFYUI_PORT = int(os.environ.get("COMFYUI_PORT", "8188"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    executor = get_executor()

    # Always register mock engine for development
    executor.register_engine(MockEngine())

    # Register TTS engine (CosyVoice / edge-tts)
    try:
        tts_engine = TTSEngine()
        await tts_engine.start()
        executor.register_engine(tts_engine)
        logger.info("TTS engine registered")
    except Exception as e:
        logger.warning("TTS engine init failed: %s", e)

    # Register ComfyUI engine if available
    if COMFYUI_ENABLED:
        try:
            from src.v6.engines.comfyui import ComfyUIEngine
            comfyui = ComfyUIEngine(host=COMFYUI_HOST, port=COMFYUI_PORT)
            await comfyui.start()
            health = await comfyui.health()
            if health.get("available"):
                executor.register_engine(comfyui)
                logger.info("ComfyUI engine registered (online) → %s:%s", COMFYUI_HOST, COMFYUI_PORT)
            else:
                logger.warning("ComfyUI engine offline at %s:%s, using mock only", COMFYUI_HOST, COMFYUI_PORT)
        except ImportError:
            logger.warning("ComfyUIEngine not available, skipping")
        except Exception as e:
            logger.warning("ComfyUI engine init failed: %s", e)

    # Register cloud engines (Jimeng/Kling/Seedance)
    try:
        from src.v6.engines.cloud_jimeng import JimengEngine
        from src.v6.engines.cloud_kling import KlingEngine
        from src.v6.engines.cloud_seedance import SeedanceEngine

        for cloud_cls in [JimengEngine, KlingEngine, SeedanceEngine]:
            try:
                cloud_engine = cloud_cls()
                await cloud_engine.start()
                executor.register_engine(cloud_engine)
                configured = "✓" if cloud_engine.is_configured else "✗"
                logger.info("Cloud engine registered: %s [%s configured]",
                            cloud_engine.engine_id, configured)
            except Exception as e:
                logger.warning("Cloud engine %s init failed: %s", cloud_cls.__name__, e)
    except ImportError as e:
        logger.warning("Cloud engines not available: %s", e)

    await executor.start()

    # Register local Docker engines from engines/*.yaml (32-node routing table)
    # Engines are registered with the EnginePool for lazy GPU loading
    try:
        from src.v6.config.engine_registry import VRAM_ESTIMATES, build_engine_registry
        from src.v6.engines.docker_base import DockerAPIEngine
        engines_dir = Path(__file__).resolve().parent.parent.parent / "engines"
        if engines_dir.exists():
            local_engines = build_engine_registry(
                engines_dir=engines_dir,
                workspace="/workspace",
            )
            # Group unique engine instances by engine_id to avoid duplicate registrations
            seen_ids: dict[str, object] = {}
            for task_type, engine in local_engines.items():
                if engine.engine_id not in seen_ids:
                    seen_ids[engine.engine_id] = engine

            pool = get_engine_pool()
            for eid, engine_instance in seen_ids.items():
                vram_mb = 0
                # Derive vram from engine name
                for name_key, est in VRAM_ESTIMATES.items():
                    if name_key in eid.lower():
                        vram_mb = est
                        break
                if vram_mb == 0:
                    vram_mb = 4000  # default estimate

                # Build async loader
                async def _make_loader(eng=engine_instance):
                    async def _loader():
                        await eng.start()
                        return eng
                    return _loader

                # Build async unloader
                async def _make_unloader(eng=engine_instance):
                    async def _unloader(_instance):
                        await eng.stop()
                    return _unloader

                pool.register(
                    engine_id=eid,
                    loader=await _make_loader(),
                    vram_mb=vram_mb,
                    unloader=await _make_unloader(),
                )

            # Still register all task_type mappings with executor for routing
            for task_type, engine in local_engines.items():
                executor.register_engine(engine)

            logger.info("Registered %d local engines in EnginePool, %d task mappings from %s",
                        len(seen_ids), len(local_engines), engines_dir)
    except Exception as e:
        logger.warning("Local engine registry / pool failed: %s", e)

    local_pool = None
    # Only start legacy local_pool if no real engines available
    has_real_engine = any(e.engine_id != "mock" for e in executor.list_engines())
    if not has_real_engine:
        local_pool = get_local_pool()
        await local_pool.start()
        logger.info("Started local_pool (mock fallback — no real engines)")
    else:
        logger.info("Real engines available, skipping local_pool mock worker")

    logger.info("kais-gold-team V6.0 started (engines: %s)", [e.engine_id for e in executor.list_engines()])
    yield
    # Shutdown
    await executor.stop()
    if local_pool:
        await local_pool.stop()
    # Unload all engines from GPU
    try:
        pool = get_engine_pool()
        count = await pool.unload_all()
        if count:
            logger.info("EnginePool: unloaded %d engine(s) on shutdown", count)
    except Exception:
        pass

    logger.info("kais-gold-team V6.0 stopped")


app = FastAPI(
    title="kais-gold-team",
    description="Unified Execution Agent for KAIS AIGC Platform V6.0",
    version="6.0.0",
    lifespan=lifespan,
)

# Register routers
app.include_router(health.router)
app.include_router(tasks.router)
app.include_router(engines.router)
app.include_router(events.router)
app.include_router(gpu_router)

# GPU guard middleware — auto-evict on OOM
app.add_middleware(GPUGuardMiddleware)


if __name__ == "__main__":
    uvicorn.run(
        "src.v6.main:app",
        host="127.0.0.1",
        port=8002,
        reload=True,
    )
