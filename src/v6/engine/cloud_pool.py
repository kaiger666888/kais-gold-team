"""Cloud Engine Pool — dispatches to real cloud engines (Kling/Jimeng/Seedance).

When API keys are configured, tasks are routed to real cloud APIs.
When keys are missing, falls back to mock mode with a clear warning.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Optional

from src.v6.models.task import (
    GenerationTask,
    TaskMetadata,
    TaskOutputs,
    TaskStatus,
)
from src.v6.store import get_task_store

logger = logging.getLogger(__name__)

# ─── Cloud Provider Registry ──────────────────────────────────────

CLOUD_PROVIDERS = {
    "kling": {
        "name": "可灵 (Kling)",
        "available": True,
        "supported_types": ["video_final", "video_preview", "image_draw"],
        "engine_class": "src.v6.engines.cloud_kling.KlingEngine",
        "env_required": ["KLING_ACCESS_KEY", "KLING_SECRET_KEY"],
    },
    "jimeng": {
        "name": "即梦 (Jimeng)",
        "available": True,
        "supported_types": ["image_draw", "image_refine", "video_final"],
        "engine_class": "src.v6.engines.cloud_jimeng.JimengEngine",
        "env_required": ["JIMENG_API_KEY", "JIMENG_SESSION_ID"],  # either one
    },
    "seedance": {
        "name": "Seedance",
        "available": True,
        "supported_types": ["video_final", "video_preview"],
        "engine_class": "src.v6.engines.cloud_seedance.SeedanceEngine",
        "env_required": ["JIMENG_API_KEY", "JIMENG_SESSION_ID"],  # same as jimeng
    },
    "runway": {
        "name": "Runway",
        "available": False,
        "supported_types": ["video_final", "video_preview"],
        "engine_class": None,
        "env_required": ["RUNWAY_API_KEY"],
    },
    "luma": {
        "name": "Luma",
        "available": False,
        "supported_types": ["video_final", "video_preview"],
        "engine_class": None,
        "env_required": ["LUMA_API_KEY"],
    },
}


class CloudPool:
    """Cloud API pool. Dispatches to real engines when configured, mock otherwise."""

    def __init__(self) -> None:
        self._engines: dict[str, Any] = {}
        self._initialized = False

    async def _ensure_engines(self) -> None:
        """Lazily initialize cloud engine instances."""
        if self._initialized:
            return

        for pid, info in CLOUD_PROVIDERS.items():
            if not info.get("engine_class"):
                continue
            try:
                # Import and instantiate engine
                module_path, class_name = info["engine_class"].rsplit(".", 1)
                import importlib
                module = importlib.import_module(module_path)
                engine_cls = getattr(module, class_name)
                engine = engine_cls()
                await engine.start()
                self._engines[pid] = engine

                configured = engine.is_configured
                status = "✓ ready" if configured else "✗ no API key (will mock)"
                logger.info("CloudPool [%s] %s — %s", pid, info["name"], status)
            except Exception as e:
                logger.warning("CloudPool [%s] failed to init: %s", pid, e)

        self._initialized = True

    def _pick_engine(self, task: GenerationTask) -> tuple[Optional[Any], str]:
        """Pick the best cloud engine for the task type."""
        task_type = task.type.value if hasattr(task.type, 'value') else str(task.type)

        # Priority: jimeng (most supported) > kling > seedance
        priority = ["jimeng", "kling", "seedance"]
        for pid in priority:
            engine = self._engines.get(pid)
            if engine and engine.is_configured:
                info = CLOUD_PROVIDERS.get(pid, {})
                if task_type in info.get("supported_types", []):
                    return engine, pid

        return None, "mock"

    async def submit(self, task: GenerationTask) -> None:
        """Submit task to cloud engine."""
        store = get_task_store()
        await self._ensure_engines()

        engine, provider = self._pick_engine(task)

        if engine and engine.is_configured:
            # Real cloud engine
            await store.update(
                task.task_id,
                status=TaskStatus.RUNNING,
                engine_id=f"cloud-{provider}",
                progress=0.0,
            )

            try:
                # Build workflow from task params
                workflow = {
                    "prompt": task.params.get("prompt", ""),
                    "width": task.params.get("width", 1024),
                    "height": task.params.get("height", 1024),
                    "negative_prompt": task.params.get("negative_prompt", ""),
                }
                params = {
                    "task_id": task.task_id,
                    "type": task.type.value if hasattr(task.type, 'value') else str(task.type),
                    "model": task.params.get("model"),
                    "duration": task.params.get("duration"),
                    "ratio": task.params.get("ratio"),
                }

                job_id = await engine.submit(workflow, params)
                logger.info("CloudPool [%s] submitted task %s as job %s",
                            provider, task.task_id, job_id)

                # Poll in background
                asyncio.create_task(self._poll_real_engine(task, engine, job_id, provider))

            except Exception as e:
                logger.error("CloudPool [%s] submit failed: %s", provider, e)
                await store.update(
                    task.task_id,
                    status=TaskStatus.FAILED,
                    error=f"Cloud engine error ({provider}): {e}",
                )
        else:
            # Fallback to mock
            logger.warning("CloudPool: no configured engine for task %s, using mock", task.task_id)
            await store.update(
                task.task_id,
                status=TaskStatus.RUNNING,
                engine_id="cloud-mock",
                progress=0.0,
            )
            asyncio.create_task(self._mock_execute(task))

    async def _poll_real_engine(self, task: GenerationTask, engine: Any,
                                 job_id: str, provider: str) -> None:
        """Poll real cloud engine until completion."""
        store = get_task_store()

        try:
            max_wait = 600  # 10 min
            import time
            start = time.time()

            while time.time() - start < max_wait:
                result = await engine.poll(job_id)
                status = result.get("status", "running")
                progress = result.get("progress", 0.0)

                await store.update(task.task_id, progress=progress)

                if status == "completed":
                    output_data = await engine.get_output(job_id)
                    artifacts = output_data.get("outputs", [])

                    outputs = TaskOutputs()
                    metadata = TaskMetadata(
                        seed=task.params.get("seed", 0),
                        cost_usd=0.15,
                        inference_time_sec=time.time() - start,
                        model_name=f"cloud-{provider}",
                        cloud_task_id=job_id,
                    )

                    for a in artifacts:
                        if a["type"] == "video" and not outputs.video:
                            outputs.video = a["url"]
                        elif a["type"] == "image" and not outputs.image:
                            outputs.image = a["url"]
                            outputs.thumbnail = a["url"]

                    await store.update(
                        task.task_id,
                        status=TaskStatus.COMPLETED,
                        outputs=outputs,
                        metadata=metadata,
                        progress=100.0,
                    )
                    logger.info("CloudPool [%s] task %s completed", provider, task.task_id)
                    return

                if status == "failed":
                    error = result.get("error", "Unknown error")
                    await store.update(
                        task.task_id,
                        status=TaskStatus.FAILED,
                        error=f"Cloud engine ({provider}): {error}",
                    )
                    return

                await asyncio.sleep(2.0)

            # Timeout
            await store.update(
                task.task_id,
                status=TaskStatus.FAILED,
                error=f"Cloud engine ({provider}) timed out after {max_wait}s",
            )

        except Exception as e:
            await store.update(
                task.task_id,
                status=TaskStatus.FAILED,
                error=f"Cloud engine ({provider}) error: {e}",
            )

    async def _mock_execute(self, task: GenerationTask) -> None:
        """Mock fallback for when no cloud engine is configured."""
        store = get_task_store()

        try:
            await asyncio.sleep(1.0)
            await store.update(task.task_id, progress=30.0)
            await asyncio.sleep(1.0)
            await store.update(task.task_id, progress=70.0)
            await asyncio.sleep(0.5)

            outputs = TaskOutputs(
                video=f"/mnt/agents/output/{task.task_id}/cloud_final.mp4",
                thumbnail=f"/mnt/agents/output/{task.task_id}/cloud_thumb.jpg",
            )
            metadata = TaskMetadata(
                seed=task.params.get("seed", 999),
                cost_usd=0.0,
                inference_time_sec=2.5,
                model_name="cloud-mock",
                cloud_task_id=f"mock_{task.task_id}",
            )

            await store.update(
                task.task_id,
                status=TaskStatus.COMPLETED,
                outputs=outputs,
                metadata=metadata,
                progress=100.0,
            )
            logger.info("Cloud mock task completed: %s", task.task_id)

        except Exception as e:
            await store.update(
                task.task_id,
                status=TaskStatus.FAILED,
                error=str(e),
            )

    def health(self) -> dict[str, Any]:
        providers = []
        for pid, info in CLOUD_PROVIDERS.items():
            engine = self._engines.get(pid)
            is_configured = engine.is_configured if engine else False
            providers.append({
                "name": pid,
                "available": info["available"] and (is_configured or not info.get("engine_class")),
                "configured": is_configured,
                "env_required": info.get("env_required", []),
            })
        return {
            "available": any(p["available"] for p in providers),
            "active_providers": providers,
        }


# Singleton
_cloud_pool: Optional[CloudPool] = None


def get_cloud_pool() -> CloudPool:
    global _cloud_pool
    if _cloud_pool is None:
        _cloud_pool = CloudPool()
    return _cloud_pool
