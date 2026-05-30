"""Local Engine Pool — ComfyUI interface (mock mode)."""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from src.v6.models.task import GenerationTask, TaskOutputs, TaskMetadata, TaskStatus
from src.v6.store import get_task_store

logger = logging.getLogger(__name__)

# Mock output paths
MOCK_OUTPUTS = {
    "video_final": TaskOutputs(
        video="/mnt/agents/output/{task_id}/final.mp4",
        thumbnail="/mnt/agents/output/{task_id}/thumb.jpg",
    ),
    "video_preview": TaskOutputs(
        video="/mnt/agents/output/{task_id}/preview.mp4",
        thumbnail="/mnt/agents/output/{task_id}/thumb.jpg",
    ),
    "image_draw": TaskOutputs(
        image="/mnt/agents/output/{task_id}/render.png",
        thumbnail="/mnt/agents/output/{task_id}/thumb.jpg",
    ),
    "image_refine": TaskOutputs(
        image="/mnt/agents/output/{task_id}/refined.png",
    ),
    "tts": TaskOutputs(
        audio="/mnt/agents/output/{task_id}/voice.wav",
    ),
    "music": TaskOutputs(
        audio="/mnt/agents/output/{task_id}/bgm.wav",
    ),
    "sfx": TaskOutputs(
        audio="/mnt/agents/output/{task_id}/sfx.wav",
    ),
    "upscale": TaskOutputs(
        image="/mnt/agents/output/{task_id}/upscaled.png",
    ),
    "face_restore": TaskOutputs(
        image="/mnt/agents/output/{task_id}/face_restored.png",
    ),
    "image_to_3d": TaskOutputs(
        image="/mnt/agents/output/{task_id}/model.glb",
    ),
}


class LocalPool:
    """Manages local ComfyUI execution. Mock mode returns preset results."""

    def __init__(self) -> None:
        self._running = False
        self._worker_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the background worker that processes the queue."""
        if self._running:
            return
        self._running = True
        self._worker_task = asyncio.create_task(self._worker_loop())
        logger.info("LocalPool worker started (mock mode)")

    async def stop(self) -> None:
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

    async def _worker_loop(self) -> None:
        store = get_task_store()
        while self._running:
            try:
                # Poll queue
                try:
                    task_id = await asyncio.wait_for(store._queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                task = await store.get(task_id)
                if not task or task.status == TaskStatus.CANCELLED:
                    continue

                # Process task in mock mode
                await self._process_mock(task)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("LocalPool worker error: %s", e)

    async def _process_mock(self, task: GenerationTask) -> None:
        """Simulate task execution with mock outputs."""
        store = get_task_store()

        # Mark running
        await store.update(task.task_id, status=TaskStatus.RUNNING, progress=0.0)

        # Simulate progress
        for pct in [25, 50, 75, 90]:
            await asyncio.sleep(0.3)  # mock delay
            await store.update(task.task_id, progress=float(pct))

        # Resolve output paths
        type_key = task.type.value
        template = MOCK_OUTPUTS.get(type_key, TaskOutputs())
        outputs = TaskOutputs(
            **{
                k: v.format(task_id=task.task_id)
                for k, v in template.model_dump().items()
                if v
            }
        )

        metadata = TaskMetadata(
            seed=task.params.get("seed", 42),
            cost_usd=0.0,
            inference_time_sec=1.2,
            gpu_memory_peak_gb=8.0,
            model_name="mock-model",
        )

        await store.update(
            task.task_id,
            status=TaskStatus.COMPLETED,
            outputs=outputs,
            metadata=metadata,
            progress=100.0,
        )
        logger.info("Mock task completed: %s", task.task_id)

    def health(self) -> dict[str, Any]:
        return {
            "available": self._running,
            "vram_total_mb": 24576,
            "vram_available_mb": 24576,
            "gpu_utilization_pct": 0.0,
        }


# Singleton
_pool: Optional[LocalPool] = None


def get_local_pool() -> LocalPool:
    global _pool
    if _pool is None:
        _pool = LocalPool()
    return _pool
