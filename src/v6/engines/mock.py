"""Mock execution engine — simulates GPU tasks for development/testing."""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from src.v6.engines.base import BaseEngine, EngineCapabilities, EngineStatus

logger = logging.getLogger(__name__)

MOCK_DELAY_SEC = 3.0
MOCK_PROGRESS_STEPS = [20.0, 40.0, 60.0, 80.0, 95.0]

# Simulated output per task type
_MOCK_OUTPUTS: dict[str, list[dict[str, Any]]] = {
    "video_final": [
        {"url": "/mock/output/{job_id}/final.mp4", "type": "video", "format": "mp4"},
        {"url": "/mock/output/{job_id}/thumb.jpg", "type": "image", "format": "jpg"},
    ],
    "video_preview": [
        {"url": "/mock/output/{job_id}/preview.mp4", "type": "video", "format": "mp4"},
    ],
    "image_draw": [
        {"url": "/mock/output/{job_id}/render.png", "type": "image", "format": "png"},
    ],
    "image_refine": [
        {"url": "/mock/output/{job_id}/refined.png", "type": "image", "format": "png"},
    ],
    "tts": [
        {"url": "/mock/output/{job_id}/voice.wav", "type": "audio", "format": "wav"},
    ],
    "music": [
        {"url": "/mock/output/{job_id}/bgm.wav", "type": "audio", "format": "wav"},
    ],
    "sfx": [
        {"url": "/mock/output/{job_id}/sfx.wav", "type": "audio", "format": "wav"},
    ],
    "upscale": [
        {"url": "/mock/output/{job_id}/upscaled.png", "type": "image", "format": "png"},
    ],
    "face_restore": [
        {"url": "/mock/output/{job_id}/face.png", "type": "image", "format": "png"},
    ],
    "image_to_3d": [
        {"url": "/mock/output/{job_id}/model.glb", "type": "model", "format": "glb"},
    ],
}


class MockEngine(BaseEngine):
    """Mock engine for development — simulates execution with delay."""

    def __init__(self, delay_sec: float = MOCK_DELAY_SEC) -> None:
        self._delay = delay_sec
        # job_id → {status, progress, outputs, error}
        self._jobs: dict[str, dict[str, Any]] = {}

    @property
    def name(self) -> str:
        return "Mock Engine"

    @property
    def engine_id(self) -> str:
        return "mock"

    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            supported_types=list(_MOCK_OUTPUTS.keys()),
            max_resolution=(4096, 4096),
            max_duration_sec=60.0,
            vram_total_mb=24576,
            vram_available_mb=24576,
            models=["mock-model-v1"],
        )

    async def submit(self, workflow: dict[str, Any], params: dict[str, Any] | None = None) -> str:
        import uuid
        job_id = params.get("task_id", str(uuid.uuid4())) if params else str(uuid.uuid4())
        self._jobs[job_id] = {"status": "queued", "progress": 0.0, "outputs": []}
        logger.info("MockEngine: submitted job %s", job_id)

        # Start background simulation
        asyncio.create_task(self._simulate(job_id, params))
        return job_id

    async def _simulate(self, job_id: str, params: dict[str, Any] | None) -> None:
        """Simulate execution with progress steps."""
        self._jobs[job_id]["status"] = "running"

        step_delay = self._delay / max(len(MOCK_PROGRESS_STEPS), 1)
        for pct in MOCK_PROGRESS_STEPS:
            await asyncio.sleep(step_delay)
            if self._jobs[job_id]["status"] == "cancelled":
                return
            self._jobs[job_id]["progress"] = pct

        # Determine task type for mock output
        task_type = "image_draw"
        if params and "type" in params:
            task_type = params["type"]

        templates = _MOCK_OUTPUTS.get(task_type, _MOCK_OUTPUTS["image_draw"])
        outputs = [
            {k: (v.format(job_id=job_id) if isinstance(v, str) else v) for k, v in t.items()}
            for t in templates
        ]

        self._jobs[job_id]["status"] = "completed"
        self._jobs[job_id]["progress"] = 100.0
        self._jobs[job_id]["outputs"] = outputs
        logger.info("MockEngine: job %s completed", job_id)

    async def poll(self, engine_job_id: str) -> dict[str, Any]:
        job = self._jobs.get(engine_job_id)
        if not job:
            return {"status": "queued", "progress": 0.0}
        return {
            "status": job["status"],
            "progress": job["progress"],
            "error": job.get("error"),
        }

    async def get_output(self, engine_job_id: str) -> dict[str, Any]:
        job = self._jobs.get(engine_job_id, {})
        return {"outputs": job.get("outputs", [])}

    async def cancel(self, engine_job_id: str) -> bool:
        job = self._jobs.get(engine_job_id)
        if job and job["status"] in ("queued", "running"):
            job["status"] = "cancelled"
            job["error"] = "Cancelled by user"
            logger.info("MockEngine: job %s cancelled", engine_job_id)
            return True
        return False

    async def health(self) -> dict[str, Any]:
        return {
            "status": EngineStatus.ONLINE.value,
            "available": True,
            "vram_total_mb": 24576,
            "vram_available_mb": 24576,
            "gpu_utilization_pct": 0.0,
            "running_jobs": sum(
                1 for j in self._jobs.values() if j["status"] == "running"
            ),
        }
