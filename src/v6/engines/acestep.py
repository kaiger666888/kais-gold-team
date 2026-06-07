"""ACE-Step Engine — runs ACE-Step music generation internally via subprocess API server.

Like ComfyUIEngine, this starts a persistent ACE-Step API server as a background
process within the gold-team container and communicates via HTTP. This avoids
the need for a separate Docker container while keeping the model loaded in VRAM.

Lifecycle:
    start()  → launches ``python -m acestep.api_server`` subprocess
    submit() → POST /release_task with params
    poll()   → POST /query_result for status
    stop()   → terminates the subprocess
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from typing import Any

import httpx

from src.v6.engines.base import BaseEngine, EngineCapabilities, EngineStatus

logger = logging.getLogger(__name__)

OUTPUT_ROOT = os.environ.get("KAIS_OUTPUT_ROOT", "/mnt/agents/output")

# ACE-Step config from env
ACESTEP_ROOT = os.environ.get("ACESTEP_ROOT", "/opt/acestep")
ACESTEP_HOST = os.environ.get("ACESTEP_API_HOST", "127.0.0.1")
ACESTEP_PORT = int(os.environ.get("ACESTEP_API_PORT", "8010"))
ACESTEP_CONFIG = os.environ.get("ACESTEP_CONFIG_PATH", "acestep-v15-xl-turbo")
ACESTEP_CONFIG2 = os.environ.get("ACESTEP_CONFIG_PATH2", "acestep-v15-xl-sft")
ACESTEP_CHECKPOINTS = os.environ.get("ACESTEP_CHECKPOINTS", "/opt/acestep/checkpoints")

_TASK_TYPE_MAP = {
    "audio_generate": "text2music",
    "music_generation": "text2music",
    "music": "text2music",  # TaskType.MUSIC alias
    "music_cover": "cover",
    "music_remix": "text2music",
    "music_repaint": "repaint",
    "music_extract": "extract",
    "music_lego": "lego",
    "music_complete": "complete",
}


class ACEStepJob:
    def __init__(self, job_id: str, params: dict) -> None:
        self.job_id = job_id
        self.params = params
        self.status = "queued"
        self.progress = 0.0
        self.output_path = ""
        self.error = ""
        self.started_at = 0.0


class ACEStepEngine(BaseEngine):
    """ACE-Step music generation engine — internal subprocess + HTTP API."""

    def __init__(self) -> None:
        self._base_url = f"http://{ACESTEP_HOST}:{ACESTEP_PORT}"
        self._process: asyncio.subprocess.Process | None = None
        self._jobs: dict[str, ACEStepJob] = {}
        self._ready = False

    @property
    def name(self) -> str:
        return "ACE-Step Music Engine"

    @property
    def engine_id(self) -> str:
        return "acestep-internal"

    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            supported_types=["audio_generate", "music_generation", "music", "music_cover",
                             "music_repaint", "music_extract", "music_lego", "music_complete"],
            max_duration_sec=60.0,
            vram_total_mb=24576,
            vram_available_mb=24576,
            models=["acestep-v15-xl-turbo", "acestep-v15-xl-sft"],
        )

    async def start(self) -> None:
        if not os.path.isdir(ACESTEP_ROOT) or ACESTEP_ROOT == "/nonexistent":
            logger.warning("ACE-Step root not found at %s — engine disabled", ACESTEP_ROOT)
            return
        # Skip startup if checkpoint dir missing (avoids 120s block)
        if not os.path.isdir(ACESTEP_CHECKPOINTS):
            logger.warning("ACE-Step checkpoints not found at %s — engine disabled", ACESTEP_CHECKPOINTS)
            return

        env = os.environ.copy()
        env.update({
            "PYTHONPATH": f"{ACESTEP_ROOT}/app:{env.get('PYTHONPATH', '')}",
            "ACESTEP_CONFIG_PATH": ACESTEP_CONFIG,
            "ACESTEP_CONFIG_PATH2": ACESTEP_CONFIG2,
            "ACESTEP_API_HOST": ACESTEP_HOST,
            "ACESTEP_API_PORT": str(ACESTEP_PORT),
            "ACESTEP_CHECKPOINTS": ACESTEP_CHECKPOINTS,
            "ACESTEP_OFFLOAD_DIT_TO_CPU": os.environ.get("ACESTEP_OFFLOAD_DIT_TO_CPU", "true"),
            "ACESTEP_NO_INIT": "false",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HOME": "/tmp/hf_cache",
            "TRANSFORMERS_CACHE": "/tmp/hf_cache",
        })

        cmd = [
            "python", "-m", "acestep.api_server",
            "--host", ACESTEP_HOST,
            "--port", str(ACESTEP_PORT),
        ]

        logger.info("Starting ACE-Step API server: %s (config=%s)", " ".join(cmd), ACESTEP_CONFIG)
        self._process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=f"{ACESTEP_ROOT}/app",
        )
        asyncio.create_task(self._watch_process())

        # Wait for API server to become ready (up to 120s for model loading)
        for i in range(60):
            await asyncio.sleep(2)
            try:
                async with httpx.AsyncClient(timeout=5) as client:
                    resp = await client.get(f"{self._base_url}/health")
                    if resp.status_code == 200:
                        self._ready = True
                        logger.info("ACE-Step API server ready after %ds", (i + 1) * 2)
                        return
            except Exception:
                pass

        logger.warning("ACE-Step API server did not become ready within 120s")

    async def _watch_process(self) -> None:
        if not self._process:
            return
        stdout, stderr = await self._process.communicate()
        if self._process.returncode and self._process.returncode != 0:
            logger.error("ACE-Step process exited (rc=%d): %s",
                         self._process.returncode, stderr.decode()[-500:])

    async def stop(self) -> None:
        if self._process and self._process.returncode is None:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=10)
            except asyncio.TimeoutError:
                self._process.kill()
            self._ready = False

    async def submit(self, workflow: dict[str, Any], params: dict[str, Any] | None = None) -> str:
        if not self._ready:
            raise RuntimeError("ACE-Step engine not ready")

        params = params or {}
        task_id = params.get("task_id", str(uuid.uuid4())[:12])
        task_type = workflow.get("task_type", "audio_generate")
        api_task_type = _TASK_TYPE_MAP.get(task_type, "text2music")

        # Extract acestep-specific params
        ace_params = workflow.get("extra", {}).get("acestep", {})
        p = {**workflow, **ace_params}
        p.pop("extra", None)

        payload: dict[str, Any] = {
            "task_type": api_task_type,
            "prompt": p.get("prompt", ""),
            "lyrics": p.get("lyrics", ""),
            "thinking": p.get("thinking", True),
            "sample_mode": p.get("sample_mode", False),
            "sample_query": p.get("sample_query", ""),
            "seed": p.get("seed", -1),
            "use_random_seed": p.get("seed", -1) == -1,
            "audio_format": p.get("audio_format", "wav"),
            "batch_size": p.get("batch_size", 1),
        }

        for key in ("model", "global_caption", "use_format", "audio_duration",
                     "bpm", "key_scale", "time_signature", "vocal_language",
                     "inference_steps", "guidance_scale", "shift"):
            val = p.get(key)
            if val is not None:
                payload[key] = val

        ref_audio = p.get("reference_audio")
        if ref_audio:
            payload["reference_audio_path"] = (
                ref_audio if ref_audio.startswith("/") else f".assets/{task_id}/{ref_audio}"
            )

        src_audio = p.get("src_audio_path")
        if src_audio:
            payload["src_audio_path"] = src_audio if src_audio.startswith("/") else f".assets/{task_id}/{src_audio}"

        job = ACEStepJob(job_id=task_id, params={"payload": payload, "task_id": task_id})
        self._jobs[task_id] = job
        job.status = "running"
        job.started_at = time.monotonic()

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(f"{self._base_url}/release_task", json=payload)
            resp.raise_for_status()
            data = resp.json()

        if data.get("code") != 200:
            job.status = "failed"
            job.error = data.get("error", "unknown")
            raise RuntimeError(f"ACE-Step submit failed: {job.error}")

        engine_job_id = data["data"]["task_id"]
        job.params["engine_job_id"] = engine_job_id
        logger.info("ACE-Step job %s submitted (engine_id=%s, type=%s)", task_id, engine_job_id, api_task_type)

        # Background poller
        asyncio.create_task(self._poll_until_done(task_id))
        return task_id

    async def _poll_until_done(self, job_id: str) -> None:
        job = self._jobs.get(job_id)
        if not job:
            return

        engine_job_id = job.params.get("engine_job_id", "")
        timeout = 300
        start = time.monotonic()

        while time.monotonic() - start < timeout:
            await asyncio.sleep(3)
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    resp = await client.post(
                        f"{self._base_url}/query_result",
                        json={"task_id_list": [engine_job_id]},
                    )
                    resp.raise_for_status()
                    data = resp.json()

                jobs_data = data.get("data", [])
                if not jobs_data:
                    job.progress = 50.0
                    continue

                status = jobs_data[0].get("status")
                if status == 1:
                    job.status = "completed"
                    job.progress = 100.0
                    # Collect output paths
                    result = jobs_data[0]
                    audio_paths = result.get("audio_paths", [])
                    if audio_paths:
                        job.output_path = audio_paths[0]
                    logger.info("ACE-Step job %s completed in %.1fs", job_id, time.monotonic() - start)
                    return
                elif status == 2:
                    job.status = "failed"
                    job.error = result.get("error", "ACE-Step job failed")
                    return
                else:
                    job.progress = min(job.progress + 5, 95.0)

            except Exception as e:
                logger.warning("ACE-Step poll error for %s: %s", job_id, e)

        job.status = "failed"
        job.error = "Timeout"

    async def poll(self, engine_job_id: str) -> dict[str, Any]:
        job = self._jobs.get(engine_job_id)
        if not job:
            return {"status": "failed", "progress": 0.0, "error": "Unknown job ID"}
        result: dict[str, Any] = {"status": job.status, "progress": job.progress}
        if job.status == "failed":
            result["error"] = job.error
        return result

    async def get_output(self, engine_job_id: str) -> dict[str, Any]:
        job = self._jobs.get(engine_job_id)
        if not job or job.status != "completed":
            return {"outputs": []}

        output_path = job.output_path
        if not output_path or not os.path.isfile(output_path):
            return {"outputs": []}

        return {"outputs": [{
            "url": f"file://{output_path}",
            "path": output_path,
            "type": "audio",
            "format": "wav" if output_path.endswith(".wav") else "mp3",
            "duration_sec": round(time.monotonic() - job.started_at, 1),
        }]}

    async def cancel(self, engine_job_id: str) -> bool:
        job = self._jobs.get(engine_job_id)
        if not job or job.status not in ("queued", "running"):
            return False
        job.status = "failed"
        job.error = "Cancelled"
        return True

    async def health(self) -> dict[str, Any]:
        if not self._ready:
            return {"status": EngineStatus.OFFLINE.value, "available": False}

        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{self._base_url}/health")
                ok = resp.status_code == 200
        except Exception:
            ok = False

        return {
            "status": EngineStatus.ONLINE.value if ok else EngineStatus.OFFLINE.value,
            "available": ok,
            "url": self._base_url,
        }
