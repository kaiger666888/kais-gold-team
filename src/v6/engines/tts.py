"""TTS Engine — runs CosyVoice / edge-tts via subprocess for gold-team V6.

Unlike ComfyUIEngine which talks to ComfyUI via HTTP, this engine invokes
the standalone ``scripts/tts_infer.py`` script directly. This avoids the need
for a ComfyUI CosyVoice custom node while still producing real audio output.

Lifecycle:
    submit() → records params, returns job_id
    poll()   → checks subprocess status
    get_output() → returns artifact URLs for completed jobs
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from src.v6.engines.base import BaseEngine, EngineCapabilities, EngineStatus

logger = logging.getLogger(__name__)

# Output directory for TTS files
OUTPUT_ROOT = os.environ.get("KAIS_OUTPUT_ROOT", "/mnt/agents/output")

# Path to the TTS inference script (relative to repo root)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # src/v6/engines/
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
SCRIPT_PATH = os.path.join(_REPO_ROOT, "scripts", "tts_infer.py")


class TTSJob:
    """Tracks a single TTS subprocess job."""

    def __init__(self, job_id: str, params: dict) -> None:
        self.job_id = job_id
        self.params = params
        self.status: str = "queued"  # queued | running | completed | failed
        self.progress: float = 0.0
        self.output_path: str = ""
        self.error: str = ""
        self.backend: str = ""
        self.duration_sec: float = 0.0
        self.started_at: float = 0.0
        self.process: Optional[asyncio.subprocess.Process] = None


class TTSEngine(BaseEngine):
    """TTS engine that runs ``scripts/tts_infer.py`` via subprocess.

    Supports both CosyVoice (if installed) and edge-tts (fallback).
    """

    def __init__(self, output_root: str = OUTPUT_ROOT) -> None:
        self._output_root = output_root
        self._jobs: dict[str, TTSJob] = {}
        self._script_path = os.path.abspath(SCRIPT_PATH)
        self._python = os.environ.get("KAIS_TTS_PYTHON", "python3")

    @property
    def name(self) -> str:
        return "TTS Engine (CosyVoice/edge-tts)"

    @property
    def engine_id(self) -> str:
        return "tts-local"

    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            supported_types=["tts"],
            max_duration_sec=300.0,
            vram_total_mb=6144,
            vram_available_mb=6144,
            models=["cosyvoice-2-0.5b", "edge-tts"],
        )

    async def start(self) -> None:
        """Verify the TTS script exists."""
        if not os.path.isfile(self._script_path):
            logger.warning("TTS script not found at %s — TTS tasks will fail", self._script_path)
        else:
            logger.info("TTS engine ready, script: %s", self._script_path)

    async def stop(self) -> None:
        """Cancel all running jobs."""
        for job in self._jobs.values():
            if job.process and job.process.returncode is None:
                job.process.kill()
        self._jobs.clear()

    async def submit(self, workflow: dict[str, Any], params: dict[str, Any] | None = None) -> str:
        """Submit a TTS task.

        Args:
            workflow: Dict with keys: text, voice, speed, backend, output_path.
            params: Optional dict with task_id for output naming.

        Returns:
            Job ID for tracking.
        """
        job_id = str(uuid.uuid4())[:12]
        params = params or {}
        task_id = params.get("task_id", job_id)

        # Extract TTS params from workflow
        text = workflow.get("text", "")
        voice = workflow.get("voice", "default")
        speed = workflow.get("speed", 1.0)
        backend = workflow.get("backend", "auto")

        if not text:
            raise ValueError("TTS workflow requires 'text' parameter")

        # Determine output path
        output_path = workflow.get("output_path", "")
        if not output_path:
            output_path = os.path.join(self._output_root, task_id, "voice.wav")

        job = TTSJob(job_id=job_id, params={
            "text": text,
            "voice": voice,
            "speed": speed,
            "backend": backend,
            "output_path": output_path,
            "task_id": task_id,
        })
        self._jobs[job_id] = job

        # Start subprocess
        cmd = [
            self._python, self._script_path,
            "--text", text,
            "--output", output_path,
            "--voice", voice,
            "--speed", str(speed),
            "--backend", backend,
        ]

        logger.info("TTS job %s: submitting '%s' (voice=%s, backend=%s)", job_id, text[:50], voice, backend)

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            job.process = process
            job.status = "running"
            job.started_at = time.monotonic()

            # Fire-and-forget: background task to collect output
            asyncio.create_task(self._watch_job(job))

        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            logger.error("TTS job %s failed to start: %s", job_id, e)

        return job_id

    async def _watch_job(self, job: TTSJob) -> None:
        """Background task that waits for subprocess completion and parses output."""
        try:
            stdout, stderr = await job.process.communicate()
            elapsed = time.monotonic() - job.started_at

            if job.process.returncode == 0:
                # Parse JSON output from script
                import json
                try:
                    result = json.loads(stdout.decode().strip().split("\n")[-1])
                except (json.JSONDecodeError, IndexError):
                    result = {}

                job.status = "completed"
                job.progress = 100.0
                job.output_path = result.get("output_path", job.params.get("output_path", ""))
                job.backend = result.get("backend", "unknown")
                job.duration_sec = result.get("duration_sec", round(elapsed, 2))
                logger.info(
                    "TTS job %s completed in %.1fs (backend=%s, output=%s)",
                    job.job_id, elapsed, job.backend, job.output_path,
                )
            else:
                error_output = stderr.decode().strip() if stderr else stdout.decode().strip()
                job.status = "failed"
                job.error = error_output[:500]
                logger.error("TTS job %s failed (rc=%d): %s", job.job_id, job.process.returncode, error_output[:200])

        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            logger.error("TTS job %s watch error: %s", job.job_id, e)

    async def poll(self, engine_job_id: str) -> dict[str, Any]:
        """Poll TTS job status."""
        job = self._jobs.get(engine_job_id)
        if not job:
            return {"status": "failed", "progress": 0.0, "error": "Unknown job ID"}

        result: dict[str, Any] = {
            "status": job.status,
            "progress": job.progress,
        }
        if job.status == "failed":
            result["error"] = job.error
        return result

    async def get_output(self, engine_job_id: str) -> dict[str, Any]:
        """Get output artifacts for completed TTS job."""
        job = self._jobs.get(engine_job_id)
        if not job or job.status != "completed":
            return {"outputs": []}

        output_path = job.output_path
        if not output_path or not os.path.isfile(output_path):
            return {"outputs": []}

        # Build a file:// URL for local access, or relative path for HTTP serving
        artifacts = [{
            "url": f"file://{output_path}",
            "path": output_path,
            "type": "audio",
            "format": "wav" if output_path.endswith(".wav") else "mp3",
            "backend": job.backend,
            "duration_sec": job.duration_sec,
        }]
        return {"outputs": artifacts}

    async def cancel(self, engine_job_id: str) -> bool:
        """Cancel a running TTS job."""
        job = self._jobs.get(engine_job_id)
        if not job or job.status not in ("queued", "running"):
            return False

        if job.process and job.process.returncode is None:
            job.process.kill()
        job.status = "failed"
        job.error = "Cancelled"
        logger.info("TTS job %s cancelled", engine_job_id)
        return True

    async def health(self) -> dict[str, Any]:
        """Check TTS engine health."""
        script_ok = os.path.isfile(self._script_path)
        output_ok = os.path.isdir(self._output_root) or os.path.isdir(os.path.dirname(self._output_root))

        # Quick check: edge-tts available?
        try:
            import edge_tts  # noqa: F401
            backends = ["edge-tts"]
        except ImportError:
            backends = []

        # Check CosyVoice
        cosy_root = os.environ.get("COSYVOICE_ROOT", os.path.expanduser("~/CosyVoice"))
        if os.path.isdir(cosy_root):
            backends.append("cosyvoice")

        status = EngineStatus.ONLINE if script_ok else EngineStatus.OFFLINE
        return {
            "status": status.value,
            "available": script_ok,
            "backends": backends,
            "output_root": self._output_root,
            "script_path": self._script_path,
        }
