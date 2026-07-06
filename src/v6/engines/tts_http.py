"""Triple-Track TTS Engine — routes TTS tasks to the unified TTS server.

Unified server (tts_unified_server.py) runs inside gold-team container on a
single port. It manages three backends internally with lazy-load / idle-unload:

  - 中文轨: GPT-SoVITS  (角色/IP克隆, ~4 GB VRAM)
  - 英文轨: Chatterbox    (Chatterbox-Turbo, ~2 GB VRAM)
  - 双语轨: CosyVoice     (CosyVoice-300M, ~2.5 GB VRAM)

This engine talks to the unified server via HTTP, same API as before.
The key difference from the old tts_http.py: only ONE server, not three.

Lifecycle mirrors :class:`BaseEngine`: submit → poll → get_output.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import time
import uuid
from typing import Any, Optional

import httpx

from src.v6.engines.base import BackendType, BaseEngine, EngineCapabilities, EngineStatus

logger = logging.getLogger(__name__)

OUTPUT_ROOT = os.environ.get("KAIS_OUTPUT_ROOT", "/mnt/agents/output")

# ── Unified server config (single endpoint) ─────────────────────────────────
UNIFIED_HOST = os.environ.get("TTS_UNIFIED_HOST", "localhost")
UNIFIED_PORT = int(os.environ.get("TTS_UNIFIED_PORT", "9880"))
UNIFIED_BASE = f"http://{UNIFIED_HOST}:{UNIFIED_PORT}"

# Language detection
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]")
_LATIN_RE = re.compile(r"[a-zA-Z]")


class TTSHttpJob:
    """Tracks a single TTS job."""

    __slots__ = (
        "job_id", "params", "status", "progress",
        "output_path", "error", "backend", "track",
        "duration_sec", "submitted_at",
    )

    def __init__(self, job_id: str, params: dict) -> None:
        self.job_id = job_id
        self.params = params
        self.status: str = "queued"
        self.progress: float = 0.0
        self.output_path: str = ""
        self.error: str = ""
        self.backend: str = ""
        self.track: str = ""
        self.duration_sec: float = 0.0
        self.submitted_at: float = 0.0


class TripleTrackTTSEngine(BaseEngine):
    """TTS engine that talks to the unified TTS server (lazy-load, single process).

    All three backends live inside the unified server — they are loaded on
    demand and automatically unloaded after idle timeout.
    """

    def __init__(self, output_root: str = OUTPUT_ROOT) -> None:
        self._output_root = output_root
        self._jobs: dict[str, TTSHttpJob] = {}
        self._client: Optional[httpx.AsyncClient] = None
        self._server_healthy: bool = False
        self._last_health_check: float = 0.0

    @property
    def name(self) -> str:
        return "TTS Triple-Track (Unified Lazy-Load)"

    @property
    def engine_id(self) -> str:
        return "tts-triple-track"

    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            supported_types=["tts"],
            max_duration_sec=300.0,
            vram_total_mb=8500,   # max if all 3 loaded
            vram_available_mb=24576,  # 3090 24GB
            models=["gpt_sovits", "chatterbox", "cosyvoice", "auto"],
        )

    @property
    def backend_type(self) -> BackendType:
        return BackendType.SUBPROCESS

    # ── BaseEngine lifecycle ──────────────────────────────────────────

    async def start(self) -> None:
        """Initialise HTTP client."""
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=5.0, read=120.0, write=30.0, pool=10.0),
        )
        # Check if unified server is reachable
        try:
            resp = await self._client.get(f"{UNIFIED_BASE}/health", timeout=5.0)
            resp.raise_for_status()
            self._server_healthy = True
            logger.info("Unified TTS server reachable at %s", UNIFIED_BASE)
        except Exception:
            self._server_healthy = False
            logger.warning(
                "Unified TTS server not reachable at %s — "
                "TTS tasks will fail until server is started. "
                "Start with: python scripts/tts_unified_server.py",
                UNIFIED_BASE,
            )

    async def stop(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
        self._jobs.clear()

    # ── Task submission ────────────────────────────────────────────────

    async def submit(
        self,
        workflow: dict[str, Any],
        params: dict[str, Any] | None = None,
    ) -> str:
        """Submit a TTS task to the unified server.

        The server handles track selection and lazy-loading internally.
        This engine supports both 'auto' mode (server decides) and
        explicit backend selection.
        """
        if not self._client:
            raise RuntimeError("TripleTrackTTSEngine not started")

        job_id = str(uuid.uuid4())[:12]
        params = params or {}
        task_id = params.get("task_id", job_id)

        text = workflow.get("text", "")
        voice = workflow.get("voice", "default")
        speed = workflow.get("speed", 1.0)
        backend = workflow.get("backend", "auto")
        reference_audio = workflow.get("reference_audio", "")
        output_path = workflow.get("output_path", "")

        if not text:
            raise ValueError("TTS workflow requires 'text'")

        if not output_path:
            output_path = os.path.join(self._output_root, task_id, "voice.wav")

        job = TTSHttpJob(job_id=job_id, params={
            "text": text, "voice": voice, "speed": speed,
            "backend": backend, "reference_audio": reference_audio,
            "output_path": output_path, "task_id": task_id,
        })
        self._jobs[job_id] = job

        payload: dict[str, Any] = {
            "text": text,
            "voice": voice,
            "speed": speed,
            "backend": backend,
            "output_path": output_path,
        }
        if reference_audio:
            payload["reference_audio"] = reference_audio

        try:
            resp = await self._client.post(
                f"{UNIFIED_BASE}/tts",
                json=payload,
                timeout=120.0,  # Long timeout for cold-start
            )
            resp.raise_for_status()
            body = resp.json()

            if "error" in body:
                job.status = "failed"
                job.error = body["error"]
            else:
                job.status = "completed"
                job.output_path = body.get("output_path", output_path)
                job.backend = body.get("backend", "unknown")
                job.track = body.get("track", backend)
                job.duration_sec = body.get("duration_sec", 0.0)
                job.progress = 100.0

            job.submitted_at = time.monotonic()
            logger.info(
                "TTS job %s → track=%s backend=%s text='%s' (%.1fs)",
                job_id, job.track, job.backend, text[:60],
                time.monotonic() - job.submitted_at + job.duration_sec,
            )

        except httpx.ConnectError:
            job.status = "failed"
            job.error = f"Unified TTS server unreachable at {UNIFIED_BASE}"
            logger.error("TTS job %s: server unreachable", job_id)
        except Exception as e:
            job.status = "failed"
            job.error = str(e)[:500]
            logger.error("TTS job %s error: %s", job_id, e)

        return job_id

    # ── Poll / Output ──────────────────────────────────────────────────

    async def poll(self, engine_job_id: str) -> dict[str, Any]:
        """Poll job status. Since unified server is synchronous, jobs
        are usually completed by the time submit returns."""
        job = self._jobs.get(engine_job_id)
        if not job:
            return {"status": "failed", "progress": 0.0, "error": "Unknown job ID"}
        result: dict[str, Any] = {"status": job.status, "progress": job.progress}
        if job.status == "failed":
            result["error"] = job.error
        return result

    async def get_output(self, engine_job_id: str) -> dict[str, Any]:
        """Return output artifacts for a completed job."""
        job = self._jobs.get(engine_job_id)
        if not job or job.status != "completed":
            return {"outputs": []}

        output_path = job.output_path
        if not output_path or not os.path.isfile(output_path):
            return {"outputs": []}

        artifacts = [{
            "url": f"file://{output_path}",
            "path": output_path,
            "type": "audio",
            "format": "wav" if output_path.endswith(".wav") else "mp3",
            "backend": job.backend,
            "track": job.track,
            "duration_sec": job.duration_sec,
        }]
        return {"outputs": artifacts}

    async def cancel(self, engine_job_id: str) -> bool:
        """Cancel a TTS job (no-op for sync server, but marks as failed)."""
        job = self._jobs.get(engine_job_id)
        if not job or job.status not in ("queued", "running"):
            return False
        job.status = "failed"
        job.error = "Cancelled"
        return True

    # ── Health ─────────────────────────────────────────────────────────

    async def health(self) -> dict[str, Any]:
        """Check unified server health."""
        if not self._client:
            return {"status": EngineStatus.OFFLINE.value, "available": False}

        try:
            resp = await self._client.get(f"{UNIFIED_BASE}/health", timeout=5.0)
            resp.raise_for_status()
            body = resp.json()
            self._server_healthy = body.get("status") == "healthy"
            return {
                "status": EngineStatus.ONLINE.value if self._server_healthy else EngineStatus.OFFLINE.value,
                "available": self._server_healthy,
                "mode": "unified-lazy-load",
                "server_url": UNIFIED_BASE,
                **{k: v for k, v in body.items() if k != "status"},
            }
        except Exception:
            self._server_healthy = False
            return {
                "status": EngineStatus.OFFLINE.value,
                "available": False,
                "server_url": UNIFIED_BASE,
                "error": "Server unreachable",
            }

    # ── Language detection (for logging, actual routing is server-side) ──

    @staticmethod
    def detect_language(text: str) -> str:
        has_cjk = bool(_CJK_RE.search(text))
        has_latin = bool(_LATIN_RE.search(text))
        if has_cjk and not has_latin:
            return "zh"
        if has_latin and not has_cjk:
            return "en"
        return "auto"
