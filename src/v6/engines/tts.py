"""Three-Track TTS Engine System for kais-gold-team V6.

Architecture:
    Track 1 (中文轨): GPT-SoVITS — Chinese TTS, 3060Ti, :9880
    Track 2 (英文轨): Chatterbox-Turbo — English TTS, 3060Ti, :9881
    Track 3 (双语轨): CosyVoice 3.0 — Bilingual TTS, 3090, :9882

All three engines expose a common HTTP API:
    POST /tts  { text, ...params }  →  { audio_url, duration_sec }
    GET  /health                    →  { status, vram_used_mb }

The TTSTracker acts as a unified facade that auto-routes by language.

Lifecycle:
    Each TTS service runs independently as a FastAPI server.
    The engine communicates via HTTP — no subprocess spawning.
    If a service is down, the tracker returns a clear error with
    suggested fallback track.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import aiohttp

from src.v6.engines.base import BaseEngine, EngineCapabilities, EngineStatus

logger = logging.getLogger(__name__)

OUTPUT_ROOT = "/mnt/agents/output"


# ─── TTS Track Definition ───

class TTSTrack(str, Enum):
    ZH = "zh"          # GPT-SoVITS
    EN = "en"          # Chatterbox-Turbo
    BILINGUAL = "bilingual"  # CosyVoice


@dataclass
class TTSServiceConfig:
    """Configuration for a single TTS HTTP service."""
    name: str
    track: TTSTrack
    host: str = "127.0.0.1"
    port: int = 9880
    health_endpoint: str = "/health"
    tts_endpoint: str = "/tts"
    timeout_sec: float = 120.0
    vram_gb: float = 4.0
    gpu_id: int = 0

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def tts_url(self) -> str:
        return f"{self.base_url}{self.tts_endpoint}"

    @property
    def health_url(self) -> str:
        return f"{self.base_url}{self.health_endpoint}"


# Default service configs (matches engine YAML)
DEFAULT_SERVICES: dict[TTSTrack, TTSServiceConfig] = {
    TTSTrack.ZH: TTSServiceConfig(
        name="GPT-SoVITS",
        track=TTSTrack.ZH,
        port=9880,
        vram_gb=4.0,
        gpu_id=1,  # 3060Ti
        timeout_sec=60.0,
    ),
    TTSTrack.EN: TTSServiceConfig(
        name="Chatterbox-Turbo",
        track=TTSTrack.EN,
        port=9881,
        vram_gb=2.0,
        gpu_id=1,  # 3060Ti
        timeout_sec=60.0,
    ),
    TTSTrack.BILINGUAL: TTSServiceConfig(
        name="CosyVoice-3.0",
        track=TTSTrack.BILINGUAL,
        port=9882,
        vram_gb=6.0,
        gpu_id=0,  # 3090
        timeout_sec=120.0,
    ),
}


# ─── Language Detection ───

def detect_language(text: str) -> str:
    """Simple CJK-based language detection.

    Returns: 'zh' if >30% CJK characters, 'en' otherwise.
    This is intentionally simple — the pipeline caller can override.
    """
    cjk_count = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    if len(text) > 0 and cjk_count / len(text) > 0.3:
        return "zh"
    return "en"


# ─── HTTP TTS Job ───

@dataclass
class TTSJob:
    """Tracks a single TTS HTTP request."""
    job_id: str
    track: TTSTrack
    params: dict[str, Any]
    status: str = "queued"  # queued | running | completed | failed
    progress: float = 0.0
    output_path: str = ""
    error: str = ""
    duration_sec: float = 0.0
    service_name: str = ""
    submitted_at: float = 0.0


# ─── Individual Track Engine ───

class HTTPTTSEngine(BaseEngine):
    """HTTP-based TTS engine for a single track.

    Talks to the TTS service via HTTP POST /tts.
    Handles health checks, timeout, and error reporting.
    """

    def __init__(self, config: TTSServiceConfig, output_root: str = OUTPUT_ROOT) -> None:
        self._config = config
        self._output_root = output_root
        self._jobs: dict[str, TTSJob] = {}
        self._session: Optional[aiohttp.ClientSession] = None
        self._healthy: bool | None = None  # None=unchecked
        self._last_health_check: float = 0.0

    @property
    def name(self) -> str:
        return f"TTS-{self._config.track.value} ({self._config.name})"

    @property
    def engine_id(self) -> str:
        return f"tts-{self._config.track.value}"

    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            supported_types=[f"tts_{self._config.track.value}"],
            max_duration_sec=self._config.timeout_sec,
            vram_total_mb=int(self._config.vram_gb * 1024),
            vram_available_mb=int(self._config.vram_gb * 1024),
            models=[self._config.name],
        )

    @property
    def track(self) -> TTSTrack:
        return self._config.track

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self._config.timeout_sec)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def start(self) -> None:
        logger.info(
            "TTS engine %s initialized: %s (port %d, GPU %d, ~%.0fGB)",
            self._config.track.value, self._config.name,
            self._config.port, self._config.gpu_id, self._config.vram_gb,
        )
        # Initial health check (non-blocking)
        try:
            await self.health()
        except Exception as e:
            logger.warning("TTS engine %s initial health check failed: %s", self._config.track.value, e)

    async def stop(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
        self._jobs.clear()

    async def submit(self, workflow: dict[str, Any], params: dict[str, Any] | None = None) -> str:
        """Submit TTS via HTTP POST to the service."""
        job_id = str(uuid.uuid4())[:12]
        params = params or {}
        task_id = params.get("task_id", job_id)

        # Extract output path
        output_path = workflow.get("output_path", "")
        if not output_path:
            output_path = f"{self._output_root}/{task_id}/voice.wav"

        job = TTSJob(
            job_id=job_id,
            track=self._config.track,
            params=workflow,
            service_name=self._config.name,
            submitted_at=time.monotonic(),
        )
        self._jobs[job_id] = job

        # Build request payload
        payload = {k: v for k, v in workflow.items() if k != "output_path"}
        payload["output_path"] = output_path

        logger.info(
            "TTS-%s job %s: submitting to %s (text=%s)",
            self._config.track.value, job_id, self._config.tts_url,
            str(workflow.get("text", ""))[:60],
        )

        try:
            job.status = "running"
            session = await self._get_session()

            async with session.post(self._config.tts_url, json=payload) as resp:
                elapsed = time.monotonic() - job.submitted_at

                if resp.status == 200:
                    result = await resp.json()
                    job.status = "completed"
                    job.progress = 100.0
                    job.output_path = result.get("audio_path", output_path)
                    job.duration_sec = result.get("duration_sec", round(elapsed, 2))
                    job.error = ""
                    logger.info(
                        "TTS-%s job %s completed in %.1fs (duration=%.1fs, output=%s)",
                        self._config.track.value, job_id, elapsed,
                        job.duration_sec, job.output_path,
                    )
                else:
                    error_text = await resp.text()
                    job.status = "failed"
                    job.error = f"HTTP {resp.status}: {error_text[:300]}"
                    logger.error(
                        "TTS-%s job %s failed: HTTP %d — %s",
                        self._config.track.value, job_id, resp.status, error_text[:200],
                    )

        except asyncio.TimeoutError:
            job.status = "failed"
            job.error = f"Timeout after {self._config.timeout_sec}s"
            logger.error("TTS-%s job %s timed out", self._config.track.value, job_id)

        except aiohttp.ClientError as e:
            job.status = "failed"
            job.error = f"Connection error: {e}"
            logger.error("TTS-%s job %s connection error: %s", self._config.track.value, job_id, e)

        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            logger.error("TTS-%s job %s unexpected error: %s", self._config.track.value, job_id, e)

        return job_id

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

        artifacts = [{
            "url": f"file://{job.output_path}",
            "path": job.output_path,
            "type": "audio",
            "format": "wav",
            "track": job.track.value,
            "service": job.service_name,
            "duration_sec": job.duration_sec,
        }]
        return {"outputs": artifacts}

    async def cancel(self, engine_job_id: str) -> bool:
        # HTTP TTS is synchronous — cancel only works for queued jobs
        job = self._jobs.get(engine_job_id)
        if not job or job.status != "queued":
            return False
        job.status = "failed"
        job.error = "Cancelled"
        return True

    async def health(self) -> dict[str, Any]:
        now = time.monotonic()
        # Cache health check for 30s
        if self._healthy is not None and (now - self._last_health_check) < 30.0:
            status = EngineStatus.ONLINE if self._healthy else EngineStatus.OFFLINE
            return {
                "status": status.value,
                "available": self._healthy,
                "track": self._config.track.value,
                "service": self._config.name,
                "port": self._config.port,
                "cached": True,
            }

        try:
            session = await self._get_session()
            async with session.get(self._config.health_url, timeout=aiohttp.ClientTimeout(total=5.0)) as resp:
                if resp.status == 200:
                    body = await resp.json()
                    self._healthy = True
                    self._last_health_check = now
                    return {
                        "status": EngineStatus.ONLINE.value,
                        "available": True,
                        "track": self._config.track.value,
                        "service": self._config.name,
                        "port": self._config.port,
                        "vram_used_mb": body.get("vram_used_mb", 0),
                        "model_loaded": body.get("model_loaded", True),
                    }
        except Exception as e:
            logger.debug("TTS-%s health check failed: %s", self._config.track.value, e)

        self._healthy = False
        self._last_health_check = now
        return {
            "status": EngineStatus.OFFLINE.value,
            "available": False,
            "track": self._config.track.value,
            "service": self._config.name,
            "port": self._config.port,
            "error": "Service not reachable",
        }


# ─── Unified TTS Tracker ───

class TTSTracker(BaseEngine):
    """Unified TTS facade that auto-routes to the correct track engine.

    Routing logic:
    - Explicit track specified → use that engine
    - language='zh' → Track 1 (GPT-SoVITS)
    - language='en' → Track 2 (Chatterbox-Turbo)
    - language='auto' or mixed → detect, route accordingly
    - Fallback: if target engine is offline, suggest alternatives
    """

    def __init__(
        self,
        services: dict[TTSTrack, TTSServiceConfig] | None = None,
        output_root: str = OUTPUT_ROOT,
    ) -> None:
        self._output_root = output_root
        self._services = services or DEFAULT_SERVICES
        self._engines: dict[TTSTrack, HTTPTTSEngine] = {}
        self._build_engines()

    def _build_engines(self) -> None:
        for track, config in self._services.items():
            self._engines[track] = HTTPTTSEngine(config, self._output_root)

    @property
    def name(self) -> str:
        return "TTS Tracker (三轨调度)"

    @property
    def engine_id(self) -> str:
        return "tts-tracker"

    @property
    def capabilities(self) -> EngineCapabilities:
        all_models = []
        total_vram = 0
        for track, engine in self._engines.items():
            all_models.extend(engine.capabilities.models)
            total_vram += engine.capabilities.vram_total_mb
        return EngineCapabilities(
            supported_types=["tts", "tts_zh", "tts_en", "tts_bilingual"],
            max_duration_sec=120.0,
            vram_total_mb=total_vram,
            vram_available_mb=total_vram,
            models=all_models,
        )

    @property
    def engines(self) -> dict[TTSTrack, HTTPTTSEngine]:
        return self._engines

    async def start(self) -> None:
        for track, engine in self._engines.items():
            try:
                await engine.start()
            except Exception as e:
                logger.warning("TTS engine %s failed to start: %s", track.value, e)
        logger.info("TTS Tracker started with %d tracks", len(self._engines))

    async def stop(self) -> None:
        for engine in self._engines.values():
            await engine.stop()

    def _pick_track(self, workflow: dict[str, Any]) -> TTSTrack:
        """Determine which track to use based on workflow params."""
        # 1. Explicit track override
        explicit = workflow.get("track", "")
        if explicit in TTSTrack.__members__.values():
            return TTSTrack(explicit)

        # 2. Language-based routing
        language = workflow.get("language", "auto").lower()
        if language == "zh":
            return TTSTrack.ZH
        if language == "en":
            return TTSTrack.EN
        if language == "bilingual":
            return TTSTrack.BILINGUAL

        # 3. Auto-detect from text
        text = workflow.get("text", "")
        detected = detect_language(text)
        return TTSTrack.ZH if detected == "zh" else TTSTrack.EN

    async def submit(self, workflow: dict[str, Any], params: dict[str, Any] | None = None) -> str:
        """Submit TTS — auto-routes to the appropriate track."""
        track = self._pick_track(workflow)
        engine = self._engines[track]

        # Check health first
        health = await engine.health()
        if not health.get("available"):
            # Fallback: try bilingual track for any language
            bilingual_engine = self._engines.get(TTSTrack.BILINGUAL)
            if bilingual_engine and bilingual_engine is not engine:
                bh = await bilingual_engine.health()
                if bh.get("available"):
                    logger.warning(
                        "TTS track %s offline, falling back to bilingual for job",
                        track.value,
                    )
                    return await bilingual_engine.submit(workflow, params)
            raise RuntimeError(
                f"TTS track '{track.value}' ({engine.name}) is offline and no fallback available. "
                f"Start the service: {engine._config.base_url}"
            )

        # Inject track info
        workflow_copy = {**workflow, "track": track.value}
        return await engine.submit(workflow_copy, params)

    async def poll(self, engine_job_id: str) -> dict[str, Any]:
        """Poll across all track engines."""
        for engine in self._engines.values():
            result = await engine.poll(engine_job_id)
            if result["status"] != "failed" or "Unknown job" not in result.get("error", ""):
                result["track"] = engine.track.value
                return result
        return {"status": "failed", "progress": 0.0, "error": "Unknown job ID"}

    async def get_output(self, engine_job_id: str) -> dict[str, Any]:
        for engine in self._engines.values():
            result = await engine.get_output(engine_job_id)
            if result["outputs"]:
                return result
        return {"outputs": []}

    async def cancel(self, engine_job_id: str) -> bool:
        for engine in self._engines.values():
            if await engine.cancel(engine_job_id):
                return True
        return False

    async def health(self) -> dict[str, Any]:
        """Report health for all tracks."""
        tracks = {}
        all_online = True
        for track, engine in self._engines.items():
            h = await engine.health()
            tracks[track.value] = h
            if not h.get("available"):
                all_online = False

        online_count = sum(1 for t in tracks.values() if t.get("available"))
        return {
            "status": EngineStatus.ONLINE.value if all_online else EngineStatus.BUSY.value if online_count > 0 else EngineStatus.OFFLINE.value,
            "available": all_online,
            "tracks": tracks,
            "online_tracks": online_count,
            "total_tracks": len(self._engines),
            "routing_summary": {
                "zh": "GPT-SoVITS (:9880)",
                "en": "Chatterbox-Turbo (:9881)",
                "bilingual": "CosyVoice-3.0 (:9882)",
            },
        }

    # ─── Convenience: direct submit to specific track ───

    async def submit_zh(self, text: str, **kwargs) -> str:
        """Submit Chinese TTS directly to GPT-SoVITS."""
        return await self._engines[TTSTrack.ZH].submit({"text": text, **kwargs})

    async def submit_en(self, text: str, **kwargs) -> str:
        """Submit English TTS directly to Chatterbox-Turbo."""
        return await self._engines[TTSTrack.EN].submit({"text": text, **kwargs})

    async def submit_bilingual(self, text: str, **kwargs) -> str:
        """Submit bilingual TTS directly to CosyVoice."""
        return await self._engines[TTSTrack.BILINGUAL].submit({"text": text, **kwargs})
