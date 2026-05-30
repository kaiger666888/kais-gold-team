"""Base class for cloud API engines (Kling, Jimeng, Seedance, etc.)."""
from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Optional

import httpx

from src.v6.engines.base import BaseEngine, EngineCapabilities, EngineStatus

logger = logging.getLogger(__name__)


class CloudEngineError(Exception):
    """Cloud API error with provider context."""
    def __init__(self, provider: str, status: int, message: str):
        self.provider = provider
        self.status = status
        super().__init__(f"[{provider}] HTTP {status}: {message}")


class BaseCloudEngine(BaseEngine):
    """Base class for cloud AI generation engines.

    Subclasses must implement:
    - _submit_to_api()   — provider-specific API call to start generation
    - _poll_api()        — check if generation is done
    - _download_result() — download result artifact

    Configuration via environment variables:
    - {PROVIDER}_API_KEY  — API key (required for real calls)
    - {PROVIDER}_BASE_URL — API base URL (optional, uses default)
    """

    provider: str = "base"
    _supported_types: list[str] = []
    _default_models: list[str] = []
    _default_base_url: str = ""

    @property
    def name(self) -> str:
        return f"{self.provider.title()} Cloud Engine"

    @property
    def engine_id(self) -> str:
        return f"cloud-{self.provider}"

    def __init__(self) -> None:
        env_prefix = self.provider.upper()
        self.api_key = os.environ.get(f"{env_prefix}_API_KEY", "")
        self.base_url = os.environ.get(f"{env_prefix}_BASE_URL", self._default_base_url)
        self._jobs: dict[str, dict[str, Any]] = {}  # job_id → state
        self._client: Optional[httpx.AsyncClient] = None
        self._started = False

    @property
    def is_configured(self) -> bool:
        """Whether the engine has valid API credentials."""
        return bool(self.api_key)

    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            supported_types=self._supported_types,
            max_resolution=(2048, 2048),
            max_duration_sec=30.0,
            vram_total_mb=0,  # Cloud — no local VRAM
            vram_available_mb=0,
            models=self._default_models,
        )

    async def start(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(300.0, connect=30.0),
            headers=self._build_auth_headers(),
        )
        self._started = True
        configured = "✓ configured" if self.is_configured else "✗ no API key"
        logger.info("CloudEngine [%s] started — %s (base_url=%s)",
                     self.provider, configured, self.base_url)

    async def stop(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
        self._started = False

    def _build_auth_headers(self) -> dict[str, str]:
        """Override for provider-specific auth headers."""
        if self.api_key:
            return {"Authorization": f"Bearer {self.api_key}"}
        return {}

    async def submit(self, workflow: dict[str, Any], params: dict[str, Any] | None = None) -> str:
        """Submit a generation task to the cloud API."""
        if not self.is_configured:
            raise CloudEngineError(self.provider, 401, "API key not configured")

        params = params or {}
        task_type = params.get("type", "image_draw")
        prompt = workflow.get("prompt", workflow.get("6", {}).get("inputs", {}).get("text", ""))
        width = workflow.get("width", workflow.get("5", {}).get("inputs", {}).get("width", 1024))
        height = workflow.get("height", workflow.get("5", {}).get("inputs", {}).get("height", 1024))

        # Build provider-specific request
        request_body = self._build_request(task_type, prompt, width, height, workflow, params)
        job_id = f"{self.provider}_{int(time.time())}_{id(request_body) % 10000}"

        self._jobs[job_id] = {
            "status": "submitted",
            "progress": 0.0,
            "submitted_at": time.time(),
            "request": request_body,
            "task_type": task_type,
            "provider_job_id": None,
            "output": None,
        }

        # Submit asynchronously
        asyncio.create_task(self._run_job(job_id, request_body))

        return job_id

    async def _run_job(self, job_id: str, request_body: dict) -> None:
        """Background task: submit → poll → download."""
        try:
            self._jobs[job_id]["status"] = "submitting"
            provider_job_id = await self._submit_to_api(request_body)
            self._jobs[job_id]["provider_job_id"] = provider_job_id
            self._jobs[job_id]["status"] = "running"
            self._jobs[job_id]["progress"] = 10.0

            # Poll until done
            max_wait = 600  # 10 min max
            start = time.time()
            while time.time() - start < max_wait:
                await asyncio.sleep(3.0)
                result = await self._poll_api(provider_job_id)
                status = result.get("status", "running")
                progress = result.get("progress", 0.0)

                self._jobs[job_id]["progress"] = 10.0 + progress * 0.8

                if status == "completed":
                    output_url = result.get("output_url")
                    if output_url:
                        local_path = await self._download_result(output_url, job_id)
                        self._jobs[job_id]["output"] = local_path
                    else:
                        self._jobs[job_id]["output"] = output_url
                    self._jobs[job_id]["progress"] = 100.0
                    self._jobs[job_id]["status"] = "completed"
                    logger.info("CloudEngine [%s] job %s completed", self.provider, job_id)
                    return

                if status == "failed":
                    self._jobs[job_id]["status"] = "failed"
                    self._jobs[job_id]["error"] = result.get("error", "Unknown error")
                    logger.error("CloudEngine [%s] job %s failed: %s",
                                 self.provider, job_id, result.get("error"))
                    return

            # Timeout
            self._jobs[job_id]["status"] = "failed"
            self._jobs[job_id]["error"] = "Generation timed out"

        except Exception as e:
            self._jobs[job_id]["status"] = "failed"
            self._jobs[job_id]["error"] = str(e)
            logger.exception("CloudEngine [%s] job %s error", self.provider, job_id)

    async def poll(self, engine_job_id: str) -> dict[str, Any]:
        job = self._jobs.get(engine_job_id)
        if not job:
            return {"status": "failed", "progress": 0.0, "error": "Unknown job ID"}

        return {
            "status": job["status"],
            "progress": job.get("progress", 0.0),
            "error": job.get("error"),
        }

    async def get_output(self, engine_job_id: str) -> dict[str, Any]:
        job = self._jobs.get(engine_job_id)
        if not job:
            return {"outputs": []}

        output_path = job.get("output")
        task_type = job.get("task_type", "image_draw")

        if not output_path:
            return {"outputs": []}

        # Determine artifact type
        if task_type in ("video_final", "video_preview"):
            artifact_type = "video"
        elif task_type in ("tts", "music", "sfx"):
            artifact_type = "audio"
        else:
            artifact_type = "image"

        return {
            "outputs": [{
                "url": output_path,
                "type": artifact_type,
                "format": "mp4" if artifact_type == "video" else "png",
            }]
        }

    async def cancel(self, engine_job_id: str) -> bool:
        job = self._jobs.get(engine_job_id)
        if not job:
            return False
        if job["status"] in ("completed", "failed"):
            return False
        job["status"] = "failed"
        job["error"] = "Cancelled by user"
        return True

    async def health(self) -> dict[str, Any]:
        if not self.is_configured:
            return {
                "status": EngineStatus.OFFLINE,
                "available": False,
                "reason": f"No {self.provider.upper()}_API_KEY configured",
            }

        # Optionally ping the API
        try:
            return await self._health_check()
        except Exception as e:
            return {
                "status": EngineStatus.ERROR,
                "available": False,
                "reason": str(e),
            }

    # ─── Provider-specific methods (override in subclasses) ──────────

    def _build_request(self, task_type: str, prompt: str,
                       width: int, height: int,
                       workflow: dict, params: dict) -> dict:
        """Build provider-specific API request body."""
        return {"prompt": prompt, "width": width, "height": height}

    async def _submit_to_api(self, request_body: dict) -> str:
        """Submit to provider API, return provider job/task ID."""
        raise NotImplementedError

    async def _poll_api(self, provider_job_id: str) -> dict[str, Any]:
        """Poll provider for job status. Returns {status, progress, output_url?}."""
        raise NotImplementedError

    async def _download_result(self, output_url: str, job_id: str) -> str:
        """Download result to local file. Returns local path."""
        import os
        output_dir = os.environ.get("OUTPUT_DIR", "/mnt/agents/output")
        task_dir = os.path.join(output_dir, job_id)
        os.makedirs(task_dir, exist_ok=True)

        # Determine extension from URL
        ext = "mp4" if "video" in output_url.lower() else "png"
        local_path = os.path.join(task_dir, f"output.{ext}")

        async with httpx.AsyncClient(timeout=120.0) as dl_client:
            resp = await dl_client.get(output_url, follow_redirects=True)
            resp.raise_for_status()
            with open(local_path, "wb") as f:
                f.write(resp.content)

        return local_path

    async def _health_check(self) -> dict[str, Any]:
        """Default health check — just check if configured."""
        return {
            "status": EngineStatus.ONLINE if self.is_configured else EngineStatus.OFFLINE,
            "available": self.is_configured,
        }
