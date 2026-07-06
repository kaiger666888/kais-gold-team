"""ComfyUI API client engine - talks to a local ComfyUI instance via HTTP."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from typing import Any, Optional

import httpx


def _extract_comfyui_error(messages) -> str:
    if isinstance(messages, str):
        return messages
    if isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, (list, tuple)) and len(msg) >= 3:
                if msg[0] == "execution_error":
                    exc = msg[2]
                    if isinstance(exc, dict):
                        parts = [exc.get("node_type", ""), exc.get("exception_message", "unknown error")]
                        return " | ".join(p for p in parts if p)
                    return str(exc)
    return json.dumps(messages, ensure_ascii=False)[:500]

from src.v6.engines.base import BackendType, BaseEngine, EngineCapabilities, EngineStatus

logger = logging.getLogger(__name__)

# Default ComfyUI API address
DEFAULT_COMFYUI_HOST = "127.0.0.1"
DEFAULT_COMFYUI_PORT = 8188
POLL_INTERVAL_SEC = 1.0
POLL_TIMEOUT_SEC = 600.0  # 10 min max wait


# Capability profiles keyed by engine_id
_ENGINE_PROFILES: dict[str, dict[str, Any]] = {
    "comfyui-primary": {
        "name": "ComfyUI Primary (RTX 3090)",
        "supported_types": [
            "video_final", "video_preview",
            "image_draw", "image_refine",
            "upscale", "face_restore", "image_to_3d",
            "image_pulid", "image_draw_ipadapter", "controlnet_depth", "wan_i2v",
        ],
        "vram_total_mb": 24576,
        "vram_available_mb": 24576,
        "models": ["flux-dev-fp8", "flux-dev-ipa", "wan2.2-14b", "ltx-2.3-fp8", "real-esrgan", "facefusion"],
    },
    "comfyui-auxiliary": {
        "name": "ComfyUI Auxiliary (RTX 3060 Ti)",
        "supported_types": ["upscale", "face_restore", "image_refine", "image_draw"],
        "vram_total_mb": 5500,
        "vram_available_mb": 5500,
        "models": ["real-esrgan", "facefusion"],
    },
}

_FALLBACK_PROFILE = {
    "name": "ComfyUI Local",
    "supported_types": [
        "video_final", "video_preview",
        "image_draw", "image_refine",
        "upscale", "face_restore", "image_to_3d",
            "image_pulid", "controlnet_depth", "wan_i2v",
    ],
    "vram_total_mb": 24576,
    "vram_available_mb": 24576,
    "models": ["flux-dev-fp8", "flux-dev-ipa", "wan2.2-14b", "ltx-2.3-fp8", "real-esrgan", "facefusion"],
}


class ComfyUIEngine(BaseEngine):
    """ComfyUI execution engine.

    Communicates with a running ComfyUI instance via its REST API:
    - POST /prompt        → submit workflow
    - GET  /history/{id}  → poll status
    - POST /interrupt     → cancel current
    - GET  /system_stats  → health check

    Supports multiple instances via ``engine_id``:
    - ``comfyui-primary`` — RTX 3090, all task types
    - ``comfyui-auxiliary`` — RTX 3060 Ti, lightweight tasks only
    """

    def __init__(
        self,
        host: str = DEFAULT_COMFYUI_HOST,
        port: int = DEFAULT_COMFYUI_PORT,
        engine_id: Optional[str] = None,
        client_id: Optional[str] = None,
        poll_interval: float = POLL_INTERVAL_SEC,
        poll_timeout: float = POLL_TIMEOUT_SEC,
    ) -> None:
        self._host = host
        self._port = port
        self._engine_id = engine_id or "comfyui-local"
        self._client_id = client_id or str(uuid.uuid4())
        self._poll_interval = poll_interval
        self._poll_timeout = poll_timeout
        self._base_url = f"http://{host}:{port}"
        self._http: Optional[httpx.AsyncClient] = None
        self._profile = _ENGINE_PROFILES.get(self._engine_id, _FALLBACK_PROFILE)

    @property
    def name(self) -> str:
        return self._profile["name"]

    @property
    def engine_id(self) -> str:
        return self._engine_id

    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            supported_types=self._profile["supported_types"],
            max_resolution=(2048, 2048),
            max_duration_sec=30.0,
            vram_total_mb=self._profile["vram_total_mb"],
            vram_available_mb=self._profile["vram_available_mb"],
            models=self._profile["models"],
        )

    @property
    def backend_type(self) -> BackendType:
        return BackendType.COMFYUI

    async def start(self) -> None:
        self._http = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(30.0, connect=5.0),
        )
        logger.info("ComfyUI engine client started → %s", self._base_url)

    async def stop(self) -> None:
        if self._http:
            await self._http.aclose()
            self._http = None

    # ─── Core API ───

    async def submit(self, workflow: dict[str, Any], params: dict[str, Any] | None = None) -> str:
        """Submit a ComfyUI API-format workflow.

        Args:
            workflow: ComfyUI API prompt JSON (``{prompt: {...}}``).
            params: Optional param overrides (currently unused, reserved).

        Returns:
            ComfyUI prompt_id for tracking.
        """
        assert self._http is not None, "Engine not started"

        payload = {
            "prompt": workflow,
            "client_id": self._client_id,
        }

        resp = await self._http.post("/prompt", json=payload)
        resp.raise_for_status()
        data = resp.json()

        prompt_id: str = data["prompt_id"]
        logger.info("ComfyUI workflow submitted: %s", prompt_id)
        return prompt_id

    async def poll(self, engine_job_id: str) -> dict[str, Any]:
        """Poll ComfyUI execution status via /history endpoint.

        Returns:
            ``{"status": "queued"|"running"|"completed"|"failed", "progress": float}``
        """
        assert self._http is not None

        resp = await self._http.get(f"/history/{engine_job_id}")
        if resp.status_code == 404:
            return {"status": "queued", "progress": 0.0}

        resp.raise_for_status()
        history = resp.json()

        if engine_job_id not in history:
            return {"status": "queued", "progress": 0.0}

        item = history[engine_job_id]

        # Check for error
        if "status" in item:
            status_data = item["status"]
            if status_data.get("status_str") == "error":
                return {
                    "status": "failed",
                    "progress": 0.0,
                    "error": _extract_comfyui_error(status_data.get("messages", [])),
                }

        # Check outputs → completed
        outputs = item.get("outputs", {})
        if outputs:
            return {"status": "completed", "progress": 100.0, "outputs": outputs}

        # Still running — try to estimate progress
        return {"status": "running", "progress": 50.0}

    async def get_output(self, engine_job_id: str) -> dict[str, Any]:
        """Get output artifacts from ComfyUI history."""
        assert self._http is not None

        resp = await self._http.get(f"/history/{engine_job_id}")
        resp.raise_for_status()
        history = resp.json()

        item = history.get(engine_job_id, {})
        outputs = item.get("outputs", {})

        artifacts: list[dict[str, Any]] = []
        for node_id, node_output in outputs.items():
            # Images
            for img in node_output.get("images", []):
                filename = img.get("filename", "")
                subfolder = img.get("subfolder", "")
                img_type = img.get("type", "output")
                url = (
                    f"{self._base_url}/view?"
                    f"filename={filename}&subfolder={subfolder}&type={img_type}"
                )
                artifacts.append({"url": url, "type": "image", "format": "png", "node": node_id})

            # Videos / animated
            for vid in node_output.get("videos", []):
                filename = vid.get("filename", "")
                subfolder = vid.get("subfolder", "")
                vid_type = vid.get("type", "output")
                url = (
                    f"{self._base_url}/view?"
                    f"filename={filename}&subfolder={subfolder}&type={vid_type}"
                )
                fmt = "mp4" if filename.endswith(".mp4") else "gif"
                artifacts.append({"url": url, "type": "video", "format": fmt, "node": node_id})

            # Audio
            for aud in node_output.get("audio", []):
                filename = aud.get("filename", "")
                url = f"{self._base_url}/view?filename={filename}"
                artifacts.append({"url": url, "type": "audio", "format": "wav", "node": node_id})

        return {"outputs": artifacts}

    async def cancel(self, engine_job_id: str) -> bool:
        """Interrupt ComfyUI execution.

        Note: ComfyUI's /interrupt cancels the *current* running prompt.
        """
        assert self._http is not None
        try:
            resp = await self._http.post("/interrupt")
            resp.raise_for_status()
            logger.info("ComfyUI interrupt sent for %s", engine_job_id)
            return True
        except Exception as e:
            logger.error("ComfyUI cancel failed: %s", e)
            return False

    async def health(self) -> dict[str, Any]:
        """Check ComfyUI availability via /system_stats."""
        if not self._http:
            return {"status": EngineStatus.OFFLINE.value, "available": False}

        try:
            resp = await self._http.get("/system_stats", timeout=5.0)
            resp.raise_for_status()
            data = resp.json()

            devices = data.get("devices", [])
            vram_total = 0
            vram_free = 0
            for d in devices:
                vram_total += d.get("vram_total", 0)
                vram_free += d.get("vram_free", 0)

            return {
                "status": EngineStatus.ONLINE.value,
                "available": True,
                "vram_total_mb": vram_total // (1024 * 1024),
                "vram_available_mb": vram_free // (1024 * 1024),
                "devices": devices,
            }
        except Exception as e:
            logger.warning("ComfyUI health check failed: %s", e)
            return {
                "status": EngineStatus.OFFLINE.value,
                "available": False,
                "error": str(e),
            }

    # ─── Convenience: submit and wait ───

    async def submit_and_wait(
        self,
        workflow: dict[str, Any],
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Submit workflow and block until completion or timeout."""
        prompt_id = await self.submit(workflow, params)

        elapsed = 0.0
        while elapsed < self._poll_timeout:
            result = await self.poll(prompt_id)
            status = result["status"]

            if status == "completed":
                output = await self.get_output(prompt_id)
                return {"status": "completed", "engine_job_id": prompt_id, **output}
            if status == "failed":
                return {"status": "failed", "engine_job_id": prompt_id, "error": result.get("error")}

            await asyncio.sleep(self._poll_interval)
            elapsed += self._poll_interval

        # Timeout
        await self.cancel(prompt_id)
        return {"status": "failed", "engine_job_id": prompt_id, "error": "Execution timed out"}
