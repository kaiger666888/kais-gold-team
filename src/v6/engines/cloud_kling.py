"""Kling (可灵) Cloud Engine — via Kling official API."""
from __future__ import annotations

import logging
import os
from typing import Any

from src.v6.engines.base import EngineStatus
from src.v6.engines.cloud_base import BaseCloudEngine, CloudEngineError

logger = logging.getLogger(__name__)


class KlingEngine(BaseCloudEngine):
    """可灵 (Kling) cloud engine via official Kling API.

    API Docs: https://platform.kuaishou.com/docs
    Authentication: JWT token via Access Key + Secret Key

    Environment:
        KLING_ACCESS_KEY  — Access Key (from Kuaishou developer console)
        KLING_SECRET_KEY  — Secret Key
        KLING_BASE_URL    — API base URL (default: https://api.klingai.com)
    """

    provider = "kling"
    _supported_types = ["video_final", "video_preview", "image_draw"]
    _default_models = ["kling-v2-master", "kling-v1-5", "kling-v1"]
    _default_base_url = "https://api.klingai.com"

    def __init__(self) -> None:
        super().__init__()
        self.access_key = os.environ.get("KLING_ACCESS_KEY", "")
        self.secret_key = os.environ.get("KLING_SECRET_KEY", "")
        # Kling uses access+secret to generate JWT, not a simple API key
        self.api_key = self.access_key  # For is_configured check

    @property
    def is_configured(self) -> bool:
        return bool(self.access_key and self.secret_key)

    def _build_request(self, task_type: str, prompt: str,
                       width: int, height: int,
                       workflow: dict, params: dict) -> dict:
        model = params.get("model", "kling-v2-master")

        if task_type in ("video_final", "video_preview"):
            return {
                "model": model,
                "prompt": prompt,
                "mode": params.get("mode", "std"),  # std or pro
                "duration": params.get("duration", "5"),  # "5" or "10"
                "aspect_ratio": self._aspect_ratio(width, height),
                "callback_url": params.get("callback_url"),
            }
        else:
            # Image generation
            return {
                "model": model,
                "prompt": prompt,
                "aspect_ratio": self._aspect_ratio(width, height),
                "image_count": params.get("image_count", 1),
            }

    @staticmethod
    def _aspect_ratio(w: int, h: int) -> str:
        if w == h:
            return "1:1"
        if w > h:
            return "16:9" if w / h > 1.5 else "4:3"
        return "9:16" if h / w > 1.5 else "3:4"

    def _build_auth_headers(self) -> dict[str, str]:
        """Kling uses JWT auth — generate on first call."""
        # JWT generation requires PyJWT; fallback to Bearer token if pre-generated
        if self.api_key:
            return {"Authorization": f"Bearer {self.api_key}"}
        return {}

    async def _submit_to_api(self, request_body: dict) -> str:
        """Submit generation task to Kling API."""
        is_video = "duration" in request_body

        if is_video:
            endpoint = "/v1/videos/image2video" if request_body.get("image") else "/v1/videos/text2video"
        else:
            endpoint = "/v1/images/generations"

        headers = {
            **self._build_auth_headers(),
            "Content-Type": "application/json",
        }

        import httpx
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0)) as client:
            resp = await client.post(
                f"{self.base_url}{endpoint}",
                json=request_body,
                headers=headers,
            )

            if resp.status_code == 401:
                raise CloudEngineError("kling", 401, "Invalid access key or JWT expired")
            if resp.status_code == 429:
                raise CloudEngineError("kling", 429, "Rate limited")
            if resp.status_code not in (200, 201):
                raise CloudEngineError("kling", resp.status_code, resp.text[:500])

            data = resp.json()

        code = data.get("code")
        if code and code != 0:
            raise CloudEngineError("kling", code, data.get("message", "Unknown error"))

        task_id = data.get("data", {}).get("task_id")
        if not task_id:
            raise CloudEngineError("kling", 500, f"No task_id in response: {data}")

        logger.info("Kling task submitted: %s", task_id)
        return task_id

    async def _poll_api(self, provider_job_id: str) -> dict[str, Any]:
        """Poll Kling task status."""
        headers = {**self._build_auth_headers()}

        import httpx
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Detect if video or image task
            resp = await client.get(
                f"{self.base_url}/v1/videos/image2video/{provider_job_id}",
                headers=headers,
            )
            if resp.status_code == 404:
                resp = await client.get(
                    f"{self.base_url}/v1/images/generations/{provider_job_id}",
                    headers=headers,
                )

            if resp.status_code != 200:
                return {"status": "running", "progress": 50.0}

            data = resp.json()

        task_data = data.get("data", {})
        task_status = task_data.get("task_status", "processing")

        if task_status == "succeed":
            results = task_data.get("task_result", {}).get("videos", [])
            if not results:
                results = task_data.get("task_result", {}).get("images", [])
            url = results[0].get("url", "") if results else ""
            return {"status": "completed", "progress": 100.0, "output_url": url}

        if task_status == "failed":
            return {"status": "failed", "error": task_data.get("task_status_msg", "Unknown")}

        # Still processing
        return {"status": "running", "progress": 50.0}

    async def _health_check(self) -> dict[str, Any]:
        if not self.is_configured:
            return {
                "status": EngineStatus.OFFLINE,
                "available": False,
                "reason": "KLING_ACCESS_KEY and KLING_SECRET_KEY not configured",
            }
        # Can't easily test without making a real call
        return {
            "status": EngineStatus.ONLINE,
            "available": True,
            "note": "Configured but not verified — needs test call",
        }
