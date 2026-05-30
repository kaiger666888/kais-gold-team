"""Jimeng (即梦) Cloud Engine — via jimeng-free-api proxy."""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

import httpx

from src.v6.engines.base import EngineStatus
from src.v6.engines.cloud_base import BaseCloudEngine, CloudEngineError

logger = logging.getLogger(__name__)


class JimengEngine(BaseCloudEngine):
    """即梦 (Jimeng) cloud engine via jimeng-free-api proxy.

    Architecture:
        gold-team → jimeng-free-api (localhost:8000) → 即梦云端 API

    Environment:
        JIMENG_API_KEY    — session ID for jimeng-free-api auth
        JIMENG_BASE_URL   — proxy URL (default: http://172.17.0.1:8000)
    """

    provider = "jimeng"
    _supported_types = ["image_draw", "image_refine", "video_final"]
    _default_models = ["jimeng-5.0", "jimeng-video-3.5-pro", "jimeng-video-seedance-2.0-fast"]
    _default_base_url = "http://jimeng-free-api:5100"

    def __init__(self) -> None:
        super().__init__()
        # Jimeng uses session ID as auth, may come from different env var
        if not self.api_key:
            self.api_key = os.environ.get("JIMENG_SESSION_ID", "")

    def _build_request(self, task_type: str, prompt: str,
                       width: int, height: int,
                       workflow: dict, params: dict) -> dict:
        if task_type in ("video_final", "video_preview"):
            return {
                "model": params.get("model", "jimeng-video-3.5-pro"),
                "prompt": prompt,
                "ratio": self._aspect_ratio(width, height),
                "duration": params.get("duration", 5),
            }
        else:
            return {
                "model": params.get("model", "jimeng-5.0"),
                "prompt": prompt,
                "ratio": self._aspect_ratio(width, height),
                "resolution": params.get("resolution", "2k"),
            }

    @staticmethod
    def _aspect_ratio(w: int, h: int) -> str:
        """Map pixel dimensions to Jimeng ratio string."""
        if w == h:
            return "1:1"
        if w > h:
            ratio = w / h
            if abs(ratio - 16/9) < 0.1:
                return "16:9"
            if abs(ratio - 3/2) < 0.1:
                return "3:2"
            if abs(ratio - 4/3) < 0.1:
                return "4:3"
            return "16:9"
        else:
            ratio = h / w
            if abs(ratio - 16/9) < 0.1:
                return "9:16"
            if abs(ratio - 3/2) < 0.1:
                return "2:3"
            return "9:16"

    async def _submit_to_api(self, request_body: dict) -> str:
        """Submit to jimeng-free-api. The proxy handles async polling internally."""
        is_video = "duration" in request_body

        if is_video:
            endpoint = f"{self.base_url}/v1/videos/generations"
        else:
            endpoint = f"{self.base_url}/v1/images/generations"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=30.0)) as client:
            resp = await client.post(endpoint, json=request_body, headers=headers)

            if resp.status_code == 429:
                raise CloudEngineError("jimeng", 429, "Rate limited — QPS=1, retry after cooldown")
            if resp.status_code == 401:
                raise CloudEngineError("jimeng", 401, "Session expired or invalid")

            if resp.status_code != 200:
                raise CloudEngineError("jimeng", resp.status_code, resp.text[:500])

            data = resp.json()

        # jimeng-free-api returns results synchronously (it polls internally)
        # For images: data.data[0].url
        # For videos: data.data[0].url
        items = data.get("data", [])
        if items and items[0].get("url"):
            # Already completed — store result directly
            return f"completed:{items[0]['url']}"

        # Some responses might have a task_id for async
        task_id = data.get("task_id") or data.get("id")
        if task_id:
            return str(task_id)

        raise CloudEngineError("jimeng", 500, f"No result or task_id in response: {data}")

    async def _poll_api(self, provider_job_id: str) -> dict[str, Any]:
        """Poll jimeng-free-api for task status."""
        # If already completed during submit
        if provider_job_id.startswith("completed:"):
            url = provider_job_id[len("completed:"):]
            return {"status": "completed", "progress": 100.0, "output_url": url}

        # Poll async task
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                f"{self.base_url}/v1/tasks/{provider_job_id}",
                headers=headers,
            )
            if resp.status_code != 200:
                return {"status": "running", "progress": 50.0}

            data = resp.json()

        status = data.get("status", "processing")
        if status in ("succeeded", "completed", "done"):
            url = data.get("output", {}).get("url") or data.get("data", [{}])[0].get("url", "")
            return {"status": "completed", "progress": 100.0, "output_url": url}
        if status in ("failed", "error"):
            return {"status": "failed", "error": data.get("error", "Unknown")}
        # Still processing
        progress = data.get("progress", 50.0)
        return {"status": "running", "progress": progress}

    async def _health_check(self) -> dict[str, Any]:
        """Ping jimeng-free-api proxy."""
        if not self.is_configured:
            return {
                "status": EngineStatus.OFFLINE,
                "available": False,
                "reason": "No JIMENG_API_KEY or JIMENG_SESSION_ID configured",
            }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{self.base_url}/ping")
                if resp.status_code == 200:
                    return {
                        "status": EngineStatus.ONLINE,
                        "available": True,
                        "proxy": self.base_url,
                    }
        except Exception as e:
            return {
                "status": EngineStatus.OFFLINE,
                "available": False,
                "reason": f"jimeng-free-api unreachable: {e}",
            }

        return {
            "status": EngineStatus.OFFLINE,
            "available": False,
            "reason": f"jimeng-free-api returned non-200",
        }
