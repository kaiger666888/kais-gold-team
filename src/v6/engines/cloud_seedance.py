"""Seedance Cloud Engine — via jimeng-free-api proxy (Seedance models)."""
from __future__ import annotations

import logging
import os
from typing import Any

from src.v6.engines.cloud_base import BaseCloudEngine, CloudEngineError
from src.v6.engines.cloud_jimeng import JimengEngine

logger = logging.getLogger(__name__)


class SeedanceEngine(JimengEngine):
    """Seedance video engine — routed through jimeng-free-api proxy.

    Seedance is a ByteDance video model available through the 即梦 API.
    Uses the same proxy infrastructure as Jimeng but with Seedance-specific models.

    Environment (same as Jimeng):
        JIMENG_API_KEY / JIMENG_SESSION_ID
        JIMENG_BASE_URL — proxy URL (default: http://172.17.0.1:8000)
    """

    provider = "seedance"
    _supported_types = ["video_final", "video_preview"]
    _default_models = ["jimeng-video-seedance-2.0-fast", "jimeng-video-seedance-2.0-pro"]
    _default_base_url = "http://jimeng-free-api:5100"

    def _build_request(self, task_type: str, prompt: str,
                       width: int, height: int,
                       workflow: dict, params: dict) -> dict:
        model = params.get("model", "jimeng-video-seedance-2.0-fast")
        return {
            "model": model,
            "prompt": prompt,
            "ratio": self._aspect_ratio(width, height),
            "duration": params.get("duration", 4),
            "file_paths": params.get("file_paths", []),
        }
