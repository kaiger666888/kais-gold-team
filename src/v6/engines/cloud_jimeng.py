"""Jimeng (即梦) Cloud Engine — via dreamina CLI (official OAuth-based tool).

Replaces the deprecated jimeng-free-api HTTP proxy with direct subprocess
calls to the dreamina Go binary. Only image generation (text2image +
image2image) is enabled; video and other task types are explicitly rejected.

Enforced constraints (2026-08-19 Kai directive — whitelist replaces the old
single-model "5.0lite" lock):
  - text2image: model_version ∈ {"5.0" (default), "5.0lite"};
    anything else (incl. the retired 5.0Pro) is rejected with HTTP 400
  - image2image: model_version FORCED to "4.6" (Kai 08-06 rule —
    5.x i2i deadlocks server-side)
  - resolution_type locked to "2k" (4k is forbidden)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

import httpx

from src.v6.engines.base import EngineStatus
from src.v6.engines.cloud_base import BaseCloudEngine, CloudEngineError

logger = logging.getLogger(__name__)

# ── Hard constraints ─────────────────────────────────────────────────────────

# Model whitelist (2026-08-19 Kai directive: t2i defaults to 5.0, 5.0Pro
# retired pipeline-wide; i2i locked to 4.6 per Kai 08-06 rule).
_T2I_DEFAULT_MODEL = "5.0"
_T2I_ALLOWED_MODELS = {"5.0", "5.0lite"}
_I2I_MODEL = "4.6"
_LOCKED_RESOLUTION = "2k"          # Only allowed resolution (4k FORBIDDEN)
_VALID_RATIOS = {"1:1", "16:9", "9:16", "3:2", "2:3", "4:3", "3:4", "21:9"}
_DREAMINA_CLI_PATHS = [
    Path(os.environ.get("DREAMINA_CLI", "/usr/local/bin/dreamina")),
    Path.home() / ".local" / "bin" / "dreamina",
    Path("/home/kai/.local/bin/dreamina"),
]


class JimengEngine(BaseCloudEngine):
    """即梦 (Jimeng) cloud engine via dreamina CLI subprocess.

    Architecture:
        gold-team container → dreamina CLI (host-mounted binary)
                           → 即梦云端 (OAuth-token based)

    The dreamina binary authenticates via OAuth Device Flow (state stored in
    ~/.dreamina/).  No API key or session ID env var is needed — the CLI
    manages its own auth.

    Only image generation is enabled.  Video / TTS / other task types raise
    a clear error directing the caller to use a different engine.

    Environment:
        DREAMINA_CLI   — override path to dreamina binary
                         (default: auto-discover)
    """

    provider = "jimeng"
    _supported_types = ["image_draw", "image_refine"]
    _default_models = ["5.0", "5.0lite"]
    _default_base_url = ""  # Not used — CLI-based, not HTTP

    def __init__(self) -> None:
        # Skip parent __init__ partially — we don't need api_key/base_url
        # because dreamina CLI handles its own OAuth auth.
        self.api_key = ""  # CLI auth is implicit; always "configured" if binary exists
        self.base_url = ""
        self._jobs: dict[str, dict[str, Any]] = {}
        self._client: Optional[httpx.AsyncClient] = None
        self._started = False
        self._cli_path: Optional[Path] = self._find_cli()

    # ── CLI discovery ────────────────────────────────────────────────────────

    @staticmethod
    def _find_cli() -> Optional[Path]:
        """Locate the dreamina binary."""
        for candidate in _DREAMINA_CLI_PATHS:
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate
        # Last resort: search PATH
        found = shutil.which("dreamina")
        if found:
            return Path(found)
        return None

    # ── Overrides for BaseCloudEngine ────────────────────────────────────────

    @property
    def name(self) -> str:
        return "Dreamina CLI Engine"

    @property
    def engine_id(self) -> str:
        return "cloud-jimeng"

    @property
    def is_configured(self) -> bool:
        """Configured = dreamina binary exists on the filesystem."""
        return self._cli_path is not None

    def _build_auth_headers(self) -> dict[str, str]:
        return {}  # CLI handles auth internally via OAuth tokens

    # ── Request building ────────────────────────────────────────────────────

    def _build_request(self, task_type: str, prompt: str,
                       width: int, height: int,
                       workflow: dict, params: dict) -> dict:
        """Build the internal request dict consumed by _submit_to_api."""
        # Reject non-image task types
        if task_type not in ("image_draw", "image_refine"):
            raise CloudEngineError(
                "jimeng", 400,
                f"Task type '{task_type}' is not supported. "
                f"This engine only supports image generation "
                f"(image_draw / image_refine). "
                f"Use a different engine for video/audio.",
            )

        # Model whitelist — see module docstring (2026-08-19 Kai directive).
        requested_model = params.get("model_version") or workflow.get("model_version")
        if task_type == "image_refine":
            if requested_model and requested_model != _I2I_MODEL:
                logger.warning(
                    "jimeng i2i model_version=%r requested but i2i is locked to %r "
                    "(Kai 08-06 rule: 5.x i2i deadlocks); forcing %s",
                    requested_model, _I2I_MODEL, _I2I_MODEL,
                )
            model = _I2I_MODEL
        else:
            model = requested_model or _T2I_DEFAULT_MODEL
            if model not in _T2I_ALLOWED_MODELS:
                raise CloudEngineError(
                    "jimeng", 400,
                    f"model_version '{model}' is not allowed for text2image. "
                    f"Allowed: {sorted(_T2I_ALLOWED_MODELS)} "
                    f"(5.0Pro was retired pipeline-wide on 2026-08-19).",
                )

        resolution = _LOCKED_RESOLUTION  # 4k stays forbidden

        # Explicit ratio wins (callers know their form factor — the executor's
        # auto-built ComfyUI workflow doesn't carry top-level width/height);
        # otherwise derive from pixel dimensions.
        ratio = params.get("ratio") or workflow.get("ratio")
        if ratio not in _VALID_RATIOS:
            ratio = self._aspect_ratio(width, height)

        # Params-level prompt is authoritative for direct HTTP callers whose
        # executor-built workflow nests the prompt inside ComfyUI nodes.
        prompt = params.get("prompt") or prompt

        request: dict[str, Any] = {
            "task_type": task_type,
            "prompt": prompt,
            "ratio": ratio,
            "model_version": model,
            "resolution_type": resolution,
        }

        # image2image: extract reference images from params/workflow
        if task_type == "image_refine":
            ref_images = params.get("ref_images") or workflow.get("ref_images", [])
            if ref_images:
                request["ref_images"] = ref_images

        return request

    @staticmethod
    def _aspect_ratio(w: int, h: int) -> str:
        """Map pixel dimensions to dreamina ratio string."""
        if w == h:
            return "1:1"
        if w > h:
            ratio = w / h
            if abs(ratio - 21 / 9) < 0.15:
                return "21:9"
            if abs(ratio - 16 / 9) < 0.1:
                return "16:9"
            if abs(ratio - 3 / 2) < 0.1:
                return "3:2"
            if abs(ratio - 4 / 3) < 0.1:
                return "4:3"
            return "16:9"
        else:
            ratio = h / w
            if abs(ratio - 16 / 9) < 0.1:
                return "9:16"
            if abs(ratio - 3 / 2) < 0.1:
                return "2:3"
            if abs(ratio - 4 / 3) < 0.1:
                return "3:4"
            return "9:16"

    # ── CLI invocation ──────────────────────────────────────────────────────

    async def _run_cli(self, args: list[str], timeout: int = 30) -> dict:
        """Execute dreamina CLI and return parsed JSON output."""
        if not self._cli_path:
            raise CloudEngineError("jimeng", 500, "dreamina CLI binary not found")

        cmd = [str(self._cli_path)] + args
        logger.debug("dreamina CLI: %s", " ".join(cmd[:3]) + "...")

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=timeout,
            ),
        )

        if result.returncode != 0:
            stderr_snippet = (result.stderr or "")[:500]
            raise CloudEngineError(
                "jimeng", 500,
                f"dreamina CLI exited {result.returncode}: {stderr_snippet}",
            )

        try:
            data = json.loads(result.stdout or "")
        except (json.JSONDecodeError, TypeError) as exc:
            raise CloudEngineError(
                "jimeng", 500,
                f"dreamina CLI returned invalid JSON: {exc}",
            ) from exc

        return data

    async def _submit_to_api(self, request_body: dict) -> str:
        """Submit to dreamina CLI. Returns provider job tracking info.

        For text2image: dreamina text2image --poll=0 → submit_id
        For image2image: dreamina image2image --poll=0 → submit_id

        Returns a string that encodes both submit_id and task_type so
        _poll_api knows how to track it.
        """
        task_type = request_body["task_type"]
        prompt = request_body["prompt"]
        ratio = request_body["ratio"]
        model_version = request_body["model_version"]
        resolution = request_body["resolution_type"]

        if task_type == "image_refine" and request_body.get("ref_images"):
            # image2image mode
            ref_paths = request_body["ref_images"]
            if isinstance(ref_paths, str):
                ref_paths = [ref_paths]
            # dreamina expects comma-separated --images
            images_arg = ",".join(ref_paths[:10])  # max 10 images

            data = await self._run_cli([
                "image2image",
                f"--images={images_arg}",
                f"--prompt={prompt}",
                f"--ratio={ratio}",
                f"--model_version={model_version}",
                f"--resolution_type={resolution}",
                "--poll=0",
            ], timeout=30)
        else:
            # text2image mode
            data = await self._run_cli([
                "text2image",
                f"--prompt={prompt}",
                f"--ratio={ratio}",
                f"--model_version={model_version}",
                f"--resolution_type={resolution}",
                "--poll=0",
            ], timeout=15)

        submit_id = data.get("submit_id", "")
        if not submit_id:
            raise CloudEngineError(
                "jimeng", 500,
                f"dreamina CLI returned empty submit_id: {data}",
            )

        return submit_id

    async def _poll_api(self, provider_job_id: str) -> dict[str, Any]:
        """Poll dreamina query_result for task status.

        NOTE: provider_job_id must be the FULL UUID.
        NOTE: response uses 'gen_status' field (NOT 'status').
        """
        data = await self._run_cli([
            "query_result",
            f"--submit_id={provider_job_id}",
        ], timeout=15)

        status = data.get("gen_status", "unknown")

        if status == "success":
            # Extract image URL from result_json.images[0].image_url
            images = data.get("result_json", {}).get("images", [])
            if images:
                url = images[0].get("image_url", "")
                if url:
                    return {"status": "completed", "progress": 100.0, "output_url": url}
            # Fallback: check for direct url fields
            url = data.get("image_url", "") or data.get("url", "")
            if url:
                return {"status": "completed", "progress": 100.0, "output_url": url}

            return {
                "status": "failed",
                "error": "gen_status=success but no image_url found",
            }

        if status in ("failed", "error", "fail"):
            fail_reason = data.get("fail_reason", "Generation failed")
            return {"status": "failed", "error": fail_reason}

        if status in ("querying", "processing", "queued"):
            # Extract progress if available
            queue = data.get("queue_info", {})
            queue_status = queue.get("queue_status", "")
            if queue_status == "Generating":
                return {"status": "running", "progress": 50.0}
            if queue_status == "Queued":
                return {"status": "running", "progress": 20.0}
            return {"status": "running", "progress": 30.0}

        # Unknown status — treat as running
        return {"status": "running", "progress": 40.0}

    async def _health_check(self) -> dict[str, Any]:
        """Check dreamina CLI availability + login status."""
        if not self._cli_path:
            searched = ", ".join(str(p) for p in _DREAMINA_CLI_PATHS)
            return {
                "status": EngineStatus.OFFLINE,
                "available": False,
                "reason": f"dreamina CLI not found (searched: {searched})",
            }

        try:
            data = await self._run_cli(["user_credit"], timeout=10)
            credit = data.get("total_credit", 0)
            if credit > 0:
                return {
                    "status": EngineStatus.ONLINE,
                    "available": True,
                    "cli": str(self._cli_path),
                    "credit": credit,
                    "vip": data.get("vip_level", "unknown"),
                    "model": _T2I_DEFAULT_MODEL,
                    "models": {
                        "t2i": sorted(_T2I_ALLOWED_MODELS),
                        "t2i_default": _T2I_DEFAULT_MODEL,
                        "i2i": _I2I_MODEL,
                    },
                    "resolution": _LOCKED_RESOLUTION,
                }
            return {
                "status": EngineStatus.ERROR,
                "available": False,
                "reason": f"Credit balance is {credit} (may be depleted)",
            }
        except Exception as e:
            return {
                "status": EngineStatus.OFFLINE,
                "available": False,
                "reason": f"dreamina CLI unreachable: {e}",
            }

    # ── Override _download_result: NOT aria2c, use httpx single-connection ──

    async def _download_result(self, output_url: str, job_id: str) -> str:
        """Download image via httpx (single connection, NOT aria2c).

        byteimg.com CDN breaks with multi-connection downloads.
        """
        output_dir = os.environ.get("OUTPUT_DIR", "/mnt/agents/output")
        task_dir = os.path.join(output_dir, job_id)
        os.makedirs(task_dir, exist_ok=True)
        local_path = os.path.join(task_dir, "output.png")

        # Validate URL scheme (basic SSRF mitigation)
        if not (output_url.startswith("http://") or output_url.startswith("https://")):
            raise CloudEngineError(
                "jimeng", 400,
                f"Image URL is not HTTP(S): {output_url[:80]}",
            )

        async with httpx.AsyncClient(timeout=60.0) as dl_client:
            resp = await dl_client.get(output_url)
            resp.raise_for_status()
            with open(local_path, "wb") as f:
                f.write(resp.content)

        logger.info("dreamina downloaded %s → %s (%d bytes)",
                     output_url[:60], local_path,
                     os.path.getsize(local_path))
        return local_path
