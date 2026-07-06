"""Docker API engine — config-driven base class for synchronous HTTP engines.

Covers 23 of 25 local engines. Each engine runs as a Docker container with an
HTTP API. The lifecycle is: start container → health check → POST task → get
response (sync, no polling) → stop container.
"""
from __future__ import annotations

import asyncio
import base64
import logging
from pathlib import Path
from typing import Any

import httpx

from src.v6.config.engine_schema import EngineConfig
from src.v6.docker.container_manager import ContainerManager
from src.v6.engines.base import BackendType, BaseEngine, EngineCapabilities, EngineStatus

logger = logging.getLogger(__name__)


class DockerAPIEngine(BaseEngine):
    """Config-driven engine for Docker containers with synchronous HTTP APIs."""

    def __init__(self, config: EngineConfig, container_mgr: ContainerManager) -> None:
        self._config = config
        self._container_mgr = container_mgr
        self._base_url = f"http://localhost:{config.api_port}"
        self._running_container: str | None = None

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def engine_id(self) -> str:
        return f"docker-{self._config.engine_id}"

    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            supported_types=self._config.task_types,
            vram_total_mb=self._config.vram_mb,
            models=list(self._config.task_type_params.keys()),
        )

    @property
    def backend_type(self) -> BackendType:
        return BackendType.DOCKER

    async def submit(self, workflow: dict[str, Any], params: dict[str, Any] | None = None) -> str:
        """Submit a task. For sync engines, this runs the full lifecycle.

        The workflow dict must contain:
          - task_id: str
          - task_type: str
          - params: dict (task parameters)
          - workspace: str (host workspace path)
        """
        params = params or {}
        task_id = workflow.get("task_id", "unknown")
        task_type = workflow.get("task_type", "")
        task_params = workflow.get("params", {})
        workspace = Path(workflow.get("workspace", "/workspace"))

        # 1. Ensure container running
        owned = await self._ensure_container_running(task_id)

        # 2. Build and send request
        try:
            endpoint = self._resolve_endpoint(task_type)
            payload = self._build_payload(task_id, task_type, task_params, workspace)
            result = await self._post_and_handle(task_id, endpoint, payload, task_params, workspace)
        finally:
            if owned:
                await self._stop_and_cleanup()

        # Return a synthetic job_id (sync engines complete immediately)
        return result.get("job_id", f"{self._config.name}-{task_id[:8]}")

    async def poll(self, engine_job_id: str) -> dict[str, Any]:
        """Sync engines complete immediately in submit(). Always done."""
        return {"status": "completed", "progress": 100.0}

    async def get_output(self, engine_job_id: str) -> dict[str, Any]:
        """Output files are already saved to .done/ during submit()."""
        return {"outputs": [], "status": "completed"}

    async def cancel(self, engine_job_id: str) -> bool:
        """Stop the container to cancel."""
        if self._running_container:
            self._container_mgr.stop_container(self._running_container)
            self._running_container = None
        return True

    async def health(self) -> dict[str, Any]:
        available = await self._container_mgr.check_existing(
            self._base_url, self._config.health_endpoint,
        )
        return {
            "status": EngineStatus.ONLINE if available else EngineStatus.OFFLINE,
            "available": available,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _ensure_container_running(self, task_id: str) -> bool:
        """Start container if not already running. Returns True if we own it."""
        existing = await self._container_mgr.check_existing(
            self._base_url, self._config.health_endpoint,
        )
        if existing:
            logger.info("[%s] reusing existing service on port %d", self._config.name, self._config.api_port)
            return False

        # Clean up zombies first
        self._container_mgr.cleanup_engine_containers(self._config.name, self._config.api_port)

        cid = await self._container_mgr.start_container(self._config, task_id)
        if not cid:
            raise RuntimeError(f"Failed to start container for {self._config.name}")

        self._running_container = cid
        healthy = await self._container_mgr.wait_for_health(
            self._base_url, self._config.health_endpoint, self._config.health_timeout,
        )
        if not healthy:
            raise RuntimeError(f"{self._config.name} health check failed")
        return True

    def _resolve_endpoint(self, task_type: str) -> str:
        """Resolve the HTTP endpoint for a task type."""
        return self._config.task_type_endpoints.get(task_type, self._config.submit_endpoint)

    def _build_payload(
        self,
        task_id: str,
        task_type: str,
        params: dict[str, Any],
        workspace: Path,
    ) -> dict[str, Any]:
        """Build the HTTP request payload from task parameters.

        Unpacks nested params (extra.{engine_name}) and resolves asset paths.
        """
        # Unpack engine-specific params
        engine_params = params.get("extra", {}).get(self._config.name, {})
        merged = {**params, **engine_params}
        merged.pop("extra", None)

        # Resolve asset file paths
        asset_dir = workspace / ".assets" / task_id
        assets_cfg = self._config.task_type_assets.get(task_type, {})
        for role, role_cfg in assets_cfg.items():
            if not isinstance(role_cfg, dict):
                continue
            accept = role_cfg.get("accept", [])
            if role == "source":
                files = self._find_assets(asset_dir, ["source*"])
            elif role == "target":
                files = self._find_assets(asset_dir, ["target*"])
            elif role in ("audio", "source_audio"):
                files = self._find_assets(asset_dir, ["audio*", "source*"])
            elif role in ("reference", "ref_audio"):
                files = self._find_assets(asset_dir, ["ref*", "reference*"])
            elif role == "image":
                files = self._find_assets(asset_dir, ["image*", "source*", "target*"])
            elif role == "video":
                files = self._find_assets(asset_dir, ["video*", "target*"])
            else:
                files = self._find_assets(asset_dir, [f"{role}*"])

            if files:
                # Store as container-internal path
                merged[f"{role}_path"] = f"/workspace/.assets/{task_id}/{files[0].name}"
                # Also set common aliases
                if role == "source":
                    merged.setdefault("source_path", f"/workspace/.assets/{task_id}/{files[0].name}")
                elif role == "target":
                    merged.setdefault("target_path", f"/workspace/.assets/{task_id}/{files[0].name}")

        # Ensure output directory exists
        output_dir = workspace / ".done" / task_id
        output_dir.mkdir(parents=True, exist_ok=True)

        return merged

    @staticmethod
    def _find_assets(asset_dir: Path, patterns: list[str]) -> list[Path]:
        """Find asset files matching any of the given glob patterns, sorted by size desc."""
        files: list[Path] = []
        if not asset_dir.exists():
            return files
        for pattern in patterns:
            files.extend(asset_dir.glob(pattern))
        # Deduplicate and sort by size (largest first)
        seen: set[str] = set()
        unique: list[Path] = []
        for f in sorted(files, key=lambda p: p.stat().st_size, reverse=True):
            if f.name not in seen:
                seen.add(f.name)
                unique.append(f)
        return unique

    async def _post_and_handle(
        self,
        task_id: str,
        endpoint: str,
        payload: dict[str, Any],
        params: dict[str, Any],
        workspace: Path,
    ) -> dict[str, Any]:
        """POST the payload and handle the response."""
        url = f"{self._base_url}{endpoint}"
        logger.info("[%s] POST %s (task=%s)", self._config.name, endpoint, task_id)

        async with httpx.AsyncClient(timeout=self._config.poll_timeout) as client:
            try:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
            except httpx.HTTPStatusError as exc:
                detail = exc.response.text[:500] if exc.response.text else str(exc)
                raise RuntimeError(f"{self._config.name} HTTP {exc.response.status_code}: {detail}") from exc
            except httpx.TransportError as exc:
                raise RuntimeError(f"{self._config.name} connection failed: {exc}") from exc

        return self._handle_response(task_id, data, params, workspace)

    def _handle_response(
        self,
        task_id: str,
        data: dict[str, Any],
        params: dict[str, Any],
        workspace: Path,
    ) -> dict[str, Any]:
        """Process the API response and save output files."""
        output_dir = workspace / ".done" / task_id

        # Check for error
        if data.get("error"):
            raise RuntimeError(f"{self._config.name}: {data['error']}")

        # Handle base64-encoded media responses
        if data.get("status") == "done" or data.get("success"):
            # Audio response (base64)
            if data.get("audio"):
                fmt = params.get("output_format", "wav")
                (output_dir / f"output.{fmt}").write_bytes(base64.b64decode(data["audio"]))
                return {"job_id": f"{self._config.name}-{task_id[:8]}", "output_saved": True}

            # Image response (base64)
            if data.get("image"):
                import re
                img_data = data["image"]
                # Strip data URL prefix if present
                if img_data.startswith("data:"):
                    img_data = re.sub(r"^data:image/\w+;base64,", "", img_data)
                ext = params.get("output_format", "png")
                (output_dir / f"output.{ext}").write_bytes(base64.b64decode(img_data))
                return {"job_id": f"{self._config.name}-{task_id[:8]}", "output_saved": True}

            # Multiple images response (base64 array) — FLUX style
            if data.get("images"):
                import re
                for idx, img_b64 in enumerate(data["images"]):
                    if img_b64.startswith("data:"):
                        img_b64 = re.sub(r"^data:image/\w+;base64,", "", img_b64)
                    ext = params.get("output_format", "png")
                    suffix = f"_{idx}" if len(data["images"]) > 1 else ""
                    (output_dir / f"output{suffix}.{ext}").write_bytes(base64.b64decode(img_b64))
                return {"job_id": f"{self._config.name}-{task_id[:8]}", "output_saved": True, "num_images": len(data["images"])}

            # File path response (engine saved to output_path)
            if data.get("output_path"):
                return {"job_id": f"{self._config.name}-{task_id[:8]}", "output_path": data["output_path"]}

            # Generic success — scan output directory
            if output_dir.exists() and any(output_dir.iterdir()):
                return {"job_id": f"{self._config.name}-{task_id[:8]}", "output_saved": True}

            return {"job_id": f"{self._config.name}-{task_id[:8]}"}

        raise RuntimeError(f"{self._config.name}: unexpected response: {data}")

    async def _stop_and_cleanup(self) -> None:
        """Stop our container and clean up GPU processes."""
        if self._running_container:
            self._container_mgr.stop_container(self._running_container)
            self._running_container = None

    async def stop(self) -> None:
        """Teardown: stop container and release GPU VRAM.

        Called by EnginePool.unload() when the engine is evicted or the
        application shuts down.  The base implementation is a no-op;
        Docker engines must override to actually stop their container.
        """
        await self._stop_and_cleanup()

        # Also clean up any zombie containers for this engine
        if self._config:
            self._container_mgr.cleanup_engine_containers(
                self._config.name, self._config.api_port,
            )
