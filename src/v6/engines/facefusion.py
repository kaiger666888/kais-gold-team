"""FaceFusion engine — specialized DockerAPIEngine for face processing tasks."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import httpx

from src.v6.config.engine_schema import EngineConfig
from src.v6.docker.container_manager import ContainerManager
from src.v6.engines.docker_base import DockerAPIEngine

logger = logging.getLogger(__name__)


class FaceFusionEngine(DockerAPIEngine):
    """FaceFusion engine with specialized source/target detection and video handling."""

    async def submit(self, workflow: dict[str, Any], params: dict[str, Any] | None = None) -> str:
        params = params or {}
        task_id = workflow.get("task_id", "unknown")
        task_type = workflow.get("task_type", "face_swap")
        task_params = workflow.get("params", {})
        workspace = Path(workflow.get("workspace", "/workspace"))

        # Ensure container running
        owned = await self._ensure_container_running(task_id)

        try:
            payload = self._build_facefusion_payload(task_id, task_type, task_params, workspace)
            result = await self._post_and_handle(task_id, "/process", payload, task_params, workspace)
        finally:
            if owned:
                await self._stop_and_cleanup()

        return result.get("job_id", f"facefusion-{task_id[:8]}")

    def _build_facefusion_payload(
        self,
        task_id: str,
        task_type: str,
        params: dict[str, Any],
        workspace: Path,
    ) -> dict[str, Any]:
        """Build FaceFusion-specific payload with source/target file detection."""
        ff_params = params.get("extra", {}).get("facefusion", {})
        p = {**params, **ff_params}
        p.pop("extra", None)

        asset_dir = workspace / ".assets" / task_id
        source_files = sorted(asset_dir.glob("source*")) if asset_dir.exists() else []
        target_files = sorted(asset_dir.glob("target*")) if asset_dir.exists() else []
        processors = p.get("processors", ["face_swapper"])

        needs_source = "face_swapper" in processors or "lip_syncer" in processors
        if needs_source and not source_files:
            raise RuntimeError("No source file found for face processing")
        if not target_files:
            raise RuntimeError("No target file found for face processing")

        target_name = target_files[0].name.lower()
        is_video = target_name.endswith((".mp4", ".avi", ".mov", ".mkv", ".webm"))
        target_ext = target_files[0].suffix.lstrip(".").replace("jpeg", "jpg")
        output_ext = p.get("output_extension", "mp4" if is_video else target_ext or "png")
        output_path = f"/workspace/.done/{task_id}/output.{output_ext}"

        payload: dict[str, Any] = {
            "target_path": f"/workspace/.assets/{task_id}/{target_files[0].name}",
            "output_path": output_path,
            "processors": processors,
            "models": p.get("models", {"face_swapper": "inswapper_128_fp16"}),
            "execution_provider": "cpu",
            "video_memory_strategy": p.get("video_memory_strategy", "strict"),
        }

        if needs_source:
            payload["source_path"] = f"/workspace/.assets/{task_id}/{source_files[0].name}"

        if is_video or output_ext in ("mp4", "avi", "mov", "mkv", "webm"):
            payload["video_encoder"] = p.get("video_encoder", "libx264")
            payload["video_preset"] = p.get("video_preset", "fast")
            payload["video_quality"] = str(p.get("video_quality", "80"))
            payload["video_fps"] = str(p.get("video_fps", "30"))

        # Create output directory on host
        (workspace / ".done" / task_id).mkdir(parents=True, exist_ok=True)

        return payload
