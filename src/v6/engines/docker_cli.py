"""CLI engine — runs Blender and other CLI-mode tools in Docker containers."""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from src.v6.config.engine_schema import EngineConfig
from src.v6.docker.container_manager import ContainerManager
from src.v6.engines.base import BackendType, BaseEngine, EngineCapabilities, EngineStatus

logger = logging.getLogger(__name__)


class DockerCLIEngine(BaseEngine):
    """Engine for CLI-mode tools (e.g. Blender) that run via `docker run --rm`."""

    def __init__(self, config: EngineConfig, container_mgr: ContainerManager) -> None:
        self._config = config
        self._container_mgr = container_mgr

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def engine_id(self) -> str:
        return f"cli-{self._config.engine_id}"

    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            supported_types=self._config.task_types,
            vram_total_mb=self._config.vram_mb,
        )

    @property
    def backend_type(self) -> BackendType:
        return BackendType.DOCKER

    async def submit(self, workflow: dict[str, Any], params: dict[str, Any] | None = None) -> str:
        """Execute the CLI command in a Docker container. Blocks until complete."""
        params = params or {}
        task_id = workflow.get("task_id", "unknown")
        task_params = workflow.get("params", {})
        workspace = Path(workflow.get("workspace", "/workspace"))

        output_dir = workspace / ".done" / task_id
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd_args = self._build_command(task_id, task_params, workspace)

        gpu_flag = "all" if self._config.gpu_device == "all" else f'"device={self._config.gpu_device}"'
        docker_cmd = [
            "docker", "run", "--rm",
            "--gpus", gpu_flag,
            "-v", f"{workspace}:/workspace",
            "-w", "/workspace",
            "--label", "kais-worker=true",
            "--label", f"task_id={task_id}",
        ]
        docker_cmd.extend(self._config.extra_docker_args)
        docker_cmd.append(self._config.docker_image)
        docker_cmd.extend(cmd_args)

        logger.info("[CLI] %s: %s ...", self._config.name, " ".join(docker_cmd[:8]))

        proc = await asyncio.create_subprocess_exec(
            *docker_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            error = (stderr.decode(errors="replace") or stdout.decode(errors="replace"))[:2000]
            raise RuntimeError(f"{self._config.name} failed (rc={proc.returncode}): {error}")

        return f"cli-{task_id[:8]}"

    async def poll(self, engine_job_id: str) -> dict[str, Any]:
        """CLI tasks complete in submit(). Always done."""
        return {"status": "completed", "progress": 100.0}

    async def get_output(self, engine_job_id: str) -> dict[str, Any]:
        return {"outputs": [], "status": "completed"}

    async def cancel(self, engine_job_id: str) -> bool:
        return True

    async def health(self) -> dict[str, Any]:
        return {"status": EngineStatus.ONLINE, "available": True}

    def _build_command(
        self, task_id: str, params: dict[str, Any], workspace: Path,
    ) -> list[str]:
        """Build CLI command args based on engine type."""
        if self._config.name == "blender":
            return self._build_blender_command(task_id, params, workspace)
        # Generic: run a script
        script = params.get("script_path", "script.sh")
        return ["bash", f"/workspace/{script}"]

    @staticmethod
    def _build_blender_command(
        task_id: str, params: dict[str, Any], workspace: Path,
    ) -> list[str]:
        """Build Blender-specific CLI args."""
        output_dir = f"/workspace/.done/{task_id}"
        output_format = params.get("output_format", "PNG")
        samples = params.get("samples", 128)

        script_path = params.get("script_path")
        if script_path:
            return [
                "blender", "-b",
                "--python", f"/workspace/{script_path}",
                "--", "--output", f"{output_dir}/frame_####",
            ]

        blend_file = params.get("blend_file", "scene.blend")
        frame_start = params.get("frame_start", 1)
        frame_end = params.get("frame_end", 250)
        return [
            "blender", "-b", blend_file,
            "-o", f"{output_dir}/frame_####",
            "-F", output_format,
            "-s", str(frame_start),
            "-e", str(frame_end),
            "-a",
            "--", "--cycles-samples", str(samples),
        ]
