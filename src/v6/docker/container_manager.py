"""Docker container lifecycle manager — start, health-check, stop, cleanup."""
from __future__ import annotations

import asyncio
import logging
import subprocess
import time
from typing import Any

from src.v6.config.engine_schema import EngineConfig

logger = logging.getLogger(__name__)

_LABEL = "kais-worker"


class ContainerManager:
    """Manages Docker container lifecycle for engine execution."""

    def __init__(self, workspace: str) -> None:
        self._workspace = workspace

    async def start_container(
        self,
        config: EngineConfig,
        task_id: str,
    ) -> str | None:
        """Start a detached API container for the given engine.

        Returns container ID on success, None on failure.
        """
        container_name = f"kais-{config.name}-{task_id[:8]}"
        gpu_flag = "all" if config.gpu_device == "all" else f'"device={config.gpu_device}"'
        cmd = [
            "docker", "run", "-d", "--rm",
            "--gpus", gpu_flag,
            "--name", container_name,
            "-p", f"{config.api_port}:{config.api_port}",
            "-v", f"{self._workspace}:/workspace",
            "-w", "/workspace",
            "--label", f"{_LABEL}=true",
            "--label", f"task_id={task_id}",
        ]
        cmd.extend(self._symlink_mount_args())
        cmd.extend(config.extra_docker_args)
        cmd.append(config.docker_image)
        if config.api_command:
            cmd.extend(config.api_command)

        logger.info("Starting container %s (image=%s, port=%d)", container_name, config.docker_image, config.api_port)
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            logger.error("Container start failed: %s", stderr.decode())
            return None
        cid = stdout.decode().strip()[:12]
        logger.info("Container started: %s (%s)", container_name, cid)
        return cid

    async def wait_for_health(self, base_url: str, path: str, timeout: int) -> bool:
        """Poll a health endpoint until 200 or timeout."""
        import httpx

        start = time.time()
        last_log = start
        async with httpx.AsyncClient(timeout=5) as client:
            while time.time() - start < timeout:
                try:
                    resp = await client.get(f"{base_url}{path}")
                    if resp.status_code == 200:
                        logger.info("API ready after %.1fs", time.time() - start)
                        return True
                except httpx.HTTPError:
                    pass
                now = time.time()
                if now - last_log > 30:
                    logger.info("Health check: still waiting (%.0fs)...", now - start)
                    last_log = now
                await asyncio.sleep(5)
        logger.error("Health check failed after %ds", timeout)
        return False

    async def check_existing(self, base_url: str, health_path: str) -> bool:
        """Check if a service is already running on the expected port."""
        import httpx
        try:
            async with httpx.AsyncClient(timeout=2) as client:
                resp = await client.get(f"{base_url}{health_path}")
                return resp.status_code == 200
        except httpx.HTTPError:
            return False

    def stop_container(self, container_id: str, grace_period: int = 10) -> None:
        """Gracefully stop and remove a container."""
        try:
            subprocess.run(
                ["docker", "stop", "-t", str(grace_period), container_id],
                capture_output=True, timeout=grace_period + 15,
            )
        except Exception:
            pass
        try:
            subprocess.run(["docker", "rm", "-f", container_id], capture_output=True, timeout=30)
        except Exception as e:
            logger.warning("Failed to remove container %s: %s", container_id, e)

    def cleanup_engine_containers(self, engine_name: str, port: int) -> None:
        """Remove zombie containers for the same engine or port."""
        cids: set[str] = set()
        # Name-based
        try:
            result = subprocess.run(
                ["docker", "ps", "-a", "-q", "--filter", f"name=kais-{engine_name}-"],
                capture_output=True, text=True, timeout=10,
            )
            for cid in result.stdout.strip().split("\n"):
                if cid:
                    cids.add(cid)
        except Exception:
            pass
        # Port-based
        try:
            result = subprocess.run(
                ["docker", "ps", "-a", "--format", "{{.ID}} {{.Ports}}"],
                capture_output=True, text=True, timeout=10,
            )
            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue
                parts = line.split(" ", 1)
                if len(parts) < 2:
                    continue
                cid, ports = parts
                if f":{port}->" in ports:
                    cids.add(cid)
        except Exception:
            pass
        for cid in cids:
            logger.info("Cleaning zombie container %s (engine=%s)", cid, engine_name)
            self.stop_container(cid)

    async def wait_vram_drain(self, baseline_mb: int, tolerance_mb: int = 100, timeout: int = 15) -> None:
        """Wait until VRAM returns to baseline after cleanup."""
        from src.v6.config.gpu_config import GPUConfig
        gpu = GPUConfig()
        target_mb = baseline_mb + tolerance_mb
        start = time.time()
        while time.time() - start < timeout:
            vram = gpu.read_vram_used_mb()
            if vram <= target_mb:
                return
            await asyncio.sleep(1)
        logger.warning("VRAM drain timeout, current: %dMB", gpu.read_vram_used_mb())

    def _symlink_mount_args(self) -> list[str]:
        """Extra -v args for workspace symlinks pointing outside the mount."""
        from pathlib import Path
        args: list[str] = []
        try:
            for child in Path(self._workspace).iterdir():
                if child.is_symlink():
                    target = child.resolve()
                    if target.exists() and not str(target).startswith(str(self._workspace)):
                        args.extend(["-v", f"{target}:{target}"])
        except Exception:
            pass
        return args
