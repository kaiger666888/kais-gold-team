"""GPU configuration for single high-spec machine.

Hardware: Single machine (192.168.71.166) with dual GPU:
    - GPU 0: RTX 3060 Ti 8G — display + NVENC/NVDEC + ffmpeg IO (host)
    - GPU 1: RTX 3090 24G — CUDA inference primary

Default gpu_index=0 targets RTX 3060 Ti (display/IO).
For inference, use gpu_index=1 (RTX 3090).
"""
from __future__ import annotations

import logging
import subprocess

from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GPUConfig:
    gpu_index: int = 0
    total_vram_mb: int = 24576
    hard_cap_mb: int = 23500
    idle_threshold_mb: int = 500

    def read_vram_used_mb(self) -> int:
        """Read current VRAM usage via nvidia-smi."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "-i", str(self.gpu_index),
                 "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                return 0
            return int(result.stdout.strip())
        except Exception:
            return 0

    def read_vram_total_mb(self) -> int:
        try:
            result = subprocess.run(
                ["nvidia-smi", "-i", str(self.gpu_index),
                 "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                return self.total_vram_mb
            return int(result.stdout.strip())
        except Exception:
            return self.total_vram_mb

    def has_available(self, required_mb: int) -> bool:
        """Check if enough VRAM is available for a task."""
        if required_mb <= 0:
            return True
        total = self.read_vram_total_mb()
        used = self.read_vram_used_mb()
        available = total - used
        return available >= required_mb

    def kill_gpu_processes(self) -> None:
        """Kill orphaned GPU compute processes."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10,
            )
            pids = [p.strip() for p in result.stdout.strip().split("\n") if p.strip()]
            for pid in pids:
                try:
                    subprocess.run(["kill", "-9", pid], capture_output=True, timeout=5)
                except Exception:
                    pass
            if pids:
                logger.info("Killed %d orphaned GPU process(es)", len(pids))
        except Exception as e:
            logger.warning("GPU process cleanup failed: %s", e)
