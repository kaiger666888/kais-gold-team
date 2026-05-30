"""V3.6 Configuration — Stage-Aware Scheduling for Dual GPU System.

Single machine (192.168.71.166), dual GPU:
    - GPU 0: RTX 3060 Ti 8G — display + NVENC/NVDEC + ffmpeg IO (host)
    - GPU 1: RTX 3090 24G (PCIe 4.0 x16) — CUDA inference primary
    + 32G RAM

Core principles:
    1. 3090 VRAM dynamic partitioning: Heavy determines Light pool size
    2. 3060Ti Combo overflow — DEPRECATED (3060Ti no longer does CUDA inference)
    3. CPU zero inference, only Blender + FFmpeg
    4. Atomic writes, Redis MULTI/EXEC, AOF everysec
"""

from __future__ import annotations

from .stage_config import STAGE_CONFIG, STAGE_CONFIG_INV
from .combo_config import COMBO_3060TI
from .models_registry import MODELS, LIGHT_MODELS, HEAVY_MODELS
from .routing_table import ROUTING_TABLE, build_routing_table

__all__ = [
    "STAGE_CONFIG",
    "STAGE_CONFIG_INV",
    "COMBO_3060TI",
    "MODELS",
    "LIGHT_MODELS",
    "HEAVY_MODELS",
    "ROUTING_TABLE",
    "build_routing_table",
]
