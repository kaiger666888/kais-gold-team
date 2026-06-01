"""Engine registry — loads YAML definitions and builds engine instances.

Scans the engines/ directory for YAML files, creates EngineConfig objects,
and instantiates the appropriate engine class (DockerAPIEngine, DockerPollingAPIEngine,
DockerCLIEngine, or FaceFusionEngine) based on engine name and mode.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from src.v6.config.engine_schema import EngineConfig
from src.v6.docker.container_manager import ContainerManager
from src.v6.engines.base import BaseEngine
from src.v6.engines.docker_base import DockerAPIEngine
from src.v6.engines.docker_cli import DockerCLIEngine
from src.v6.engines.docker_polling import DockerPollingAPIEngine
from src.v6.engines.facefusion import FaceFusionEngine

logger = logging.getLogger(__name__)

# VRAM estimates per engine (MB) — for routing decisions
VRAM_ESTIMATES: dict[str, int] = {
    "facefusion": 24000,
    "wan": 16000,
    "ltx": 16000,
    "trellis": 16000,
    "hunyuan3d": 16000,
    "latentsync": 16000,
    "acestep": 16000,
    "flux": 22000,
    "flux-ipa": 22000,
    "sdxl": 8000,
    "hunyuan3d_mini": 8000,
    "stable_audio": 8000,
    "yue": 8000,
    "foleycrafter": 8000,
    "blender": 8000,
    "forge": 4000,
    "gpt_sovits": 4000,
    "liveportrait": 4000,
    "motiongpt": 4000,
    "musetalk": 4000,
    "parallax": 4000,
    "rife": 4000,
    "seed_vc": 4000,
    "woosh": 4000,
    "uvr5": 4000,
    "light": 2000,
    "moondream": 4000,
}

# Per-task-type endpoint overrides for engines with multiple endpoints
_TASK_TYPE_ENDPOINTS: dict[str, dict[str, str]] = {
    "light": {
        "transcribe": "/transcribe",
        "image_tag": "/tag_image",
        "tts": "/tts",
        "voice_convert": "/voice_convert",
    },
    "forge": {
        "tts": "/tts",
        "voice_convert": "/voice_convert",
    },
}

# Extra Docker args per engine (model mounts, env vars, etc.)
_EXTRA_DOCKER_ARGS: dict[str, list[str]] = {
    "acestep": [
        "-w", "/workspace",
        "-e", "PYTHONPATH=/opt/acestep/app",
        "-e", "HF_HUB_OFFLINE=1",
        "-e", "ACESTEP_LM_MODEL_PATH=acestep-5Hz-lm-1.7B",
    ],
    "facefusion": [
        "-w", "/app",
    ],
}


def build_engine_registry(
    engines_dir: Path,
    tools_config: dict[str, Any] | None = None,
    workspace: str = "/workspace",
) -> dict[str, BaseEngine]:
    """Scan engines/*.yaml, build task_type -> BaseEngine mapping.

    Args:
        engines_dir: Directory containing engine YAML definitions.
        tools_config: Optional per-engine config (docker_image, api_port overrides).
        workspace: Host workspace path for container volume mounts.

    Returns:
        Dict mapping task_type name -> BaseEngine instance.
    """
    tools_config = tools_config or {}
    container_mgr = ContainerManager(workspace)
    registry: dict[str, BaseEngine] = {}

    for yaml_path in sorted(engines_dir.glob("*.yaml")):
        if yaml_path.name in ("README.yaml",):
            continue

        try:
            yaml_cfg = yaml.safe_load(yaml_path.read_text())
        except Exception as e:
            logger.warning("Failed to load %s: %s", yaml_path, e)
            continue

        if not yaml_cfg or "name" not in yaml_cfg:
            continue

        config = EngineConfig.from_yaml(yaml_cfg, tools_config)

        # Skip engines without docker_image or api_port (config not ready)
        if not config.docker_image:
            continue
        if config.mode == "api" and config.api_port == 0:
            continue

        # Fill in VRAM estimate
        config.vram_mb = VRAM_ESTIMATES.get(config.name, 4000)

        # Fill in per-task-type endpoints
        if config.name in _TASK_TYPE_ENDPOINTS:
            config.task_type_endpoints = _TASK_TYPE_ENDPOINTS[config.name]

        # Fill in extra docker args
        if config.name in _EXTRA_DOCKER_ARGS:
            config.extra_docker_args = _EXTRA_DOCKER_ARGS[config.name]

        # Create the appropriate engine class
        engine = _create_engine(config, container_mgr)

        # Register for all task types
        for tt in config.task_types:
            if tt in registry:
                logger.warning("Duplicate task_type '%s': overriding %s with %s",
                               tt, registry[tt].engine_id, engine.engine_id)
            registry[tt] = engine

        logger.info("Registered engine '%s' for %d task types: %s",
                     config.name, len(config.task_types), config.task_types)

    return registry


def _create_engine(config: EngineConfig, container_mgr: ContainerManager) -> BaseEngine:
    """Create the appropriate engine class based on config."""
    if config.name == "acestep":
        return DockerPollingAPIEngine(config, container_mgr)
    if config.name == "blender":
        return DockerCLIEngine(config, container_mgr)
    if config.name == "facefusion":
        return FaceFusionEngine(config, container_mgr)
    return DockerAPIEngine(config, container_mgr)
