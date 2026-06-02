"""Engine configuration schema — loaded from YAML definitions."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class EngineConfig:
    """Declarative configuration for a Docker-based engine."""

    name: str
    engine_id: str
    docker_image: str
    api_port: int
    requires_gpu: bool
    task_types: list[str]
    submit_endpoint: str = "/process"
    health_endpoint: str = "/health"
    health_timeout: int = 120
    poll_timeout: int = 600
    poll_interval: float = 3.0
    extra_docker_args: list[str] = field(default_factory=list)
    api_command: list[str] = field(default_factory=list)
    vram_mb: int = 0
    # Per-task-type endpoint overrides: task_type -> endpoint path
    task_type_endpoints: dict[str, str] = field(default_factory=dict)
    # Asset role definitions from YAML: task_type -> {role: {required, accept, ...}}
    task_type_assets: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Param schemas from YAML: task_type -> {param_name: {type, default, ...}}
    task_type_params: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Mode: "api", "cli", or "polling"
    mode: str = "api"
    # GPU device: "all" or a specific device id like "0", "1"
    gpu_device: str = "all"

    @classmethod
    def from_yaml(
        cls,
        yaml_cfg: dict[str, Any],
        tools_config: dict[str, Any] | None = None,
    ) -> EngineConfig:
        """Build an EngineConfig from a parsed YAML engine definition.

        Args:
            yaml_cfg: Parsed YAML content (e.g. from acestep.yaml).
            tools_config: Optional tools config from config.yaml for fallback values.
        """
        tools_config = tools_config or {}
        tool_cfg = tools_config.get(yaml_cfg.get("name", ""), {})

        task_types_raw = yaml_cfg.get("task_types", {})
        if isinstance(task_types_raw, dict):
            task_type_names = list(task_types_raw.keys())
        elif isinstance(task_types_raw, list):
            task_type_names = task_types_raw
        else:
            task_type_names = []

        # Extract per-task-type assets and params
        task_type_assets: dict[str, dict[str, Any]] = {}
        task_type_params: dict[str, dict[str, Any]] = {}
        if isinstance(task_types_raw, dict):
            for tt_name, tt_cfg in task_types_raw.items():
                if isinstance(tt_cfg, dict):
                    if "assets" in tt_cfg:
                        task_type_assets[tt_name] = tt_cfg["assets"]
                    if "params" in tt_cfg:
                        task_type_params[tt_name] = tt_cfg["params"]

        return cls(
            name=yaml_cfg.get("name", ""),
            engine_id=yaml_cfg.get("engine", yaml_cfg.get("name", "")),
            docker_image=yaml_cfg.get("docker_image", tool_cfg.get("docker_image", "")),
            api_port=yaml_cfg.get("api_port", tool_cfg.get("api_port", 0)),
            requires_gpu=yaml_cfg.get("requires_gpu", True),
            task_types=task_type_names,
            mode=yaml_cfg.get("mode", "api"),
            task_type_assets=task_type_assets,
            task_type_params=task_type_params,
            gpu_device=yaml_cfg.get("gpu_device", "all"),
        )
