"""Abstract base class for execution engines."""
from __future__ import annotations

import abc
import enum
from dataclasses import dataclass, field
from typing import Any, Optional


class EngineStatus(str, enum.Enum):
    """Engine health status."""
    ONLINE = "online"
    OFFLINE = "offline"
    BUSY = "busy"
    ERROR = "error"


@dataclass
class EngineCapabilities:
    """Describes what an engine can do."""
    supported_types: list[str] = field(default_factory=list)
    max_resolution: tuple[int, int] = (2048, 2048)
    max_duration_sec: float = 30.0
    vram_total_mb: int = 0
    vram_available_mb: int = 0
    models: list[str] = field(default_factory=list)


class BaseEngine(abc.ABC):
    """Abstract base for GPU execution engines.

    Every engine must implement:
    - submit: queue a workflow for execution
    - poll: check current status and progress
    - get_output: retrieve finished results
    - cancel: abort a running job
    - health: report engine status
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable engine name."""

    @property
    @abc.abstractmethod
    def engine_id(self) -> str:
        """Unique engine identifier (e.g. 'comfyui-local')."""

    @property
    @abc.abstractmethod
    def capabilities(self) -> EngineCapabilities:
        """Engine capability descriptor."""

    @abc.abstractmethod
    async def submit(self, workflow: dict[str, Any], params: dict[str, Any] | None = None) -> str:
        """Submit a workflow for execution.

        Args:
            workflow: Engine-specific workflow definition (e.g. ComfyUI API JSON).
            params: Optional override parameters.

        Returns:
            Engine-side job/task ID for tracking.
        """

    @abc.abstractmethod
    async def poll(self, engine_job_id: str) -> dict[str, Any]:
        """Poll execution status.

        Returns:
            Dict with at least: ``status`` (queued|running|completed|failed),
            ``progress`` (0.0–100.0).
        """

    @abc.abstractmethod
    async def get_output(self, engine_job_id: str) -> dict[str, Any]:
        """Retrieve output artifacts for a completed job.

        Returns:
            Dict with ``outputs`` list of artifact dicts (url, type, format).
        """

    @abc.abstractmethod
    async def cancel(self, engine_job_id: str) -> bool:
        """Attempt to cancel a running job.

        Returns:
            True if cancellation was accepted.
        """

    @abc.abstractmethod
    async def health(self) -> dict[str, Any]:
        """Return engine health info.

        Must include: ``status`` (EngineStatus), ``available`` (bool).
        """

    async def start(self) -> None:
        """Initialize engine resources. Override if needed."""

    async def stop(self) -> None:
        """Teardown engine resources. Override if needed."""
