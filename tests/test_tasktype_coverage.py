"""TaskType routing coverage tests — every TaskType must resolve to at least one engine.

Verifies:
  1. Every TaskType enum value has >=1 engine in supported_types
  2. VIDEO_FINAL routes without params.extra.mode (default Wan I2V)
  3. IMAGE_DRAW routes without params.extra.mode (default FLUX Dev)
  4. UPSCALE routes without params.extra.mode (default image upscale)
  5. MUSIC routes to an engine
  6. TTS routes to an engine
  7. IMAGE_TO_3D routes to an engine
"""
from __future__ import annotations

import pytest

from src.v6.engine.router import EngineRouter
from src.v6.engines.base import BaseEngine, EngineCapabilities
from src.v6.engines.mock import MockEngine
from src.v6.executor import TaskExecutor
from src.v6.models.task import GenerationTask, TaskStatus, TaskType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# TaskTypes covered by the MockEngine's _MOCK_OUTPUTS keys
_MOCK_COVERED_TYPES = {
    TaskType.VIDEO_FINAL,
    TaskType.VIDEO_PREVIEW,
    TaskType.IMAGE_DRAW,
    TaskType.IMAGE_REFINE,
    TaskType.TTS,
    TaskType.MUSIC,
    TaskType.SFX,
    TaskType.UPSCALE,
    TaskType.FACE_RESTORE,
    TaskType.IMAGE_TO_3D,
}

# TaskTypes covered by dedicated subprocess engines
_SUBPROCESS_COVERED_TYPES = {
    TaskType.TTS,
    TaskType.TTS_ZH,
    TaskType.TTS_EN,
    TaskType.TTS_BILINGUAL,
    TaskType.MUSIC,
    TaskType.SFX,
    TaskType.IMAGE_TO_3D,
    TaskType.IMAGE_TO_3D_MV,
}

# Types that need ComfyUI but have no dedicated engine (routed via comfyui-primary)
_COMFYUI_ONLY_TYPES = {
    TaskType.IMAGE_PULID,
    TaskType.IMAGE_DRAW_IPADAPTER,
    TaskType.CONTROLNET_DEPTH,
    TaskType.WAN_I2V,
}


def _make_task(
    task_type: TaskType = TaskType.IMAGE_TO_3D,
    params: dict | None = None,
    task_id: str = "test-task-001",
) -> GenerationTask:
    """Create a GenerationTask with sensible defaults for routing tests."""
    return GenerationTask(
        task_id=task_id,
        type=task_type,
        params=params or {},
        status=TaskStatus.QUEUED,
    )


class _StubEngine(BaseEngine):
    """Lightweight stub engine for test registration."""

    def __init__(self, eid: str, supported: list[str]) -> None:
        self._eid = eid
        self._supported = supported

    @property
    def name(self) -> str:
        return f"StubEngine({self._eid})"

    @property
    def engine_id(self) -> str:
        return self._eid

    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(supported_types=self._supported)

    async def submit(self, workflow, params=None):
        return "stub-job"

    async def poll(self, engine_job_id: str):
        return {"status": "completed", "progress": 100.0}

    async def get_output(self, engine_job_id: str):
        return {"outputs": []}

    async def cancel(self, engine_job_id: str):
        return True

    async def health(self):
        return {"status": "online", "available": True}


def _build_test_executor() -> TaskExecutor:
    """Build a TaskExecutor with engines covering all TaskTypes."""
    executor = TaskExecutor()

    # Mock engine covers the common ComfyUI-routed types
    mock = MockEngine()
    executor.register_engine(mock)

    # Stub dedicated engines for subprocess/special types
    executor.register_engine(_StubEngine(
        "hunyuan3d-local",
        [TaskType.IMAGE_TO_3D.value],
    ))
    executor.register_engine(_StubEngine(
        "hunyuan3d-mv-local",
        [TaskType.IMAGE_TO_3D_MV.value],
    ))
    executor.register_engine(_StubEngine(
        "acestep-internal",
        [TaskType.MUSIC.value, TaskType.SFX.value],
    ))
    executor.register_engine(_StubEngine(
        "tts-tracker",
        [t.value for t in (TaskType.TTS, TaskType.TTS_ZH, TaskType.TTS_EN, TaskType.TTS_BILINGUAL)],
    ))

    # Stubs for ComfyUI-routed types that mock doesn't explicitly list
    executor.register_engine(_StubEngine(
        "comfyui-primary",
        [t.value for t in (
            TaskType.IMAGE_PULID, TaskType.IMAGE_DRAW_IPADAPTER,
            TaskType.CONTROLNET_DEPTH, TaskType.WAN_I2V,
            TaskType.VIDEO_FINAL, TaskType.VIDEO_PREVIEW,
            TaskType.IMAGE_DRAW, TaskType.IMAGE_REFINE,
            TaskType.UPSCALE, TaskType.FACE_RESTORE,
        )],
    ))

    return executor


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTaskTypeRoutingCoverage:
    """Verify every TaskType routes to at least one engine."""

    def test_every_tasktype_has_engine_coverage(self):
        """Every TaskType enum value must have at least one engine in supported_types."""
        executor = _build_test_executor()
        all_supported: set[str] = set()
        for engine in executor.list_engines():
            all_supported.update(engine.capabilities.supported_types)

        uncovered: list[str] = []
        for tt in TaskType:
            if tt.value not in all_supported:
                uncovered.append(tt.value)

        assert not uncovered, (
            f"TaskTypes with NO engine coverage: {uncovered}. "
            f"Add an engine that declares them in supported_types."
        )

    def test_video_final_routes_to_engine(self):
        """VIDEO_FINAL routes to an engine without params.extra.mode (default Wan I2V)."""
        executor = _build_test_executor()
        router = EngineRouter(
            local_available=True,
            local_vram_used_gb=0.0,
            primary_available=True,
            auxiliary_available=True,
        )
        task = _make_task(TaskType.VIDEO_FINAL, params={"image": "test.png"})
        pool, engine_id = router.route(task)
        engine = executor._resolve_engine(engine_id, task)
        assert engine is not None, f"VIDEO_FINAL did not resolve to any engine (routed to '{engine_id}')"

    def test_image_draw_routes_to_engine(self):
        """IMAGE_DRAW routes to an engine without params.extra.mode (default FLUX Dev)."""
        executor = _build_test_executor()
        router = EngineRouter(
            local_available=True,
            local_vram_used_gb=0.0,
            primary_available=True,
            auxiliary_available=True,
        )
        task = _make_task(TaskType.IMAGE_DRAW, params={"prompt": "a landscape"})
        pool, engine_id = router.route(task)
        engine = executor._resolve_engine(engine_id, task)
        assert engine is not None, f"IMAGE_DRAW did not resolve to any engine (routed to '{engine_id}')"

    def test_upscale_routes_to_engine(self):
        """UPSCALE routes to an engine without params.extra.mode (default image upscale)."""
        executor = _build_test_executor()
        router = EngineRouter(
            local_available=True,
            local_vram_used_gb=0.0,
            primary_available=True,
            auxiliary_available=True,
        )
        task = _make_task(TaskType.UPSCALE, params={"image": "test.png"})
        pool, engine_id = router.route(task)
        engine = executor._resolve_engine(engine_id, task)
        assert engine is not None, f"UPSCALE did not resolve to any engine (routed to '{engine_id}')"

    def test_music_routes_to_engine(self):
        """MUSIC routes to an engine (ACE-Step via DEDICATED_ENGINES)."""
        executor = _build_test_executor()
        router = EngineRouter(
            local_available=True,
            local_vram_used_gb=0.0,
            primary_available=True,
            auxiliary_available=True,
        )
        task = _make_task(TaskType.MUSIC, params={"prompt": "upbeat pop"})
        pool, engine_id = router.route(task)
        engine = executor._resolve_engine(engine_id, task)
        assert engine is not None, f"MUSIC did not resolve to any engine (routed to '{engine_id}')"
        # MUSIC is a dedicated type — should route to acestep-internal
        assert engine_id == "acestep-internal", f"MUSIC should route to acestep-internal, got '{engine_id}'"

    def test_tts_routes_to_engine(self):
        """TTS routes to an engine (subprocess TTS engines via DEDICATED_ENGINES)."""
        executor = _build_test_executor()
        router = EngineRouter(
            local_available=True,
            local_vram_used_gb=0.0,
            primary_available=True,
            auxiliary_available=True,
        )
        task = _make_task(TaskType.TTS, params={"text": "hello", "voice": "default"})
        pool, engine_id = router.route(task)
        engine = executor._resolve_engine(engine_id, task)
        assert engine is not None, f"TTS did not resolve to any engine (routed to '{engine_id}')"
        assert engine_id == "tts-tracker", f"TTS should route to tts-tracker, got '{engine_id}'"

    def test_image_to_3d_routes_to_engine(self):
        """IMAGE_TO_3D routes to an engine (Hunyuan3D subprocess engine)."""
        executor = _build_test_executor()
        router = EngineRouter(
            local_available=True,
            local_vram_used_gb=0.0,
            primary_available=True,
            auxiliary_available=True,
        )
        task = _make_task(TaskType.IMAGE_TO_3D, params={"input_image": "test.png"})
        pool, engine_id = router.route(task)
        engine = executor._resolve_engine(engine_id, task)
        assert engine is not None, f"IMAGE_TO_3D did not resolve to any engine (routed to '{engine_id}')"
        assert engine_id == "hunyuan3d-local", f"IMAGE_TO_3D should route to hunyuan3d-local, got '{engine_id}'"
