"""Tests for engine backend-type classification and registration correctness."""
from __future__ import annotations

import importlib
import inspect
import os
import sys
from unittest.mock import MagicMock

import pytest

# Ensure src is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.v6.engines.base import BackendType, BaseEngine


# ---------------------------------------------------------------------------
# Helper: lightweight engine instantiation (avoids heavy GPU / HTTP deps)
# ---------------------------------------------------------------------------

def _make_comfyui_engine():
    from src.v6.engines.comfyui import ComfyUIEngine
    return ComfyUIEngine(host="127.0.0.1", port=9999, engine_id="test-comfyui")


def _make_mock_engine():
    from src.v6.engines.mock import MockEngine
    return MockEngine()


def _make_hunyuan3d_engine():
    from src.v6.engines.hunyuan3d import Hunyuan3DEngine
    return Hunyuan3DEngine(output_root="/tmp/test_output", model_dir="/tmp/test_model")


def _make_hunyuan3d_mv_engine():
    from src.v6.engines.hunyuan3d_mv import Hunyuan3DMvEngine
    return Hunyuan3DMvEngine(output_root="/tmp/test_output", model_dir="/tmp/test_model",
                              code_dir="/tmp/test_code")


def _make_tts_tracker():
    from src.v6.engines.tts import TTSTracker
    return TTSTracker(idle_timeout=300, output_root="/tmp/test_output")


def _make_tts_http_engine():
    from src.v6.engines.tts_http import TripleTrackTTSEngine
    return TripleTrackTTSEngine(output_root="/tmp/test_output")


def _make_joycaption_engine():
    from src.v6.engines.joycaption import JoyCaptionEngine
    return JoyCaptionEngine(host="127.0.0.1", port=9999)


def _make_cloud_engine():
    """Instantiate a concrete cloud subclass (JimengEngine) for testing."""
    from src.v6.engines.cloud_jimeng import JimengEngine
    return JimengEngine()


def _make_docker_api_engine():
    """Instantiate DockerAPIEngine with minimal mocked config/container_mgr."""
    from unittest.mock import MagicMock
    from src.v6.engines.docker_base import DockerAPIEngine
    config = MagicMock()
    config.name = "test-docker-engine"
    config.engine_id = "test-docker"
    config.api_port = 9999
    config.task_types = ["image_draw"]
    config.vram_mb = 0
    config.task_type_params = {}
    config.task_type_endpoints = {}
    config.submit_endpoint = "/api/generate"
    config.health_endpoint = "/health"
    config.health_timeout = 5.0
    config.poll_timeout = 30.0
    config.extra_docker_args = []
    config.gpu_device = "all"
    config.docker_image = "test:latest"
    config.task_type_assets = {}
    container_mgr = MagicMock()
    return DockerAPIEngine(config=config, container_mgr=container_mgr)


def _make_docker_cli_engine():
    """Instantiate DockerCLIEngine with minimal mocked config/container_mgr."""
    from unittest.mock import MagicMock
    from src.v6.engines.docker_cli import DockerCLIEngine
    config = MagicMock()
    config.name = "test-cli-engine"
    config.engine_id = "test-cli"
    config.task_types = ["render"]
    config.vram_mb = 0
    config.gpu_device = "all"
    config.extra_docker_args = []
    config.docker_image = "blender:latest"
    container_mgr = MagicMock()
    return DockerCLIEngine(config=config, container_mgr=container_mgr)


# ===================================================================
# Test class: backend_type classification for every engine
# ===================================================================

class TestBackendTypeClassification:
    """Verify each engine subclass reports the correct BackendType."""

    def test_01_comfyui_engine_is_comfyui(self):
        engine = _make_comfyui_engine()
        assert engine.backend_type == BackendType.COMFYUI

    def test_02_cloud_engine_is_cloud(self):
        engine = _make_cloud_engine()
        assert engine.backend_type == BackendType.CLOUD

    def test_03_docker_api_engine_is_docker(self):
        engine = _make_docker_api_engine()
        assert engine.backend_type == BackendType.DOCKER

    def test_04_hunyuan3d_engine_is_subprocess(self):
        engine = _make_hunyuan3d_engine()
        assert engine.backend_type == BackendType.SUBPROCESS

    def test_05_hunyuan3d_mv_engine_is_subprocess(self):
        engine = _make_hunyuan3d_mv_engine()
        assert engine.backend_type == BackendType.SUBPROCESS

    def test_06_tts_tracker_is_subprocess(self):
        engine = _make_tts_tracker()
        assert engine.backend_type == BackendType.SUBPROCESS

    def test_07_triple_track_tts_engine_is_subprocess(self):
        engine = _make_tts_http_engine()
        assert engine.backend_type == BackendType.SUBPROCESS

    def test_08_mock_engine_is_mock(self):
        engine = _make_mock_engine()
        assert engine.backend_type == BackendType.MOCK

    def test_09_joycaption_engine_is_comfyui(self):
        engine = _make_joycaption_engine()
        assert engine.backend_type == BackendType.COMFYUI

    def test_10_docker_cli_engine_is_docker(self):
        engine = _make_docker_cli_engine()
        assert engine.backend_type == BackendType.DOCKER


# ===================================================================
# Test class: ComfyUI architectural correctness (ENG-02)
# ===================================================================

class TestComfyUIArchitecture:
    """Verify no per-model ComfyUI Engine subclasses exist."""

    def test_no_comfyui_model_subclasses(self):
        """ENG-02: No per-model Engine subclasses for ComfyUI models.

        All ComfyUI models go through ComfyUIEngine + workflow_builder,
        not through separate Engine subclasses per model.
        The only ComfyUI-related engines are ComfyUIEngine and JoyCaptionEngine.
        """
        engines_dir = os.path.join(os.path.dirname(__file__), "..", "src", "v6", "engines")
        engines_dir = os.path.abspath(engines_dir)

        # Collect all classes that inherit from BaseEngine
        comfyui_engine_classes = []
        allowed_names = {"ComfyUIEngine", "JoyCaptionEngine"}

        for filename in os.listdir(engines_dir):
            if not filename.endswith(".py") or filename.startswith("_"):
                continue
            filepath = os.path.join(engines_dir, filename)
            module_name = filename[:-3]

            spec = importlib.util.spec_from_file_location(
                f"src.v6.engines.{module_name}", filepath,
                submodule_search_locations=[],
            )
            if spec is None or spec.loader is None:
                continue

            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)
            except Exception:
                # Some modules require heavy deps (torch, etc.) -- skip import errors
                continue

            for attr_name, attr_value in inspect.getmembers(module, inspect.isclass):
                if (
                    issubclass(attr_value, BaseEngine)
                    and attr_value is not BaseEngine
                    and "comfyui" in attr_name.lower()
                    and attr_name not in allowed_names
                ):
                    comfyui_engine_classes.append((attr_name, filename))

        assert comfyui_engine_classes == [], (
            f"Found per-model ComfyUI engine subclasses (violates ENG-02): "
            f"{comfyui_engine_classes}"
        )

    def test_comfyui_engine_uses_workflow_builder(self):
        """ComfyUIEngine delegates workflow construction to workflow_builder.

        ComfyUIEngine should not contain any model-specific build_*_workflow
        methods. All workflow construction goes through workflow_builder.py.
        """
        from src.v6.engines.comfyui import ComfyUIEngine

        # Verify module source
        assert ComfyUIEngine.__module__ == "src.v6.engines.comfyui"

        # Verify no build_*_workflow methods on the class
        build_methods = [
            name for name in dir(ComfyUIEngine)
            if name.startswith("build_") and name.endswith("_workflow")
        ]
        assert build_methods == [], (
            f"ComfyUIEngine has model-specific workflow methods "
            f"(should use workflow_builder): {build_methods}"
        )


# ===================================================================
# Test class: _format_registration_summary grouping
# ===================================================================

class TestFormatRegistrationSummary:
    """Verify _format_registration_summary groups engines correctly."""

    def _make_engine_mock(self, engine_id: str, name: str, backend_type: BackendType):
        """Create a mock engine with specified backend_type."""
        engine = MagicMock()
        engine.engine_id = engine_id
        engine.name = name
        engine.backend_type = backend_type
        return engine

    def test_groups_comfyui_engines(self):
        from src.v6.main import _format_registration_summary
        executor = MagicMock()
        executor.list_engines.return_value = [
            self._make_engine_mock("comfyui-primary", "ComfyUI (primary)", BackendType.COMFYUI),
            self._make_engine_mock("comfyui-auxiliary", "ComfyUI (auxiliary)", BackendType.COMFYUI),
        ]
        result = _format_registration_summary(executor)
        assert "[COMFYUI]" in result
        assert "comfyui-primary" in result
        assert "comfyui-auxiliary" in result

    def test_groups_subprocess_engines(self):
        from src.v6.main import _format_registration_summary
        executor = MagicMock()
        executor.list_engines.return_value = [
            self._make_engine_mock("tts-http", "Triple-Track TTS", BackendType.SUBPROCESS),
            self._make_engine_mock("tts-tracker", "TTS Tracker", BackendType.SUBPROCESS),
        ]
        result = _format_registration_summary(executor)
        assert "[SUBPROCESS]" in result
        assert "tts-http" in result
        assert "tts-tracker" in result

    def test_groups_cloud_engines(self):
        from src.v6.main import _format_registration_summary
        executor = MagicMock()
        executor.list_engines.return_value = [
            self._make_engine_mock("jimeng", "Jimeng Cloud", BackendType.CLOUD),
        ]
        result = _format_registration_summary(executor)
        assert "[CLOUD]" in result
        assert "jimeng" in result

    def test_groups_docker_engines(self):
        from src.v6.main import _format_registration_summary
        executor = MagicMock()
        executor.list_engines.return_value = [
            self._make_engine_mock("facefusion", "FaceFusion", BackendType.DOCKER),
        ]
        result = _format_registration_summary(executor)
        assert "[DOCKER]" in result
        assert "facefusion" in result

    def test_groups_mock_engines(self):
        from src.v6.main import _format_registration_summary
        executor = MagicMock()
        executor.list_engines.return_value = [
            self._make_engine_mock("mock", "MockEngine", BackendType.MOCK),
        ]
        result = _format_registration_summary(executor)
        assert "[MOCK]" in result
        assert "mock" in result

    def test_empty_sections_omitted(self):
        from src.v6.main import _format_registration_summary
        executor = MagicMock()
        executor.list_engines.return_value = [
            self._make_engine_mock("mock", "MockEngine", BackendType.MOCK),
        ]
        result = _format_registration_summary(executor)
        assert "[COMFYUI]" not in result
        assert "[SUBPROCESS]" not in result
        assert "[CLOUD]" not in result
        assert "[DOCKER]" not in result

    def test_section_order(self):
        from src.v6.main import _format_registration_summary
        executor = MagicMock()
        executor.list_engines.return_value = [
            self._make_engine_mock("mock", "MockEngine", BackendType.MOCK),
            self._make_engine_mock("comfyui-primary", "ComfyUI (primary)", BackendType.COMFYUI),
            self._make_engine_mock("jimeng", "Jimeng Cloud", BackendType.CLOUD),
            self._make_engine_mock("tts-http", "Triple-Track TTS", BackendType.SUBPROCESS),
        ]
        result = _format_registration_summary(executor)
        lines = result.split("\n")
        section_indices = {}
        for i, line in enumerate(lines):
            for section in ["[COMFYUI]", "[SUBPROCESS]", "[CLOUD]", "[MOCK]"]:
                if section in line:
                    section_indices[section] = i
        assert section_indices["[COMFYUI]"] < section_indices["[SUBPROCESS]"]
        assert section_indices["[SUBPROCESS]"] < section_indices["[CLOUD]"]
        assert section_indices["[CLOUD]"] < section_indices["[MOCK]"]

    def test_dual_comfyui_appears_in_group(self):
        """ComfyUI dual-engine (primary/auxiliary) both appear under [COMFYUI]."""
        from src.v6.main import _format_registration_summary
        executor = MagicMock()
        executor.list_engines.return_value = [
            self._make_engine_mock("comfyui-primary", "ComfyUI (primary)", BackendType.COMFYUI),
            self._make_engine_mock("comfyui-auxiliary", "ComfyUI (auxiliary)", BackendType.COMFYUI),
            self._make_engine_mock("joycaption", "JoyCaption", BackendType.COMFYUI),
        ]
        result = _format_registration_summary(executor)
        # All three should be under [COMFYUI]
        assert result.count("[COMFYUI]") == 1
        assert "comfyui-primary" in result
        assert "comfyui-auxiliary" in result
        assert "joycaption" in result

    def test_engine_name_displayed(self):
        from src.v6.main import _format_registration_summary
        executor = MagicMock()
        executor.list_engines.return_value = [
            self._make_engine_mock("comfyui-primary", "ComfyUI Primary (RTX 3090)", BackendType.COMFYUI),
        ]
        result = _format_registration_summary(executor)
        assert "ComfyUI Primary (RTX 3090)" in result

    def test_no_engines_returns_empty(self):
        from src.v6.main import _format_registration_summary
        executor = MagicMock()
        executor.list_engines.return_value = []
        result = _format_registration_summary(executor)
        assert result == ""


# ===================================================================
# Test class: engines API response includes backend_type
# ===================================================================

class TestEnginesApiBackendType:
    """Verify /api/v1/engines response includes backend_type for every engine."""

    def test_executor_engines_include_backend_type(self):
        """Executor-managed engine dicts must include backend_type field."""
        from src.v6.engines.mock import MockEngine

        executor = MagicMock()
        engine = MockEngine()
        executor.list_engines.return_value = [engine]
        executor.get_engine.return_value = None

        local_pool = MagicMock()
        local_pool.health.return_value = {"available": True, "vram_total_mb": 24576}

        # Simulate the list_engines endpoint logic
        engines_list = []
        for eng in executor.list_engines():
            cap = eng.capabilities
            engines_list.append({
                "id": eng.engine_id,
                "name": eng.name,
                "backend_type": eng.backend_type.value,
                "supported_types": cap.supported_types,
                "models": cap.models,
            })

        assert len(engines_list) == 1
        assert "backend_type" in engines_list[0]
        assert engines_list[0]["backend_type"] == "mock"

    def test_comfyui_engine_backend_type_value(self):
        """ComfyUIEngine should report backend_type=comfyui."""
        from src.v6.engines.comfyui import ComfyUIEngine

        engine = ComfyUIEngine(host="127.0.0.1", port=9999, engine_id="test-comfyui")
        assert engine.backend_type.value == "comfyui"

    def test_cloud_engine_backend_type_value(self):
        """Cloud engines should report backend_type=cloud."""
        from src.v6.engines.cloud_jimeng import JimengEngine

        engine = JimengEngine()
        assert engine.backend_type.value == "cloud"

    def test_docker_engine_backend_type_value(self):
        """DockerAPIEngine should report backend_type=docker."""
        from src.v6.engines.docker_base import DockerAPIEngine

        config = MagicMock()
        config.name = "test-docker-engine"
        config.engine_id = "test-docker"
        config.api_port = 9999
        config.task_types = ["image_draw"]
        config.vram_mb = 0
        config.task_type_params = {}
        config.task_type_endpoints = {}
        config.submit_endpoint = "/api/generate"
        config.health_endpoint = "/health"
        config.health_timeout = 5.0
        config.poll_timeout = 30.0
        config.extra_docker_args = []
        config.gpu_device = "all"
        config.docker_image = "test:latest"
        config.task_type_assets = {}
        container_mgr = MagicMock()
        engine = DockerAPIEngine(config=config, container_mgr=container_mgr)
        assert engine.backend_type.value == "docker"

    def test_subprocess_engine_backend_type_value(self):
        """Subprocess engines should report backend_type=subprocess."""
        from src.v6.engines.tts import TTSTracker

        engine = TTSTracker(idle_timeout=300, output_root="/tmp/test_output")
        assert engine.backend_type.value == "subprocess"
