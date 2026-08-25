"""Regression verification tests for Phase 15-18 features.

Verifies that ACE-Step music generation, cloud engine fallback, and
movie-agent removal remain intact across codebase evolution.

Test classes:
    TestACEStepRegression  — ACE-Step engine registration and MUSIC routing
    TestCloudFallback      — Cloud engine classification and fallback paths
    TestMovieAgentRemoval  — Zero movie-agent references in source and config
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

# Ensure src is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.v6.engines.base import BackendType, BaseEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _grep_count(pattern: str, path: str, include: str = "*", exclude_pycache: bool = True) -> int:
    """Run grep -r and return the number of matching lines (exit code 0)."""
    cmd = ["grep", "-r", "-E", pattern, path]
    if include != "*":
        cmd.extend(["--include", include])
    if exclude_pycache:
        cmd.extend(["--exclude-dir", "__pycache__"])
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        return len(result.stdout.strip().splitlines())
    return 0


# ===================================================================
# Test class: ACE-Step regression — REMOVED in v1.5
# ===================================================================
# ACE-Step engine deleted from gold-team in v1.5. Music generation now
# runs entirely through Node-layer ComfyUI routes (src/routes/v1/ace/).
# The 4 regression tests below are obsolete:
#   - test_01_engine_instantiation
#   - test_02_backend_type_is_docker
#   - test_03_supported_types_includes_music
#   - test_04_task_type_map_music_to_text2music


# ===================================================================
# Test class: Cloud engine fallback
# ===================================================================

class TestCloudFallback:
    """Verify cloud engine classification and fallback paths."""

    def test_05_kling_backend_type_cloud(self):
        """KlingEngine has backend_type == CLOUD."""
        from src.v6.engines.cloud_kling import KlingEngine
        engine = KlingEngine()
        assert engine.backend_type == BackendType.CLOUD

    def test_06_jimeng_backend_type_cloud(self):
        """JimengEngine has backend_type == CLOUD."""
        from src.v6.engines.cloud_jimeng import JimengEngine
        engine = JimengEngine()
        assert engine.backend_type == BackendType.CLOUD

    def test_07_seedance_backend_type_cloud(self):
        """SeedanceEngine has backend_type == CLOUD."""
        from src.v6.engines.cloud_seedance import SeedanceEngine
        engine = SeedanceEngine()
        assert engine.backend_type == BackendType.CLOUD

    def test_08_kling_supported_types_includes_video_final(self):
        """KlingEngine supports video_final tasks."""
        from src.v6.engines.cloud_kling import KlingEngine
        engine = KlingEngine()
        assert "video_final" in engine._supported_types, (
            f"'video_final' not in KlingEngine._supported_types: {engine._supported_types}"
        )

    def test_09_jimeng_supported_types_includes_image_draw(self):
        """JimengEngine supports image_draw tasks."""
        from src.v6.engines.cloud_jimeng import JimengEngine
        engine = JimengEngine()
        assert "image_draw" in engine._supported_types, (
            f"'image_draw' not in JimengEngine._supported_types: {engine._supported_types}"
        )

    def test_10_seedance_supported_types_includes_video_final(self):
        """SeedanceEngine supports video_final tasks."""
        from src.v6.engines.cloud_seedance import SeedanceEngine
        engine = SeedanceEngine()
        assert "video_final" in engine._supported_types, (
            f"'video_final' not in SeedanceEngine._supported_types: {engine._supported_types}"
        )

    def test_11_cloud_fallback_video_final_without_comfyui(self):
        """When no ComfyUI engine is registered, VIDEO_FINAL resolves to cloud engine.

        This proves the fallback path exists: if ComfyUI is down, the executor
        can still route tasks to cloud engines.
        """
        from unittest.mock import MagicMock

        from src.v6.engines.cloud_kling import KlingEngine
        from src.v6.engines.mock import MockEngine
        from src.v6.executor import TaskExecutor
        from src.v6.models.task import TaskType

        executor = TaskExecutor()

        # Register only a cloud engine and a mock — no ComfyUI
        cloud_engine = KlingEngine()
        mock_engine = MockEngine()

        executor.register_engine(cloud_engine)
        executor.register_engine(mock_engine)

        # Verify no ComfyUI engine is present
        for eng in executor.list_engines():
            assert eng.backend_type != BackendType.COMFYUI, "Unexpected ComfyUI engine registered"

        # Resolve engine for cloud-kling (simulating router assigning cloud engine)
        # The _resolve_engine method should find the Kling cloud engine
        task = MagicMock()
        task.type = TaskType.VIDEO_FINAL
        task.params = {"extra": {}}

        resolved = executor._resolve_engine("cloud-kling", task)
        assert resolved is not None, "No engine resolved for cloud-kling"
        assert resolved.backend_type == BackendType.CLOUD, (
            f"Resolved engine is not CLOUD: {resolved.backend_type}"
        )


# ===================================================================
# Test class: Movie-agent removal
# ===================================================================

class TestMovieAgentRemoval:
    """Verify zero movie-agent references in source code and configuration."""

    def test_12_no_movie_agent_in_source(self):
        """No Python/YAML/JSON file under docker/gold-team/src/ contains 'movie-agent' or 'movie_agent'.

        This ensures all movie-agent coupling has been removed from the
        gold-team source code (Phase 15 cleanup).
        """
        src_dir = os.path.join(os.path.dirname(__file__), "..", "src")
        src_dir = os.path.abspath(src_dir)

        count = _grep_count(
            r"movie[-_.]agent|movie_agent",
            src_dir,
            include="*.py",
        )
        # Also check YAML and JSON
        count += _grep_count(
            r"movie[-_.]agent|movie_agent",
            src_dir,
            include="*.yaml",
        )
        count += _grep_count(
            r"movie[-_.]agent|movie_agent",
            src_dir,
            include="*.yml",
        )
        count += _grep_count(
            r"movie[-_.]agent|movie_agent",
            src_dir,
            include="*.json",
        )

        assert count == 0, (
            f"Found {count} movie-agent/movie_agent reference(s) in docker/gold-team/src/"
        )

    def test_13_no_movie_agent_in_active_compose_files(self):
        """Active docker-compose files contain no movie-agent service definitions.

        Checks docker-compose.v9.yml, docker-compose.test.yml,
        docker-compose.real.yml, and docker-compose.smoke.yml.
        Old v6/v8 files are deprecated and may still have references.
        """
        project_root = os.path.join(os.path.dirname(__file__), "..", "..")
        project_root = os.path.abspath(project_root)

        active_files = [
            "docker-compose.v9.yml",
            "docker-compose.test.yml",
            "docker-compose.real.yml",
            "docker-compose.smoke.yml",
        ]

        total_matches = 0
        for fname in active_files:
            fpath = os.path.join(project_root, fname)
            if not os.path.isfile(fpath):
                continue
            total_matches += _grep_count("movie-agent", fpath, include="*.yml")

        assert total_matches == 0, (
            f"Found {total_matches} movie-agent reference(s) in active docker-compose files"
        )

    def test_14_no_movie_agent_imports(self):
        """No Python import references movie_agent module."""
        src_dir = os.path.join(os.path.dirname(__file__), "..", "src")
        src_dir = os.path.abspath(src_dir)

        result = subprocess.run(
            ["grep", "-r", "-E", r"from\s+movie_agent|import\s+movie_agent",
             src_dir, "--include", "*.py", "--exclude-dir", "__pycache__"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0, (
            f"Found movie_agent import(s) in source:\n{result.stdout}"
        )
