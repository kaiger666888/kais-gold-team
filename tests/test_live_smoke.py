"""Live smoke tests for gold-team FastAPI application.

Spins up the actual FastAPI app in-process using httpx.AsyncClient with
ASGITransport, verifying the HTTP contract for health, engine listing,
task submission/retrieval, multi-type routing, and duplicate rejection.

httpx.ASGITransport does not emit ASGI lifespan events, so the fixture
manually enters the FastAPI lifespan context to register engines before
tests run.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from uuid import uuid4

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

# Ensure docker/gold-team is on sys.path so src.v6 imports work
_GOLD_TEAM_ROOT = str(Path(__file__).resolve().parent.parent)
if _GOLD_TEAM_ROOT not in sys.path:
    sys.path.insert(0, _GOLD_TEAM_ROOT)

# Set environment variables BEFORE importing the app to control engine startup.
# - Disable ComfyUI (no GPU in test environment)
# - Point ACE-Step at a non-existent host (won't start)
# - Disable TTS unified subprocess
os.environ.setdefault("COMFYUI_ENABLED", "false")
os.environ.setdefault("ACESTEP_API_HOST", "127.0.0.1")
os.environ.setdefault("ACESTEP_ROOT", "/nonexistent")
os.environ.setdefault("TTS_UNIFIED_ENABLED", "false")


def _reset_singletons():
    """Reset module-level singletons so each test starts clean.

    Without this, asyncio Queues inside singletons (TaskStore, EngineRouter,
    Executor) are bound to the event loop of a prior test and raise
    RuntimeError when a new function-scoped loop is created.
    """
    import src.v6.store as store_mod
    import src.v6.executor as executor_mod
    import src.v6.engine.router as router_mod
    import src.v6.engine_pool as pool_mod
    import src.v6.engine.local_pool as local_pool_mod
    import src.v6.engine.cloud_pool as cloud_pool_mod

    store_mod._store = None
    executor_mod._executor = None
    router_mod._router = None
    pool_mod._pool = None
    local_pool_mod._pool = None
    cloud_pool_mod._pool = None


@pytest_asyncio.fixture
async def client():
    """Create an httpx AsyncClient wired to the FastAPI app via ASGITransport.

    Resets singletons, then manually enters the lifespan context so engines
    (mock) are registered before requests are made.
    """
    _reset_singletons()

    from src.v6.main import app, lifespan

    async with lifespan(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac


class TestLiveSmoke:
    """End-to-end smoke tests hitting the real FastAPI app in-process."""

    @pytest.mark.asyncio
    async def test_health_endpoint(self, client: AsyncClient):
        """GET /health returns 200 with valid JSON structure."""
        resp = await client.get("/health")
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"

        data = resp.json()
        # Validate required keys
        for key in ("status", "version", "uptime_sec", "engines", "timestamp"):
            assert key in data, f"Missing key '{key}' in /health response"

        # At least the mock engine should be registered
        engines_info = data["engines"]
        assert engines_info["total"] >= 1, (
            f"Expected at least 1 engine, got {engines_info['total']}"
        )

    @pytest.mark.asyncio
    async def test_engines_endpoint(self, client: AsyncClient):
        """GET /api/v1/engines returns engine list with mock engine having backend_type."""
        resp = await client.get("/api/v1/engines")
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"

        data = resp.json()
        assert "engines" in data, "Response missing 'engines' key"
        engines = data["engines"]
        assert isinstance(engines, list), f"'engines' should be a list, got {type(engines)}"
        assert len(engines) >= 1, "Expected at least one engine in listing"

        # Find the mock engine
        mock_engine = None
        for eng in engines:
            if eng.get("id") == "mock":
                mock_engine = eng
                break

        assert mock_engine is not None, (
            f"Mock engine not found in engines list: {[e.get('id') for e in engines]}"
        )
        assert mock_engine.get("backend_type") == "mock", (
            f"Mock engine backend_type should be 'mock', got '{mock_engine.get('backend_type')}'"
        )

        # Verify each engine has the required keys
        required_keys = {"id", "name", "backend_type", "supported_types", "status"}
        for eng in engines:
            missing = required_keys - set(eng.keys())
            assert not missing, (
                f"Engine '{eng.get('id')}' missing keys: {missing}"
            )

    @pytest.mark.asyncio
    async def test_task_submit_and_retrieve(self, client: AsyncClient):
        """POST /api/v1/tasks accepts a valid payload and returns 202 with task_id."""
        task_id = f"smoke-{uuid4().hex[:12]}"

        # Submit task
        resp = await client.post("/api/v1/tasks", json={
            "task_id": task_id,
            "type": "image_draw",
            "params": {"prompt": "test image"},
        })
        assert resp.status_code == 202, f"Expected 202, got {resp.status_code}: {resp.text}"

        data = resp.json()
        assert data["task_id"] == task_id, (
            f"Expected task_id '{task_id}', got '{data.get('task_id')}'"
        )
        assert data["status"] == "queued", (
            f"Expected status 'queued', got '{data.get('status')}'"
        )

        # Retrieve the task
        get_resp = await client.get(f"/api/v1/tasks/{task_id}")
        assert get_resp.status_code == 200, (
            f"Expected 200 on GET task, got {get_resp.status_code}: {get_resp.text}"
        )

        task_data = get_resp.json()
        assert task_data["task_id"] == task_id, (
            f"Retrieved task_id mismatch: expected '{task_id}', got '{task_data.get('task_id')}'"
        )
        assert "status" in task_data, "Retrieved task missing 'status' field"

    @pytest.mark.asyncio
    async def test_task_submit_all_major_types(self, client: AsyncClient):
        """Submit one task per major TaskType, assert each returns 202 with status queued."""
        major_types = ["image_draw", "tts", "music", "upscale", "video_final"]

        for task_type in major_types:
            task_id = f"smoke-{task_type}-{uuid4().hex[:8]}"
            resp = await client.post("/api/v1/tasks", json={
                "task_id": task_id,
                "type": task_type,
                "params": {"prompt": f"test {task_type}"},
            })
            assert resp.status_code == 202, (
                f"Type '{task_type}': expected 202, got {resp.status_code}: {resp.text}"
            )
            data = resp.json()
            assert data["status"] == "queued", (
                f"Type '{task_type}': expected status 'queued', got '{data.get('status')}'"
            )

    @pytest.mark.asyncio
    async def test_duplicate_task_rejection(self, client: AsyncClient):
        """Submitting the same task_id twice returns 400 with duplicate_task_id error."""
        task_id = f"smoke-dup-{uuid4().hex[:8]}"
        payload = {
            "task_id": task_id,
            "type": "image_draw",
            "params": {"prompt": "duplicate test"},
        }

        # First submission: 202
        resp1 = await client.post("/api/v1/tasks", json=payload)
        assert resp1.status_code == 202, (
            f"First submission expected 202, got {resp1.status_code}: {resp1.text}"
        )

        # Second submission: 400
        resp2 = await client.post("/api/v1/tasks", json=payload)
        assert resp2.status_code == 400, (
            f"Second submission expected 400, got {resp2.status_code}: {resp2.text}"
        )

        error_data = resp2.json()
        detail = error_data.get("detail", {})
        assert "duplicate_task_id" in str(detail), (
            f"Expected 'duplicate_task_id' in error detail, got: {detail}"
        )
