"""Unit tests for ACEStepEngine dual-mode detection (external container vs localhost subprocess).

These tests verify that ACEStepEngine.start() correctly detects whether
ACESTEP_API_HOST points to an external container (e.g. "kais-acestep") or a
local address ("127.0.0.1" / "localhost") and takes the appropriate startup path.

RED phase: Tests written BEFORE implementation changes.
"""
from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.v6.engines.acestep import ACEStepEngine, _is_external_host
from src.v6.engines.base import BackendType


class TestIsExternalHost:
    """Test the _is_external_host helper function."""

    def test_external_hostname(self):
        """Non-local hostname like 'kais-acestep' is external."""
        assert _is_external_host("kais-acestep") is True

    def test_external_ip(self):
        """Non-local IP like '192.168.1.100' is external."""
        assert _is_external_host("192.168.1.100") is True

    def test_localhost_is_local(self):
        """'localhost' is NOT external."""
        assert _is_external_host("localhost") is False

    def test_127_is_local(self):
        """'127.0.0.1' is NOT external."""
        assert _is_external_host("127.0.0.1") is False

    def test_ipv6_loopback_is_local(self):
        """'::1' (IPv6 loopback) is NOT external."""
        assert _is_external_host("::1") is False


class TestACEStepDualMode:
    """Test ACEStepEngine.start() dual-mode behavior."""

    @pytest.fixture
    def engine(self):
        """Create a fresh ACEStepEngine instance."""
        return ACEStepEngine()

    @pytest.mark.asyncio
    async def test_external_mode_skips_subprocess(self, engine):
        """When ACESTEP_API_HOST is 'kais-acestep', start() does NOT launch a subprocess.

        Instead it health-checks the external URL and sets _ready=True on 200.
        """
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("src.v6.engines.acestep.ACESTEP_HOST", "kais-acestep"), \
             patch("src.v6.engines.acestep.ACESTEP_PORT", 8010), \
             patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_response), \
             patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_subprocess, \
             patch("asyncio.create_task"):

            # Rebuild _base_url since ACESTEP_HOST changed
            engine._base_url = "http://kais-acestep:8010"
            await engine.start()

        # No subprocess should be launched in external mode
        mock_subprocess.assert_not_called()
        assert engine._process is None
        assert engine._ready is True

    @pytest.mark.asyncio
    async def test_external_mode_health_check_retries(self, engine):
        """External mode retries health-check until success.

        Mock httpx to fail 3 times then return 200. Verify _ready becomes True.
        """
        mock_response_ok = MagicMock()
        mock_response_ok.status_code = 200

        call_count = 0

        async def mock_get(url, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                raise Exception("connection refused")
            return mock_response_ok

        with patch("src.v6.engines.acestep.ACESTEP_HOST", "kais-acestep"), \
             patch("src.v6.engines.acestep.ACESTEP_PORT", 8010), \
             patch("httpx.AsyncClient.get", side_effect=mock_get), \
             patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_subprocess, \
             patch("asyncio.create_task"):

            engine._base_url = "http://kais-acestep:8010"
            await engine.start()

        mock_subprocess.assert_not_called()
        assert call_count == 4  # 3 failures + 1 success
        assert engine._ready is True

    @pytest.mark.asyncio
    async def test_external_mode_skips_dir_checks(self, engine):
        """In external mode, ACESTEP_ROOT directory check is skipped entirely.

        Even when ACESTEP_ROOT points to /nonexistent, start() should proceed.
        """
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("src.v6.engines.acestep.ACESTEP_HOST", "kais-acestep"), \
             patch("src.v6.engines.acestep.ACESTEP_ROOT", "/nonexistent"), \
             patch("src.v6.engines.acestep.ACESTEP_CHECKPOINTS", "/also-nonexistent"), \
             patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_response), \
             patch("asyncio.create_subprocess_exec", new_callable=AsyncMock), \
             patch("asyncio.create_task"):

            engine._base_url = "http://kais-acestep:8010"
            await engine.start()

        # Engine should be ready even though ACESTEP_ROOT doesn't exist
        assert engine._ready is True

    @pytest.mark.asyncio
    async def test_localhost_mode_enters_subprocess_path(self, engine):
        """When ACESTEP_API_HOST is '127.0.0.1', start() enters the subprocess path.

        With ACESTEP_ROOT missing, it should return early with a warning (existing behavior).
        """
        with patch("src.v6.engines.acestep.ACESTEP_HOST", "127.0.0.1"), \
             patch("src.v6.engines.acestep.ACESTEP_ROOT", "/nonexistent"):

            engine._base_url = "http://127.0.0.1:8010"
            await engine.start()

        # Should not be ready -- root dir is missing, subprocess path returns early
        assert engine._ready is False
        assert engine._process is None

    @pytest.mark.asyncio
    async def test_external_mode_timeout_does_not_set_ready(self, engine):
        """When external health-check times out, _ready stays False."""
        async def mock_get_fail(url, **kwargs):
            raise Exception("connection refused")

        with patch("src.v6.engines.acestep.ACESTEP_HOST", "kais-acestep"), \
             patch("src.v6.engines.acestep.ACESTEP_PORT", 8010), \
             patch("httpx.AsyncClient.get", side_effect=mock_get_fail), \
             patch("asyncio.create_subprocess_exec", new_callable=AsyncMock), \
             patch("asyncio.create_task"):

            engine._base_url = "http://kais-acestep:8010"
            await engine.start()

        # Should not be ready -- health-check never succeeds
        assert engine._ready is False


class TestACEStepBackendType:
    """Regression test for ACEStepEngine.backend_type classification.

    v1.4 FIX-04 / FIX-06: ACEStepEngine previously inherited the BaseEngine
    default of BackendType.MOCK, causing it to appear under the [MOCK] group
    in registration logs and to report ``backend_type: "mock"`` in the
    ``/api/v1/engines`` response — even though ACE-Step runs in a real
    sidecar container. These tests pin the classification to DOCKER so the
    bug cannot silently regress.
    """

    def test_backend_type_is_docker(self):
        """ACEStepEngine.backend_type must be BackendType.DOCKER."""
        engine = ACEStepEngine()
        assert engine.backend_type == BackendType.DOCKER

    def test_backend_type_is_not_mock(self):
        """ACEStepEngine.backend_type must NOT be MOCK (the inherited default).

        This is the direct regression guard for v1.3 ENG-04.
        """
        engine = ACEStepEngine()
        assert engine.backend_type != BackendType.MOCK

    def test_backend_type_is_valid_enum(self):
        """backend_type must be a BackendType enum member (not a string)."""
        engine = ACEStepEngine()
        assert isinstance(engine.backend_type, BackendType)

    def test_backend_type_value_string_for_api_response(self):
        """The enum's string value matches what /api/v1/engines should return."""
        engine = ACEStepEngine()
        assert engine.backend_type.value == "docker"
