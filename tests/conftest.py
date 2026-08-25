"""Shared fixtures for gold-team workflow builder tests."""
import sys
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _add_gold_team_to_path():
    """Ensure docker/gold-team is on sys.path so src.v6 imports work."""
    gold_team_root = str(Path(__file__).resolve().parent.parent)
    if gold_team_root not in sys.path:
        sys.path.insert(0, gold_team_root)


@pytest.fixture
def sample_seed() -> int:
    """Deterministic seed for reproducible workflow tests."""
    return 42
