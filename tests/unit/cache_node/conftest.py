"""Shared fixtures for cache-node unit tests.

Both services have a package called `app`, so we put only cache_node/ on
sys.path here. Run these tests on their own: pytest tests/unit/cache_node
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "cache_node"))

from app.cache.cache_engine import CacheEngine  # noqa: E402


class FakeClock:
    """A controllable clock so tests move time forward instead of sleeping."""

    def __init__(self, start: float = 1_000_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def cache(clock: FakeClock) -> CacheEngine:
    """An engine with capacity 3 driven by the fake clock."""
    return CacheEngine(capacity=3, clock=clock)
