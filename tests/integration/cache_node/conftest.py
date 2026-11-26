"""Fixtures for testing the cache node over HTTP.

We use FastAPI's TestClient, which is built on HTTPX. It calls the app
in-process (no real port needed) and, when used in a `with` block, runs the
app's lifespan - so the engine and the TTL cleaner really start and stop.

Settings are read from the environment when app.config is first imported, so
we set them BEFORE importing the app. Small capacity and a fast cleanup
interval keep the tests short.
"""
import os
import sys
from pathlib import Path

import pytest

os.environ["NODE_ID"] = "test-node"
os.environ["CLUSTER_SECRET"] = "test-secret"
os.environ["CACHE_CAPACITY"] = "3"
os.environ["TTL_CLEANUP_INTERVAL"] = "0.05"

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "cache_node"))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

AUTH = {"X-Cluster-Secret": "test-secret"}


@pytest.fixture
def client():
    """Authenticated client against a fresh, empty node (new engine per test)."""
    with TestClient(app, headers=AUTH) as c:
        yield c


@pytest.fixture
def anon_client():
    """Client that sends NO secret header."""
    with TestClient(app) as c:
        yield c
