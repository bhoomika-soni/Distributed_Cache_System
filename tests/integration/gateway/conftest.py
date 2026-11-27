"""Fixtures for end-to-end gateway tests (real processes, real HTTP).

Each fixture is module-scoped, so the processes start once per file. Tests use
unique keys so they do not interfere with each other.
"""
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from harness import HangingServer, free_port, run_gateway, running_cluster  # noqa: E402


@pytest.fixture(scope="module")
def cluster():
    """Two real cache nodes + a gateway."""
    with running_cluster(node_count=2) as c:
        yield c


@pytest.fixture
def client(cluster):
    with httpx.Client(base_url=cluster.gateway_url, timeout=10) as c:
        yield c


@pytest.fixture(scope="module")
def dead_node_client():
    """Gateway whose only node does not exist (nothing listens on that port)."""
    # Short timeout: on Windows a closed port can take the full timeout to fail.
    nodes = {"node-1": f"http://127.0.0.1:{free_port()}"}
    with run_gateway(nodes, request_timeout=0.5, max_retries=2) as gw:
        with httpx.Client(base_url=gw.url, timeout=15) as c:
            yield c


@pytest.fixture(scope="module")
def hanging_node():
    server = HangingServer()
    yield server
    server.close()


@pytest.fixture(scope="module")
def hanging_client(hanging_node):
    """Gateway whose node accepts connections but never replies (0.3s timeout, 2 retries)."""
    nodes = {"node-1": f"http://127.0.0.1:{hanging_node.port}"}
    with run_gateway(nodes, request_timeout=0.3, max_retries=2) as gw:
        with httpx.Client(base_url=gw.url, timeout=15) as c:
            yield c


@pytest.fixture(scope="module")
def wrong_secret_client(cluster):
    """Gateway configured with a different CLUSTER_SECRET than the node."""
    nodes = {"node-1": cluster.node_urls["node-1"]}
    with run_gateway(nodes, secret="not-the-nodes-secret") as gw:
        with httpx.Client(base_url=gw.url, timeout=15) as c:
            yield c
